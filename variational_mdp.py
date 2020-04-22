from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K, Sequential
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Reshape, Dense, Lambda, \
    Conv2D, Conv2DTranspose, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
from util.pdf import logistic, binary_concrete
import numpy as np

epsilon = 1e-12
max_val = 1e9
min_val = -1e9


class VariationalMDPStateAbstraction(Model):
    def __init__(self, state_shape: Tuple[int, ...], action_shape: Tuple[int], reward_shape: Tuple[int],
                 label_shape: Tuple[int], encoder_network: Model, transition_network: Model, reward_network: Model,
                 decoder_network: Model, latent_state_size: int = 128, temperature_1: float = 2 / 3,
                 temperature_2: float = 1 / 2, nb_gaussian_posterior: int = 3,
                 action_pre_processing_network: Model = None, state_pre_processing_network: Model = None,
                 state_post_processing_network: Model = None, name: str = 'vae_mdp'):
        super(VariationalMDPStateAbstraction, self).__init__()
        self.temperature = [temperature_1, temperature_2]
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.latent_state_size = latent_state_size
        self.label_shape = label_shape

        if not (len(state_shape) == len(action_shape) == len(reward_shape) == len(label_shape)):
            if state_pre_processing_network is None:
                raise ValueError("states, actions, rewards and label dimensions should be the same. "
                                 "Please provide a state_pre_processing_network and state_post_processing_network"
                                 " if states have different dimensions.")
            if not (len(tuple(filter(lambda dim: dim is not None, state_pre_processing_network.output.shape))) ==
                    len(action_shape) == len(reward_shape) == len(label_shape) and
                    state_shape == tuple(filter(lambda dim: dim is not None, state_pre_processing_network.input.shape))
                    == tuple(filter(lambda dim: dim is not None, state_post_processing_network.output.shape))):
                raise ValueError("Please provide state_pre_processing_network and state_post_processing_network"
                                 " with correct input and output shapes.")

        input_state = Input(shape=state_shape, name="state")
        action = Input(shape=action_shape, name="action")
        reward = Input(shape=reward_shape, name="reward")
        input_next_state = Input(shape=state_shape, name="next_state")
        label = Input(shape=label_shape, name="label")

        state, next_state = input_state, input_next_state
        if state_pre_processing_network is not None:
            # the concatenate axis is 0 to stack the inputs afterwards
            concatenate_inputs = Concatenate(axis=0)([input_state, input_next_state])
            stack_inputs = Reshape((2,) + state_shape)(concatenate_inputs)
            # Apply the same pre-processing to both state inputs
            apply_pre_processing = \
                TimeDistributed(state_pre_processing_network, input_shape=(2,) + state_shape)(stack_inputs)
            state, next_state = apply_pre_processing[:, 0], apply_pre_processing[:, 1]

        # Encoder network
        encoder_input = Concatenate(name="encoder_input")([state, action, reward, next_state])
        encoder = encoder_network(encoder_input)
        log_alpha_layer = \
            Dense(units=latent_state_size - np.prod(label_shape), activation='linear', name='log_alpha')(encoder)
        label_layer = Model(inputs=label, outputs=label, name='label_layer')
        self.encoder_network = \
            Model(inputs=[input_state, action, reward, input_next_state, label],
                  outputs=[log_alpha_layer, label_layer.output], name='encoder')

        # Transition network
        # inputs are binary concrete random variables, outputs are locations of logistic distributions
        latent_state = Input(shape=(latent_state_size,), name="latent_state")
        action_layer_1 = action if not action_pre_processing_network \
            else clone_model(action_pre_processing_network)(action)
        transition_network_input = \
            Concatenate(name="transition_network_input")([latent_state, action_layer_1])
        transition = transition_network(transition_network_input)
        transition_output_layer = \
            Dense(units=latent_state_size, activation='linear', name='logistic_locations')(transition)
        self.transition_network = Model(inputs=[latent_state, action], outputs=transition_output_layer,
                                        name="transition_network")

        # Reward network
        next_latent_state = Input(shape=(latent_state_size,), name="next_latent_state")
        action_layer_2 = action if not action_pre_processing_network else action_pre_processing_network(action)
        reward_network_input = \
            Concatenate(name="reward_network_input")([latent_state, action_layer_2, next_latent_state])
        reward_1 = reward_network(reward_network_input)
        reward_mean = Dense(np.prod(reward_shape), activation='linear')(reward_1)
        reward_mean = Reshape(reward_shape, name='reward_mean')(reward_mean)
        reward_log_var = Dense(np.prod(reward_shape), activation='linear')(reward_1)
        reward_log_var = Reshape(reward_shape, name='reward_log_var')(reward_log_var)
        self.reward_network = Model(inputs=[latent_state, action, next_latent_state],
                                    outputs=[reward_mean, reward_log_var], name='reward_network')

        # Reconstruction network
        # inputs are binary concrete random variables, outputs are given in parameter
        decoder = decoder_network(next_latent_state)
        if state_post_processing_network is not None:
            decoder = state_post_processing_network(decoder)
        # 1 mean per dimension, nb Normal Gaussian
        decoder_output_mean = Dense(nb_gaussian_posterior * np.prod(state_shape), activation='linear')(decoder)
        decoder_output_mean = \
            Reshape((nb_gaussian_posterior,) + state_shape, name="GMM_means")(decoder_output_mean)
        # 1 var per dimension, nb Normal Gaussian
        decoder_output_log_var = Dense(nb_gaussian_posterior * np.prod(state_shape), activation='linear')(decoder)
        decoder_output_log_var = \
            Reshape((nb_gaussian_posterior,) + state_shape, name="GMM_log_vars")(decoder_output_log_var)
        # prior over Normal Gaussian
        decoder_prior = Dense(nb_gaussian_posterior, activation='softmax', name="GMM_priors")(decoder)
        self.reconstruction_network = Model(inputs=next_latent_state,
                                            outputs=[decoder_output_mean, decoder_output_log_var, decoder_prior],
                                            name='reconstruction_network')
        self._vae = None

    def sample_logistic(self, log_alpha: tf.Tensor):
        """
        Reparameterization trick for sampling binary concrete random variables.
        The logistic random variable with location log_alpha is a binary concrete random variable before applying the
        sigmoid function
        """
        batch = K.shape(log_alpha)[0]
        dim = K.int_shape(log_alpha)[1]
        U = K.random_uniform(shape=(batch, dim), minval=epsilon)
        L = K.log(U) - K.log(K.ones(shape=(batch, dim,)) - U)
        return logistic.sample(self.temperature[0], log_alpha, L)

    @property
    def vae(self) -> Model:
        """
        VAE-MDP network
        Make the VAE input time distributed so that the encoder provides z, z' with
        z ~ Q(z_t|s_{t-1},a_{t-1},r_{t-1}, s_t, l_t) and z' ~ Q(z_{t+1}|s_t, a_t, r_t, s_{t+1}, l_{t+1})
        Time distributed layers do not support multiple outputs or inputs.
        """
        if self._vae is None:
            vae_input = [Input(shape=(2,) + self.state_shape, name="incident_states"),
                         Input(shape=(2,) + self.action_shape, name="actions"),
                         Input(shape=(2,) + self.reward_shape, name="rewards"),
                         Input(shape=(2,) + self.state_shape, name="next_states"),
                         Input(shape=(2,) + self.label_shape, name="labels")]
            shapes = [self.state_shape, self.action_shape, self.reward_shape, self.state_shape, self.label_shape]
            flat_shapes = [np.prod(shape) for shape in shapes]
            indices = [sum(flat_shapes[:i + 1]) for i in range(len(flat_shapes))]
            vae_flat_input = \
                Concatenate(name="flat_input")([Reshape(target_shape=(2, shape,))(x)
                                                for x, shape in zip(vae_input, flat_shapes)])
            Q = TimeDistributed(Lambda(lambda x: Concatenate()(
                self.encoder_network([Reshape(target_shape=shapes[0])(x[:, : indices[0]]),
                                      Reshape(target_shape=shapes[1])(x[:, indices[0]: indices[1]]),
                                      Reshape(target_shape=shapes[2])(x[:, indices[1]: indices[2]]),
                                      Reshape(target_shape=shapes[3])(x[:, indices[2]: indices[3]]),
                                      Reshape(target_shape=shapes[4])(x[:, indices[3]:])])), name='encoder'),
                                input_shape=(2, sum(flat_shapes),))(vae_flat_input)
            log_alpha_layer_size = self.latent_state_size - np.prod(self.label_shape)
            logistic_z = Lambda(self.sample_logistic, name="logistic_z")(Q[:, 0, :log_alpha_layer_size])
            logistic_z_prime = Lambda(self.sample_logistic, name="logistic_z_prime")(Q[:, 1, :log_alpha_layer_size])

            z = Concatenate()([Lambda(K.sigmoid)(logistic_z), Q[:, 0, log_alpha_layer_size:]])
            z_prime = Concatenate()([Lambda(K.sigmoid)(logistic_z_prime), Q[:, 1, log_alpha_layer_size:]])
            P_transition = self.transition_network([z, vae_input[1][:, 1]])
            P_reward = self.reward_network([z, vae_input[1][:, 1], z_prime])
            P_state = self.reconstruction_network(z_prime)
            self._vae = Model(inputs=vae_input, outputs=P_state + [P_transition] + P_reward, name='vae')
        return self._vae

    def call(self, inputs, training=None, mask=None):
        return self.vae.call(inputs, training=training, mask=mask)


@tf.function
def compute_loss(vae_mdp: VariationalMDPStateAbstraction, x):
    # inputs are assumed to have shape
    # [(?, 2, state_shape), (?, 2, action_shape), (?, 2, reward_shape), (?, 2, state_shape), (?, 2, label_shape)]
    s_0, a_0, r_0, _, l_1 = (input[i][:, 0] for i, input in enumerate(x))
    s_1, a_1, r_1, s_2, l_2 = (input[i][:, 1] for i, input in enumerate(x))
    [log_alpha, label] = vae_mdp.encoder_network([s_0, a_0, r_0, s_1, l_1])
    [log_alpha_prime, label_prime] = vae_mdp.encoder_network([s_1, a_1, r_1, s_2, l_2])

    # z ~ sigmoid(Logistic(alpha)), z' ~ sigmoid(Logistic(alpha'))
    logistic_z, logistic_z_prime = vae_mdp.sample_logistic(log_alpha), vae_mdp.sample_logistic(log_alpha_prime)
    z = tf.concat([tf.sigmoid(logistic_z), label], axis=-1)
    z_prime = tf.concat([tf.sigmoid(logistic_z_prime), label_prime], axis=-1)

    # change label l'=1 to 100 and label l'=0 to -1000 so that
    # sigmoid(logistic_z'[label]) = 1 if l'=1 and sigmoid(logistic_z'[label]) = 0 if l'=0
    logistic_label_prime = tf.cond(tf.greater(label_prime, tf.zeros(vae_mdp.label_shape)),
                                   lambda x: x * 1e2, lambda x: x - 1e3)
    logistic_z_prime = tf.concat([logistic_z_prime, logistic_label_prime], axis=-1)

    # binary-concrete log-logistic probability Q(logistic_z'|s_1, a_1, r_1, s_2, l_2), logistic_z' ~ Logistic(alpha')
    log_q_z_prime = tf.clip_by_value(
        binary_concrete.log_logistic_density(vae_mdp.temperature[0], log_alpha_prime, logistic_z_prime,
                                             tf.math.log, tf.math.exp, tf.math.log1p),
        clip_value_min=min_val, clip_value_max=max_val)

    # logistic log probability P(z'|z, a_1)
    log_p_z_prime = tf.clip_by_value(
        logistic.log_density(vae_mdp.temperature[1], vae_mdp.transition_network([z, a_1]), logistic_z_prime,
                             tf.math.log, tf.math.exp, tf.math.log1p),
        clip_value_min=min_val, clip_value_max=max_val)

    # Normal log-probability P(r_1 | z, a_1, z')
    [reward_mean, reward_log_var] = vae_mdp.reward_network([z, a_1, z_prime])
    log_p_rewards = \
        tfp.distributions.MultivariateNormalDiag(loc=reward_mean, scale_diag=tf.math.exp(reward_log_var)).log_prob(r_1)

    # Reconstruction P(s_2 | z'), modeled by a GMM
    [reconstruction_mean, reconstruction_log_var, reconstruction_prior_components] = \
        vae_mdp.reconstruction_network(z_prime)
    log_p_reconstruction = \
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=reconstruction_prior_components),
            components_distribution=tfp.distributions.MultivariateNormalDiag(
                loc=reconstruction_mean, scale_diag=tf.math.exp(reconstruction_log_var))
        ).log_prob(s_2)

    return - tf.reduce_mean(
        log_p_rewards + log_p_reconstruction - tf.reduce_sum(log_q_z_prime - log_p_z_prime, axis=1)
    )


@tf.function
def compute_apply_gradients(vae_mdp: VariationalMDPStateAbstraction, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(vae_mdp, x)
    gradients = tape.gradient(loss, vae_mdp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae_mdp.trainable_variables))


if __name__ == '__main__':
    # Example
    state_dim = (4,)
    # state_dim = (128, 128, 3,)
    action_dim = (2,)
    reward_dim = (1,)
    label_dim = (3,)

    #  input_pre_processing_network = Input(shape=state_dim)
    #  pre_processing_network = Conv2D(filters=16, kernel_size=5, activation='relu', strides=(2, 2)) \
    #      (input_pre_processing_network)
    #  pre_processing_network = Conv2D(filters=32, kernel_size=3, activation='relu', strides=(2, 2)) \
    #      (pre_processing_network)
    #  pre_processing_network = BatchNormalization()(pre_processing_network)
    #  pre_processing_network = MaxPooling2D()(pre_processing_network)
    #  pre_processing_network = Flatten()(pre_processing_network)
    #  pre_processing_network = Model(inputs=input_pre_processing_network, outputs=pre_processing_network)

    # Encoder body
    encoder_input = Input(shape=(np.prod(state_dim) * 2 + np.prod(action_dim) + np.prod(reward_dim),))
    # x = Input(shape=(np.prod(tuple(filter(lambda dim: dim is not None, pre_processing_network.output.shape))) * 2 +
    #                 np.prod(action_dim) + np.prod(reward_dim),))
    q = Dense(32, activation='relu')(encoder_input)
    q = Dense(64, activation='relu')(q)
    q = Model(inputs=encoder_input, outputs=q, name="encoder_network_body")

    # Transition network body
    transition_input = Input(shape=(256,))
    p_t = Dense(128, activation='relu')(transition_input)
    p_t = Dense(128, activation='relu')(p_t)
    p_t = Model(inputs=transition_input, outputs=p_t, name="transition_network_body")

    # Reward network body
    p_r_input = Input(shape=(384,))
    p_r = Dense(128, activation='relu')(p_r_input)
    p_r = Dense(64, activation='relu')(p_r)
    p_r = Model(inputs=p_r_input, outputs=p_r, name="reward_network_body")

    # Decoder network body
    p_decoder_input = Input(shape=(128,))
    p_decode = Dense(64, activation='relu')(p_decoder_input)
    p_decode = Dense(32, activation='relu')(p_decode)
    p_decode = Model(inputs=p_decoder_input, outputs=p_decode, name="decoder_body")

    #  p_deconv_input = Input(tuple(filter(lambda dim: dim is not None, p_decode.output.shape)))
    #  p_deconv = Dense(units=np.prod(tuple(filter(lambda dim: dim is not None, pre_processing_network.output.shape))))\
    #      (p_deconv_input)
    #  p_deconv = Reshape((32, 32, 32))(p_deconv)
    #  p_deconv = \
    #      Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=(2, 2), padding='SAME')(p_deconv)
    #  p_deconv = \
    #      Conv2DTranspose(filters=16, kernel_size=5, activation='relu', strides=(2, 2), padding='SAME')(p_deconv)
    #  p_deconv = \
    #      Conv2DTranspose(filters=3, kernel_size=3, padding='SAME')(p_deconv)
    #  p_deconv = Model(inputs=p_deconv_input, outputs=p_deconv)

    action_processor = Sequential(name='action_processor')
    action_processor.add(Dense(32, activation='sigmoid', name="process_action"))
    action_processor.add(Dense(128, activation='sigmoid', name="process_action"))
    model = VariationalMDPStateAbstraction(state_dim, action_dim, reward_dim, label_dim, q, p_t, p_r, p_decode,
                                           action_pre_processing_network=action_processor)
    # state_pre_processing_network=pre_processing_network,
    # state_post_processing_network=p_deconv)
    plot_model(model.vae, dpi=300, expand_nested=True, show_shapes=True)
