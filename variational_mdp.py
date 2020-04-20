from typing import Tuple

from tensorflow.keras import backend as K, Sequential
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Reshape, Dense, Lambda, \
    Conv2D, Conv2DTranspose, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
from util.pdf import logistic
import numpy as np


class VariationalMDPStateAbstraction:
    def __init__(self, state_shape: Tuple[int, ...], action_shape: Tuple[int], reward_shape: Tuple[int],
                 label_shape: Tuple[int], encoder_network: Model, transition_network: Model, reward_network: Model,
                 decoder_network: Model, latent_state_size: int = 128,
                 temperature_1: float = 2 / 3, temperature_2: float = 1 / 2, nb_gaussian_posterior: int = 3,
                 action_pre_processing_network: Model = None, state_pre_processing_network: Model = None,
                 state_post_processing_network: Model = None):
        self.temperature = [temperature_1, temperature_2]

        if not (len(state_shape) == len(action_shape) == len(reward_shape) == len(label_shape)):
            if state_pre_processing_network is None:
                raise ValueError("states, actions, rewards and label dimensions should be the same. "
                                 "Please provide a state_pre_processing_network and state_post_processing_network"
                                 " if states have different dimensions.")
        if not (len(tuple(filter(lambda dim: dim is not None, state_pre_processing_network.output.shape))) ==
                len(action_shape) == len(reward_shape) == len(label_shape) and
                state_shape == tuple(filter(lambda dim: dim is not None, state_pre_processing_network.input.shape)) ==
                tuple(filter(lambda dim: dim is not None, state_post_processing_network.output.shape))):
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
        self.encoder_network.summary()

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
        self.transition_network.summary()

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
        self.reward_network.summary()

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
        self.reconstruction_network.summary()

        # VAE-MDP network
        # Make the VAE input time distributed so that the encoder provides z, z' with
        # z ~ Q(z_t|s_{t-1},a_{t-1},r_{t-1}, s_t, l_t) and z' ~ Q(z_{t+1}|s_t, a_t, r_t, s_{t+1}, l_{t+1})
        # Time distributed layers do not support multiple outputs or inputs.
        vae_input = [Input(shape=(2,) + state_shape, name="incident_states"),
                     Input(shape=(2,) + action_shape, name="actions"),
                     Input(shape=(2,) + reward_shape, name="rewards"),
                     Input(shape=(2,) + state_shape, name="next_states"),
                     Input(shape=(2,) + label_shape, name="labels")]
        shapes = [state_shape, action_shape, reward_shape, state_shape, label_shape]
        flat_shapes = [np.prod(shape) for shape in shapes]
        indices = [sum(flat_shapes[:i + 1]) for i in range(len(flat_shapes))]
        vae_flat_input = \
            Concatenate(name="flat_input")([Reshape(target_shape=(2, shape, ))(x)
                                            for x, shape in zip(vae_input, flat_shapes)])
        Q = TimeDistributed(Lambda(lambda x: Concatenate()(
            self.encoder_network([Reshape(target_shape=shapes[0])(x[:, : indices[0]]),
                                  Reshape(target_shape=shapes[1])(x[:, indices[0]: indices[1]]),
                                  Reshape(target_shape=shapes[2])(x[:, indices[1]: indices[2]]),
                                  Reshape(target_shape=shapes[3])(x[:, indices[2]: indices[3]]),
                                  Reshape(target_shape=shapes[4])(x[:, indices[3]:])])), name='encoder'),
                            input_shape=(2, sum(flat_shapes),))(vae_flat_input)

        # reparameterization trick for sampling binary concrete random variables
        # The logistic random variable with location log_alpha is a binary concrete random variable before applying the
        # sigmoid function
        def sample_logistic(log_alpha):
            batch = K.shape(log_alpha)[0]
            dim = K.int_shape(log_alpha)[1]
            U = K.random_uniform(shape=(batch, dim), minval=1e-12)
            L = K.log(U) - K.log(K.ones(shape=(batch, dim,)) - U)
            return logistic.sample(temperature_1, log_alpha, L)

        log_alpha_layer_size = latent_state_size - np.prod(label_shape)
        logistic_z = Lambda(sample_logistic, name="logistic_z")(Q[:, 0, :log_alpha_layer_size])
        logistic_z_prime = Lambda(sample_logistic, name="logistic_z_prime")(Q[:, 1, :log_alpha_layer_size])

        z = Concatenate()([Lambda(K.sigmoid)(logistic_z), Q[:, 0, log_alpha_layer_size:]])
        z_prime = Concatenate()([Lambda(K.sigmoid)(logistic_z_prime), Q[:, 1, log_alpha_layer_size:]])
        P_transition = self.transition_network([z, vae_input[1][:, 1, :]])
        P_reward = self.reward_network([z, vae_input[1][:, 1, :], z_prime])
        P_state = self.reconstruction_network(z_prime)
        self.vae = \
            Model(inputs=vae_input,
                  outputs=P_state + [P_transition] + P_reward,
                  name='vae_mdp')
        self.vae.summary()


if __name__ == '__main__':
    # Example
    # state_dim = (4, )
    state_dim = (128, 128, 3,)
    action_dim = (2,)
    reward_dim = (1,)
    label_dim = (3,)

    input_pre_processing_network = Input(shape=state_dim)
    pre_processing_network = Conv2D(filters=16, kernel_size=5, activation='relu', strides=(2, 2)) \
        (input_pre_processing_network)
    pre_processing_network = Conv2D(filters=32, kernel_size=3, activation='relu', strides=(2, 2)) \
        (pre_processing_network)
    pre_processing_network = BatchNormalization()(pre_processing_network)
    pre_processing_network = MaxPooling2D()(pre_processing_network)
    pre_processing_network = Flatten()(pre_processing_network)
    pre_processing_network = Model(inputs=input_pre_processing_network, outputs=pre_processing_network)

    # Encoder body
    # x = Input(shape=(np.prod(state_dim) * 2 + np.prod(action_dim) + np.prod(reward_dim),))
    x = Input(shape=(np.prod(tuple(filter(lambda dim: dim is not None, pre_processing_network.output.shape))) * 2 +
                     np.prod(action_dim) + np.prod(reward_dim),))
    q = Dense(32, activation='relu')(x)
    q = Dense(64, activation='relu')(q)
    q = Model(inputs=x, outputs=q, name="encoder_network_body")

    # Transition network body
    x = Input(shape=(256,))
    p_t = Dense(128, activation='relu')(x)
    p_t = Dense(128, activation='relu')(p_t)
    p_t = Model(inputs=x, outputs=p_t, name="transition_network_body")

    # Reward network body
    x = Input(shape=(384,))
    p_r = Dense(128, activation='relu')(x)
    p_r = Dense(64, activation='relu')(p_r)
    p_r = Model(inputs=x, outputs=p_r, name="reward_network_body")

    # Decoder network body
    x = Input(shape=(128,))
    p_decode = Dense(64, activation='relu')(x)
    p_decode = Dense(32, activation='relu')(p_decode)
    p_decode = Model(inputs=x, outputs=p_decode, name="decoder_body")

    p_deconv_input = Input(tuple(filter(lambda dim: dim is not None, p_decode.output.shape)))
    p_deconv = Dense(units=np.prod(tuple(filter(lambda dim: dim is not None, pre_processing_network.output.shape))))(p_deconv_input)
    p_deconv = Reshape((32, 32, 32))(p_deconv)
    p_deconv = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=(2, 2), padding='SAME')(p_deconv)
    p_deconv = Conv2DTranspose(filters=16, kernel_size=5, activation='relu', strides=(2, 2), padding='SAME')(p_deconv)
    p_deconv = Conv2DTranspose(filters=3, kernel_size=3, padding='SAME')(p_deconv)
    p_deconv = Model(inputs=p_deconv_input, outputs=p_deconv)

    action_processor = Sequential(name='action_processor')
    action_processor.add(Dense(32, activation='sigmoid', name="process_action"))
    action_processor.add(Dense(128, activation='sigmoid', name="process_action"))
    model = VariationalMDPStateAbstraction(state_dim, action_dim, reward_dim, label_dim, q, p_t, p_r, p_decode,
                                           action_pre_processing_network=action_processor,
                                           state_pre_processing_network=pre_processing_network,
                                           state_post_processing_network=p_deconv)
    plot_model(model.vae, dpi=300, expand_nested=True, show_shapes=True)
