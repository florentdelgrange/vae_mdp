import os
from typing import Tuple
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Reshape, Dense, Lambda
from tensorflow.keras.utils import Progbar

from util.pdf import logistic, binary_concrete
from tensorflow.python import debug as tf_debug


epsilon = 1e-12
max_val = 1e9
min_val = -1e9
max_log_val = np.log(1e9)
min_log_val = np.log(epsilon)


class VariationalMDPStateAbstraction(Model):
    def __init__(self,
                 state_shape: Tuple[int, ...],
                 action_shape: Tuple[int],
                 reward_shape: Tuple[int],
                 label_shape: Tuple[int],
                 encoder_network: Model,
                 transition_network: Model,
                 reward_network: Model,
                 decoder_network: Model,
                 latent_state_size: int = 16,
                 temperature_1: float = 2 / 3,
                 temperature_2: float = 1 / 2,
                 nb_gaussian_posteriors: int = 3,
                 action_pre_processing_network: Model = None,
                 state_pre_processing_network: Model = None,
                 state_post_processing_network: Model = None,
                 name: str = 'vae_mdp'):
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
            Dense(units=latent_state_size, activation='linear', name='transition_logistic_locations')(transition)
        self.transition_network = Model(inputs=[latent_state, action], outputs=transition_output_layer,
                                        name="transition_network")

        # Reward network
        next_latent_state = Input(shape=(latent_state_size,), name="next_latent_state")
        action_layer_2 = action if not action_pre_processing_network else action_pre_processing_network(action)
        reward_network_input = \
            Concatenate(name="reward_network_input")([latent_state, action_layer_2, next_latent_state])
        reward_1 = reward_network(reward_network_input)
        reward_mean = Dense(np.prod(reward_shape), activation='linear', name='reward_mean_0')(reward_1)
        reward_mean = Reshape(reward_shape, name='reward_mean')(reward_mean)
        reward_log_var = Dense(np.prod(reward_shape), activation='linear', name='reward_log_var_0')(reward_1)
        reward_log_var = Reshape(reward_shape, name='reward_log_var')(reward_log_var)
        self.reward_network = Model(inputs=[latent_state, action, next_latent_state],
                                    outputs=[reward_mean, reward_log_var], name='reward_network')

        # Reconstruction network
        # inputs are binary concrete random variables, outputs are given in parameter
        decoder = decoder_network(next_latent_state)
        if state_post_processing_network is not None:
            decoder = state_post_processing_network(decoder)
        # 1 mean per dimension, nb Normal Gaussian
        decoder_output_mean = \
            Dense(nb_gaussian_posteriors * np.prod(state_shape), activation='linear', name='GMM_means_0')(decoder)
        decoder_output_mean = \
            Reshape((nb_gaussian_posteriors,) + state_shape, name="GMM_means")(decoder_output_mean)
        # 1 var per dimension, nb Normal Gaussian
        decoder_output_log_var = \
            Dense(nb_gaussian_posteriors * np.prod(state_shape), activation='linear', name='GMM_log_vars_0')(decoder)
        decoder_output_log_var = \
            Reshape((nb_gaussian_posteriors,) + state_shape, name="GMM_log_vars")(decoder_output_log_var)
        # prior over Normal Gaussian
        decoder_prior = Dense(nb_gaussian_posteriors, activation='softmax', name="GMM_priors")(decoder)
        self.reconstruction_network = Model(inputs=next_latent_state,
                                            outputs=[decoder_output_mean, decoder_output_log_var, decoder_prior],
                                            name='reconstruction_network')

        vae_input = [Input(shape=(2,) + self.state_shape, name="incident_states"),
                     Input(shape=(2,) + self.action_shape, name="actions"),
                     Input(shape=(2,) + self.reward_shape, name="rewards"),
                     Input(shape=(2,) + self.state_shape, name="next_states"),
                     Input(shape=(2,) + self.label_shape, name="labels")]
        self._set_inputs(vae_input)
        self._vae = None

    @tf.function
    def sample_logistic(self, log_alpha: tf.Tensor):
        """
        Reparameterization trick for sampling binary concrete random variables.
        The logistic random variable with location log_alpha is a binary concrete random variable before applying the
        sigmoid function.
        """
        U = tf.random.uniform(shape=tf.shape(log_alpha), minval=epsilon, name="sample_U")
        L = tf.math.log(U) - tf.math.log(tf.ones(shape=tf.shape(log_alpha), dtype=log_alpha.dtype) - U,
                                         name="log-logistic_noise")
        return logistic.sample(self.temperature[0], log_alpha, L)

    def encode(self, state, action, reward, label, state_prime) -> tfp.distributions.Distribution:
        """
        Encode the sample (s, a, r, l, s') into a Bernoulli probability distribution over binary latent states z
        Note: the Bernoulli distribution is constructed via the Binary Concrete distribution learned by the encoder with
              a temperature that converges to 0.
        """
        [log_alpha, label] = self.encoder_network([state, action, reward, label, state_prime])
        return tfp.distributions.Bernoulli(
            probs=tf.concat([tf.exp(log_alpha) / (tf.ones(tf.shape(log_alpha)) + tf.exp(log_alpha)), label], axis=-1))

    def decode(self, latent_state) -> tfp.distributions.Distribution:
        """
        Decode a binary latent state into a probability distribution over original states of the MDP, modeled by a GMM
        """
        [reconstruction_mean, reconstruction_log_var, reconstruction_prior_components] = \
            self.reconstruction_network(latent_state)
        return tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=reconstruction_prior_components),
            components_distribution=tfp.distributions.MultivariateNormalDiag(
                loc=reconstruction_mean, scale_diag=tf.math.exp(reconstruction_log_var)))

    def latent_transition_probability_distribution(self, latent_state, action) -> tfp.distributions.Distribution:
        """
        Retrieves a Bernoulli probability distribution P(z'|z, a) over successor latent states z', given a binary
        latent state z and action a.
        Note: the Bernoulli distribution is constructed via the Binary Concrete distribution learned by the encoder with
              a temperature that converges to 0.
        """
        log_alpha = self.transition_network([latent_state, action])
        return tfp.distributions.Bernoulli(
            probs=(tf.exp(log_alpha) / (tf.ones(tf.shape(latent_state)) + tf.exp(log_alpha))))

    def reward_probability_distribution(self, latent_state, action, next_latent_state) \
            -> tfp.distributions.Distribution:
        """
        Retrieves a probability distribution P(r|z, a, z') over rewards obtained when the latent transition z -> z' is
        encountered when action a is chosen
        """
        [reward_mean, reward_log_var] = self.reward_network([latent_state, action, next_latent_state])
        return tfp.distributions.MultivariateNormalDiag(loc=reward_mean, scale_diag=tf.math.exp(reward_log_var))

    @property
    def vae(self) -> Model:
        """
        VAE-MDP network
        """
        # Make the VAE input time distributed so that the encoder provides z, z' with
        # z ~ Q(z_t|s_{t-1},a_{t-1},r_{t-1}, s_t, l_t) and z' ~ Q(z_{t+1}|s_t, a_t, r_t, s_{t+1}, l_{t+1})
        # Time distributed layers do not support multiple outputs or inputs.
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
def compute_loss(vae_mdp: VariationalMDPStateAbstraction, x,
                 reconstruction_state_observer=None, reconstruction_reward_observer=None, kl_observer=None):
    # inputs are assumed to have shape
    # x = [(?, 2, state_shape), (?, 2, action_shape), (?, 2, reward_shape), (?, 2, state_shape), (?, 2, label_shape)]
    s_0, a_0, r_0, _, l_1 = (input[:, 0, :] for input in x)
    s_1, a_1, r_1, s_2, l_2 = (input[:, 1, :] for input in x)

    [log_alpha, label] = vae_mdp.encoder_network([s_0, a_0, r_0, s_1, l_1])
    [log_alpha_prime, label_prime] = vae_mdp.encoder_network([s_1, a_1, r_1, s_2, l_2])

    # z, z' ~ BinConcrete = sigmoid(BinConcreteLogistic)
    logistic_z, logistic_z_prime = vae_mdp.sample_logistic(log_alpha), vae_mdp.sample_logistic(log_alpha_prime)
    z = tf.concat([tf.sigmoid(logistic_z), label], axis=-1, name="concat_z")
    z_prime = tf.concat([tf.sigmoid(logistic_z_prime), label_prime], axis=-1, name="concat_z_prime")

    # binary-concrete log-logistic probability Q(logistic_z'|s_1, a_1, r_1, s_2, l_2), logistic_z' ~ Logistic(alpha')
    log_q_z_prime = \
        binary_concrete.log_logistic_density(vae_mdp.temperature[0], log_alpha_prime,
                                             tf.math.log, tf.math.exp, tf.math.log1p)(logistic_z_prime)

    # change label l'=1 to 100 and label l'=0 to -100 so that
    # sigmoid(logistic_z'[l']) = 1 if l'=1 and sigmoid(logistic_z'[l']) = 0 if l'=0
    logistic_label_prime = ((label_prime * 2) - tf.ones(tf.shape(label_prime))) * 1e2
    logistic_z_prime = tf.concat([logistic_z_prime, logistic_label_prime], axis=-1, name="concat_logistic_z_prime")

    # logistic log probability P(logistic_z'|z, a_1)
    log_p_z_prime = logistic.log_density(vae_mdp.temperature[1], vae_mdp.transition_network([z, a_1]),
                                         tf.math.log, tf.math.exp, tf.math.log1p)(logistic_z_prime)

    # Normal log-probability P(r_1 | z, a_1, z')
    reward_distribution = vae_mdp.reward_probability_distribution(z, a_1, z_prime)
    log_p_rewards = reward_distribution.log_prob(r_1)

    # Reconstruction P(s_2 | z')
    state_distribution = vae_mdp.decode(z_prime)
    log_p_reconstruction = state_distribution.log_prob(s_2)

    # the probability of encoding labels into z' is one
    log_q_z_prime = tf.concat([log_q_z_prime, tf.zeros(shape=tf.shape(label_prime))], axis=-1, name="q_z_prime_final")

    kl_terms = tf.reduce_sum(log_q_z_prime - log_p_z_prime, axis=1)

    if reconstruction_state_observer:
        reconstruction_state_observer(
            tf.reduce_sum(tf.square(reward_distribution.sample(sample_shape=tf.shape(s_2)) - s_2), axis=1))
    if reconstruction_reward_observer:
        reconstruction_reward_observer(
            tf.reduce_sum(tf.square(reward_distribution.sample(sample_shape=tf.shape(r_1)) - r_1), axis=1))
    if kl_observer:
        kl_observer(kl_terms)

    return - tf.reduce_mean(
        log_p_rewards + log_p_reconstruction - kl_terms
    )


@tf.function
def compute_apply_gradients(vae_mdp: VariationalMDPStateAbstraction, x, optimizer,
                            reconstruction_state_observer=None,
                            reconstruction_reward_observer=None,
                            kl_observer=None):
    with tf.GradientTape() as tape:
        loss = compute_loss(vae_mdp, x, reconstruction_state_observer, reconstruction_reward_observer, kl_observer)
    gradients = tape.gradient(loss, vae_mdp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae_mdp.trainable_variables))
    return loss


def train(vae_mdp: VariationalMDPStateAbstraction,
          dataset: tf.data.Dataset,
          epochs: int = 8,
          batch_size: int = 32,
          optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-4),
          checkpoint: tf.train.Checkpoint = None,
          manager: tf.train.CheckpointManager = None,
          log_interval: int = 80,
          dataset_size: int = None):
    import datetime

    if checkpoint is not None and manager is not None:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    global_step = checkpoint.save_counter
    print("Step: {}".format(global_step.numpy()))

    for epoch in range(epochs):
        progressbar = Progbar(target=dataset_size,
                              stateful_metrics=['steps', 'ELBO', 'State reconstruction MSE',
                                                'Reward reconstruction MSE', 'KL terms'],
                              interval=0.1)
        loss = tf.keras.metrics.Mean()
        reward_observer = tf.keras.metrics.Mean()
        state_observer = tf.keras.metrics.Mean()
        kl_observer = tf.keras.metrics.Mean()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        for x in dataset.batch(batch_size, drop_remainder=True):
            gradients = compute_apply_gradients(vae_mdp, x, optimizer, state_observer, reward_observer, kl_observer)
            loss(gradients)

            progressbar.add(batch_size, values=[('steps', global_step.numpy()),
                                                ('ELBO', - loss.result()),
                                                ('State reconstruction MSE', state_observer.result()),
                                                ('Reward reconstruction MSE', reward_observer.result()),
                                                ('KL terms', kl_observer.result())])

            if checkpoint is not None and manager is not None:
                global_step.assign_add(1)

            if global_step.numpy() % log_interval == 0:
                manager.save()
                with train_summary_writer.as_default():
                    tf.summary.scalar('ELBO', - loss.result(), step=global_step.numpy())
                    tf.summary.scalar('State reconstruction MSE', state_observer.result(),
                                      step=global_step.numpy())
                    tf.summary.scalar('Reward reconstruction MSE', reward_observer.result(),
                                      step=global_step.numpy())
                    tf.summary.scalar('KL terms', kl_observer.result(), step=global_step.numpy())

        # vae_mdp.save(os.path.join(manager.directory, 'vae_state_abstraction_epoch_{}.hdf5'.format(epoch)))
