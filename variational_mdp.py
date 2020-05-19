import os
from typing import Tuple, Optional, List, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Reshape, Dense
from tensorflow.keras.utils import Progbar

debug = True
if debug:
   tf.debugging.enable_check_numerics()

epsilon = 1e-12
max_val = 1e9
min_val = -1e9
max_log_val = np.log(1e9)
min_log_val = np.log(epsilon)


class VariationalMarkovDecisionProcess(Model):
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
                 temperature_1_decay_rate: float = 0,
                 temperature_2_decay_rate: float = 0,
                 mixture_components: int = 3,
                 action_pre_processing_network: Model = None,
                 state_pre_processing_network: Model = None,
                 state_post_processing_network: Model = None,
                 pre_loaded_model: bool = False,
                 debug: bool = False):
        super(VariationalMarkovDecisionProcess, self).__init__()
        self.temperature = [tf.cast(temperature_1, 'float32'), tf.cast(temperature_2, 'float32')]
        self.decay_temperature = [temperature_1_decay_rate, temperature_2_decay_rate]
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.latent_state_size = latent_state_size
        self.label_shape = label_shape
        self.mixture_components = mixture_components

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

        if not pre_loaded_model:
            # Encoder network
            encoder_input = Concatenate(name="encoder_input")([state, action, reward, next_state])
            encoder = encoder_network(encoder_input)
            log_alpha_layer = \
                Dense(units=latent_state_size - np.prod(label_shape), activation=None, name='log_alpha')(encoder)
            self.encoder_network = \
                Model(inputs=[input_state, action, reward, input_next_state, label],
                      outputs=log_alpha_layer, name='encoder')

            # Transition network
            # inputs are binary concrete random variables, outputs are locations of logistic distributions
            latent_state = Input(shape=(latent_state_size,), name="latent_state")
            action_layer_1 = action if not action_pre_processing_network \
                else clone_model(action_pre_processing_network)(action)
            transition_network_input = \
                Concatenate(name="transition_network_input")([latent_state, action_layer_1])
            transition = transition_network(transition_network_input)
            transition_output_layer = \
                Dense(units=latent_state_size, activation=None, name='transition_logistic_locations')(transition)
            self.transition_network = Model(inputs=[latent_state, action], outputs=transition_output_layer,
                                            name="transition_network")

            # Reward network
            next_latent_state = Input(shape=(latent_state_size,), name="next_latent_state")
            action_layer_2 = action if not action_pre_processing_network else action_pre_processing_network(action)
            reward_network_input = \
                Concatenate(name="reward_network_input")([latent_state, action_layer_2, next_latent_state])
            reward_1 = reward_network(reward_network_input)
            reward_mean = Dense(units=np.prod(reward_shape), activation=None, name='reward_mean_0')(reward_1)
            reward_mean = Reshape(reward_shape, name='reward_mean')(reward_mean)
            reward_log_covar = Dense(units=np.prod(reward_shape),
                                     activation=None,
                                     name='reward_log_diag_covar_0')(reward_1)
            reward_log_covar = Reshape(reward_shape, name='reward_log_diag_covar')(reward_log_covar)
            self.reward_network = Model(inputs=[latent_state, action, next_latent_state],
                                        outputs=[reward_mean, reward_log_covar], name='reward_network')

            # Reconstruction network
            # inputs are binary concrete random variables, outputs are given in parameter
            decoder = decoder_network(next_latent_state)
            if state_post_processing_network is not None:
                decoder = state_post_processing_network(decoder)
            # 1 mean per dimension, nb Normal Gaussian
            decoder_output_mean = \
                Dense(units=mixture_components * np.prod(state_shape), activation=None, name='GMM_means_0')(decoder)
            decoder_output_mean = \
                Reshape((mixture_components,) + state_shape, name="GMM_means")(decoder_output_mean)
            # n diagonal co-variance matrices
            decoder_output_log_covar = Dense(
                units=mixture_components * np.prod(state_shape),
                activation=None,
                name='GMM_log_diag_covar_0'
            )(decoder)
            decoder_output_log_covar = \
                Reshape((mixture_components,) + state_shape, name="GMM_log_diag_covar")(decoder_output_log_covar)
            # number of Normal Gaussian forming the mixture model
            decoder_prior = Dense(units=mixture_components, activation='softmax', name="GMM_priors")(decoder)
            self.reconstruction_network = Model(inputs=next_latent_state,
                                                outputs=[decoder_output_mean, decoder_output_log_covar, decoder_prior],
                                                name='reconstruction_network')
        else:
            self.encoder_network = encoder_network
            self.transition_network = transition_network
            self.reward_network = reward_network
            self.reconstruction_network = decoder_network

        vae_input = [Input(shape=(2,) + self.state_shape, name="incident_states"),
                     Input(shape=(2,) + self.action_shape, name="actions"),
                     Input(shape=(2,) + self.reward_shape, name="rewards"),
                     Input(shape=(2,) + self.state_shape, name="next_states"),
                     Input(shape=(2,) + self.label_shape, name="labels")]
        self._set_inputs(vae_input)

        self._observers = []
        self.debug = debug
        self.eval = False  # flag used for testing

    def attach_observers(self, observers: List[Callable]):
        self._observers.extend(observers)

    def detach_observers(self):
        self._observers = []

    @tf.function
    def sample_logistic(self, temperature: float, log_alpha: tf.Tensor):
        """
        Reparameterization trick for sampling binary concrete random variables.
        The logistic random variable with location log_alpha is a binary concrete random variable before applying the
        sigmoid function.
        """
        L = logistic_noise(shape=tf.shape(log_alpha))
        return (L + log_alpha) / temperature

    @tf.function
    def sample_bernoulli(self, log_alpha: tf.Tensor):
        L = logistic_noise(shape=tf.shape(log_alpha))
        return tf.map_fn(lambda x: 1 if x + log_alpha >= 0 else 0, L)

    def encode(self, state, action, reward, state_prime, label) -> tfp.distributions.Distribution:
        """
        Encode the sample (s, a, r, l, s') into a Bernoulli probability distribution over binary latent states z
        Note: the Bernoulli distribution is constructed via the Binary Concrete distribution learned by the encoder with
              a temperature that converges to 0.
        """
        log_alpha = self.encoder_network([state, action, reward, state_prime, label])
        return tfp.distributions.Bernoulli(probs=tf.concat([tf.sigmoid(log_alpha), label], axis=-1))

    def decode(self, latent_state) -> tfp.distributions.Distribution:
        """
        Decode a binary latent state into a probability distribution over original states of the MDP, modeled by a GMM
        """
        [reconstruction_mean, reconstruction_log_diag_covar, reconstruction_prior_components] = \
            self.reconstruction_network(latent_state)
        if self.mixture_components == 1:
            return tfp.distributions.MultivariateNormalDiag(
                loc=reconstruction_mean[0],
                scale_diag=tf.exp(reconstruction_log_diag_covar[0]),
                allow_nan_stats=(not self.debug),
                validate_args=self.debug
            )
        else:
            return tfp.distributions.MixtureSameFamily(
                mixture_distribution=tfp.distributions.Categorical(probs=reconstruction_prior_components),
                components_distribution=tfp.distributions.MultivariateNormalDiag(
                    loc=reconstruction_mean,
                    scale_diag=tf.exp(reconstruction_log_diag_covar),
                    allow_nan_stats=(not self.debug)),
                validate_args=self.debug)

    def latent_transition_probability_distribution(self, latent_state, action) -> tfp.distributions.Distribution:
        """
        Retrieves a Bernoulli probability distribution P(z'|z, a) over successor latent states z', given a binary
        latent state z and action a.
        Note: the Bernoulli distribution is constructed via the Binary Concrete distribution learned by the encoder with
              a temperature that converges to 0.
        """
        log_alpha = self.transition_network([latent_state, action])
        return tfp.distributions.Bernoulli(probs=(tf.sigmoid(log_alpha)))

    def reward_probability_distribution(self, latent_state, action, next_latent_state) \
            -> tfp.distributions.Distribution:
        """
        Retrieves a probability distribution P(r|z, a, z') over rewards obtained when the latent transition z -> z' is
        encountered when action a is chosen
        """
        [reward_mean, reward_log_diag_covar] = self.reward_network([latent_state, action, next_latent_state])
        return tfp.distributions.MultivariateNormalDiag(loc=reward_mean,
                                                        scale_diag=tf.math.exp(reward_log_diag_covar),
                                                        allow_nan_stats=(not self.debug),
                                                        validate_args=self.debug)

    def decay_temperatures(self, step: int = 0):
        for i in (0, 1):
            if self.decay_temperature[i] != 0:
                self.temperature[i] = self.temperature[i] * (1 - self.decay_temperature[i]) if step == 0 \
                    else self.temperature[i] * (1 - self.decay_temperature[i]) ** step

    def call(self, inputs, training=None, mask=None):
        # inputs are assumed to have shape
        # [(?, 2, state_shape), (?, 2, action_shape), (?, 2, reward_shape), (?, 2, state_shape), (?, 2, label_shape)]
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        log_alpha = self.encoder_network([s_0, a_0, r_0, s_1, l_1])
        log_alpha_prime = self.encoder_network([s_1, a_1, r_1, s_2, l_2])

        # z ~ BinConcrete(temperature, alpha) = sigmoid(z ~ Logistic(temperature, log alpha))
        logistic_z = self.sample_logistic(self.temperature[0], log_alpha)
        logistic_z_prime = self.sample_logistic(self.temperature[0], log_alpha_prime)
        z = tf.concat([tf.sigmoid(logistic_z), l_1], axis=-1) if not self.eval else \
            tf.concat([self.sample_bernoulli(log_alpha), l_1], axis=-1)
        z_prime = tf.concat([tf.sigmoid(logistic_z_prime), l_2], axis=-1) if not self.eval else \
            tf.concat([self.sample_bernoulli(log_alpha_prime), l_2], axis=-1)

        # change label l_2=1 to 100 or label l_2=0 to -100 so that
        # sigmoid(logistic_z'[-1]) = 1 if l_2=1 and sigmoid(logistic_z'[-1]) = 0 if l_2=0
        logistic_l_2 = ((l_2 * 2) - 1) * 1e2
        logistic_z_prime = tf.concat([logistic_z_prime, logistic_l_2], axis=-1)
        log_alpha_prime = tf.concat([log_alpha_prime, logistic_l_2 * self.temperature[0]], axis=-1)

        if debug:
            tf.print(logistic_z, "sampled logistic z")
            tf.print(logistic_z_prime, "sampled logistic z'")
            tf.print(z, "sampled z")
            tf.print(z_prime, "sampled z'")

        # log logistic probability Q(logistic_z'|s_1, a_1, r_1, s_2, l_2),
        # with logistic_z' ~ Logistic(temperature_0, log alpha')
        log_q_z_prime = tfp.distributions.Logistic(
            scale=1 / self.temperature[0], loc=log_alpha_prime / self.temperature[0]
        ).log_prob(logistic_z_prime) if not self.eval else self.encode(s_1, a_1, r_1, s_2, l_2).log_prob(z_prime)

        if debug:
            tf.print(log_q_z_prime, "Log-logistic Q(z')")

        # log logistic probability P(logistic_z'|z, a_1)
        # prior distribution over logistic z' ~ Logistic(temperature_1, transition_locations)
        log_p_z_prime = tfp.distributions.Logistic(
            scale=1 / self.temperature[1], loc=self.transition_network([z, a_1]) / self.temperature[1]
        ).log_prob(logistic_z_prime) if not self.eval else \
            self.latent_transition_probability_distribution(z, a_1).log_prob(z_prime)

        if debug:
            tf.print(self.transition_network([z, a_1]), "log logistic locations P_transition")
            tf.print(log_p_z_prime, "log P(logistic z'|z, a)")

        # Normal log-probability P(r_1 | z, a_1, z')
        reward_distribution = self.reward_probability_distribution(z, a_1, z_prime)
        log_p_rewards = reward_distribution.log_prob(r_1)

        if debug:
            tf.print(log_p_rewards, "log P(r | z, a, z')")

        # Reconstruction P(s_2 | z')
        state_distribution = self.decode(z_prime)
        log_p_reconstruction = state_distribution.log_prob(s_2)

        if debug:
            tf.print(log_p_reconstruction, "log P(s' | z')")

        kl_terms = tf.reduce_sum(log_q_z_prime - log_p_z_prime, axis=1)

        observables = {
            'next_state': s_2,
            'reward': r_1,
            'kl_terms': kl_terms,
            'reconstruction_state_distribution': state_distribution,
            'reconstruction_reward_distribution': reward_distribution,
        }
        for observer in self._observers:
            observer(observables)

        return [log_p_reconstruction, log_p_rewards, kl_terms]


def logistic_noise(shape: tf.TensorShape):
    U = tf.random.uniform(shape=shape, minval=epsilon)
    return tf.math.log(U) - tf.math.log(1 - U)


@tf.function
def compute_loss(vae_mdp: VariationalMarkovDecisionProcess, x, eval=False):
    eval_flag = vae_mdp.eval
    vae_mdp.eval = eval

    log_p_states, log_p_rewards, kl_terms = vae_mdp(x)

    vae_mdp.eval = eval_flag

    return - tf.reduce_mean(
        log_p_states + log_p_rewards - kl_terms
    )


@tf.function
def compute_apply_gradients(vae_mdp: VariationalMarkovDecisionProcess, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(vae_mdp, x)
    gradients = tape.gradient(loss, vae_mdp.trainable_variables)

    if debug:
        for gradient, variable in zip(gradients, vae_mdp.trainable_variables):
            tf.print(gradient, "Gradient for {}".format(variable.name))

    optimizer.apply_gradients(zip(gradients, vae_mdp.trainable_variables))
    return loss


def load(tf_model_path: str) -> VariationalMarkovDecisionProcess:
    model = tf.keras.models.load_model(tf_model_path)
    vae_mdp = VariationalMarkovDecisionProcess(
        tuple(model.encoder_network.input[0].shape.as_list()[1:]),
        tuple(model.encoder_network.input[1].shape.as_list()[1:]),
        tuple(model.encoder_network.input[2].shape.as_list()[1:]),
        tuple(model.encoder_network.input[4].shape.as_list()[1:]),
        model.encoder_network,
        model.transition_network,
        model.reward_network,
        model.reconstruction_network,
        model.transition_network.inputs[0].shape[-1],
        pre_loaded_model=True)
    return vae_mdp


def train(vae_mdp: VariationalMarkovDecisionProcess,
          dataset: Optional[tf.data.Dataset] = None,
          dataset_generator: Optional[Callable[[], tf.data.Dataset]] = None,
          epochs: int = 8,
          batch_size: int = 128,
          optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-4),
          checkpoint: Optional[tf.train.Checkpoint] = None,
          manager: Optional[tf.train.CheckpointManager] = None,
          log_interval: int = 80,
          dataset_size: Optional[int] = None,
          log_name: str = 'vae',
          decay_period: int = 0,
          logs: bool = True,
          display_progressbar: bool = True,
          eval_ratio: float = 0.1):

    assert 0 <= eval_ratio < 1
    eval = False

    if (dataset is None and dataset_generator is None) or (dataset is not None and dataset_generator is not None):
        raise ValueError("Both a dataset and a dataset generator are passed, or neither.")

    if checkpoint is not None and manager is not None:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('logs/gradient_tape', log_name, current_time)
    if not os.path.exists(train_log_dir) and logs:
        os.makedirs(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir) if logs else None

    global_step = checkpoint.save_counter if checkpoint else tf.Variable(0)
    start_step = global_step.numpy()
    print("Step: {}".format(global_step.numpy()))
    if global_step.numpy() != 0 and decay_period != 0:
        vae_mdp.decay_temperatures(global_step.numpy() // decay_period)

    # Metrics
    def initialize_metrics(group_name: str = ''):
        return {
            group_name + 'loss': tf.keras.metrics.Mean(),
            group_name + 'state_mse': tf.keras.metrics.Mean(),
            group_name + 'reward_mse': tf.keras.metrics.Mean(),
            group_name + 'kl_terms': tf.keras.metrics.Mean(),
            group_name + 'mean_bits_used': tf.keras.metrics.Mean()
        }

    metrics = initialize_metrics()
    eval_metrics = initialize_metrics('eval_')
    for metric, condition, flag in ((metrics, False, ''), (eval_metrics, True, 'eval_')):

        def observe_state_reconstruction(observable, metric=metric, condition=condition, flag=flag):
            if 'next_state' and 'reconstruction_state_distribution' in observable and eval is condition:
                return metric[flag + 'state_mse'](
                    tf.reduce_sum(tf.square(observable['reconstruction_state_distribution'].sample(
                        sample_shape=tf.shape(observable['next_state'])) - observable['next_state']), axis=1))
            else:
                return None

        def observe_reward_reconstruction(observable, metric=metric, condition=condition, flag=flag):
            if 'reward' and 'reconstruction_reward_distribution' in observable and eval is condition:
                return metric[flag + 'reward_mse'](
                    tf.reduce_sum(tf.square(observable['reconstruction_reward_distribution'].sample(
                        sample_shape=tf.shape(observable['reward'])) - observable['reward']), axis=1))
            else:
                return None

        def observe_kl_terms(observable, metric=metric, condition=condition, flag=flag):
            if 'kl_terms' in observable and eval is condition:
                return metric[flag + 'kl_terms'](observable['kl_terms'])
            else:
                return None

        vae_mdp.attach_observers([observe_state_reconstruction, observe_reward_reconstruction, observe_kl_terms])

    for epoch in range(epochs):

        dataset_train_size, dataset_test_size = (1 - eval_ratio) * dataset_size, eval_ratio * dataset_size \
            if dataset_size is not None else (None, None)

        progressbar = Progbar(target=dataset_train_size,
                              stateful_metrics=['ELBO', 'state_MSE',
                                                'reward_MSE', 'KL_terms', 't1', 't2', 'bits_used'],
                              interval=0.1) if display_progressbar else None

        print("Epoch: {}/{}".format(epoch + 1, epochs))

        if dataset_generator is not None:
            dataset = dataset_generator()

        for x in dataset.batch(batch_size, drop_remainder=True):

            if not eval:
                gradients = compute_apply_gradients(vae_mdp, x, optimizer)
                metrics['loss'](gradients)

                metrics['mean_bits_used'](mean_latent_bits_used(vae_mdp, x))

                metrics_values = [('ELBO', - metrics['loss'].result()),
                                  ('state_MSE', metrics['state_mse'].result()),
                                  ('reward_MSE', metrics['reward_mse'].result()),
                                  ('KL_terms', metrics['kl_terms'].result()),
                                  ('bits_used', metrics['mean_bits_used'].result())]

                if decay_period != 0:
                    metrics_values.append(('t_1', vae_mdp.temperature[0].numpy()))
                    metrics_values.append(('t_2', vae_mdp.temperature[1].numpy()))

                if dataset_size is not None and display_progressbar and \
                        (global_step.numpy() - start_step) * batch_size < dataset_size * (epoch + 1):
                    progressbar.add(batch_size, values=metrics_values)

                if decay_period != 0 and global_step.numpy() % decay_period == 0:
                    vae_mdp.decay_temperatures()

            else:
                compute_loss(vae_mdp, x, eval=True)

            if checkpoint is not None and manager is not None:
                global_step.assign_add(1)

            if global_step.numpy() % log_interval == 0:
                if manager:
                    manager.save()
                if logs:
                    with train_summary_writer.as_default():
                        for key, values in metrics.items():
                            tf.summary.scalar(key, values, step=global_step.numpy())

            eval = (global_step.numpy() - start_step) * batch_size // (epoch + 1) >= dataset_size * (1 - eval_ratio)

        tf.saved_model.save(
            vae_mdp,
            os.path.join(manager.directory, os.pardir,
                         '{}_step{}_eval_ELBO{}'.format(log_name, global_step.numpy(),
                                                        - eval_metrics['eval_loss'].result())))
        print('\nEvaluation')
        for metric in metrics.values():
            metric.reset_states()
        for key, metric in eval_metrics.items():
            print(key, metric.result())
            metric.reset_states()

        if epoch == 0:
            dataset_size = (global_step.numpy() - start_step) * batch_size

        eval = False


def mean_latent_bits_used(vae_mdp: VariationalMarkovDecisionProcess, batch, eps=1e-3):
    """
    Compute the mean number of bits used in the latent space for the given dataset batch.
    This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
    """
    mean_bits_used = 0
    for i in (0, 1):
        s, a, r, s_prime, l_prime = (x[:, i, :] for x in batch)
        mean = tf.reduce_mean(vae_mdp.encode(s, a, r, s_prime, l_prime).mean(), axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used += tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()
    return mean_bits_used / 2
