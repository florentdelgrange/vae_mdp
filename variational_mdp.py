import os
from typing import Tuple, Optional, List, Callable, Dict, Union
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Reshape, Dense, Lambda
from tensorflow.keras.utils import Progbar

import tf_agents.policies.tf_policy
import tf_agents.agents.tf_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.utils import common

from util.io.dataset_generator import map_rl_trajectory_to_vae_input

tfd = tfp.distributions
tfb = tfp.bijectors

debug = False
debug_verbosity = -1
debug_gradients = False
check_numerics = False

debug_gradients &= debug
if check_numerics:
    tf.debugging.enable_check_numerics()

epsilon = 1e-25


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
                 encoder_temperature: float = 2. / 3,
                 prior_temperature: float = 1. / 2,
                 encoder_temperature_decay_rate: float = 0.,
                 prior_temperature_decay_rate: float = 0.,
                 regularizer_scale_factor: float = 0.,
                 regularizer_decay_rate: float = 0.,
                 kl_scale_factor: float = 1.,
                 kl_annealing_growth_rate: float = 0.,
                 mixture_components: int = 3,
                 multivariate_normal_raw_scale_diag_activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softplus,
                 multivariate_normal_full_covariance: bool = False,
                 action_pre_processing_network: Model = None,
                 state_pre_processing_network: Model = None,
                 state_post_processing_network: Model = None,
                 pre_loaded_model: bool = False):

        super(VariationalMarkovDecisionProcess, self).__init__()

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.reward_shape = reward_shape
        self.latent_state_size = latent_state_size
        self.label_shape = label_shape
        self.mixture_components = mixture_components
        self.full_covariance = multivariate_normal_full_covariance

        self.encoder_temperature = encoder_temperature
        self.prior_temperature = prior_temperature
        self.regularizer_scale_factor = regularizer_scale_factor
        self.kl_scale_factor = kl_scale_factor
        self.encoder_temperature_decay_rate = encoder_temperature_decay_rate
        self.prior_temperature_decay_rate = prior_temperature_decay_rate
        self.regularizer_decay_rate = regularizer_decay_rate
        self.kl_growth_rate = kl_annealing_growth_rate

        self.scale_activation = multivariate_normal_raw_scale_diag_activation

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

        state, next_state = input_state, input_next_state
        if state_pre_processing_network is not None:
            # the concatenate axis is 0 to stack the inputs afterwards
            concatenate_inputs = Concatenate(axis=0)([input_state, input_next_state])
            stack_inputs = Reshape((2,) + state_shape)(concatenate_inputs)
            # Apply the same pre-processing to both state inputs
            apply_pre_processing = TimeDistributed(
                state_pre_processing_network, input_shape=(2,) + state_shape)(stack_inputs)
            state, next_state = apply_pre_processing[:, 0], apply_pre_processing[:, 1]

        if not pre_loaded_model:
            # Encoder network
            encoder_input = Concatenate(name="encoder_input")([state, action, reward, next_state])
            encoder = encoder_network(encoder_input)
            log_alpha_layer = Dense(
                units=latent_state_size - np.prod(label_shape), activation=None, name='log_alpha')(encoder)
            self.encoder_network = Model(
                inputs=[input_state, action, reward, input_next_state],
                outputs=log_alpha_layer,
                name='encoder')

            # Transition network
            # inputs are binary concrete random variables, outputs are locations of logistic distributions
            latent_state = Input(shape=(latent_state_size,), name="latent_state")
            if action_pre_processing_network:
                action_layer_1 = clone_model(action_pre_processing_network)(action)
            else:
                action_layer_1 = action

            transition_network_input = Concatenate(name="transition_network_input")([latent_state, action_layer_1])
            transition = transition_network(transition_network_input)
            transition_output_layer = Dense(
                units=latent_state_size, activation=None, name='transition_logistic_locations')(transition)
            self.transition_network = Model(
                inputs=[latent_state, action], outputs=transition_output_layer, name="transition_network")

            # Reward network
            next_latent_state = Input(shape=(latent_state_size,), name="next_latent_state")
            action_layer_2 = action if not action_pre_processing_network else action_pre_processing_network(action)
            reward_network_input = Concatenate(name="reward_network_input")(
                [latent_state, action_layer_2, next_latent_state])
            reward_1 = reward_network(reward_network_input)
            reward_mean = Dense(units=np.prod(reward_shape), activation=None, name='reward_mean_0')(reward_1)
            reward_mean = Reshape(reward_shape, name='reward_mean')(reward_mean)
            reward_raw_covar = Dense(
                units=np.prod(reward_shape),
                activation=None,
                name='reward_raw_diag_covariance_0'
            )(reward_1)
            reward_raw_covar = Reshape(reward_shape, name='reward_raw_diag_covariance')(reward_raw_covar)
            self.reward_network = Model(
                inputs=[latent_state, action, next_latent_state],
                outputs=[reward_mean, reward_raw_covar], name='reward_network')

            # Reconstruction network
            # inputs are latent binary states, outputs are given in parameter
            decoder = decoder_network(next_latent_state)
            if state_post_processing_network is not None:
                decoder = state_post_processing_network(decoder)
            # 1 mean per dimension, nb Normal Gaussian
            decoder_output_mean = Dense(
                units=mixture_components * np.prod(state_shape), activation=None, name='GMM_means_0')(decoder)
            decoder_output_mean = Reshape((mixture_components,) + state_shape, name="GMM_means")(decoder_output_mean)
            if self.full_covariance and len(state_shape) == 1:
                d = np.prod(state_shape) * (np.prod(state_shape) + 1) / 2
                decoder_raw_output = Dense(
                    units=mixture_components * d,
                    activation=None,
                    name='GMM_tril_params_0'
                )(decoder)
                decoder_raw_output = Reshape((mixture_components, d,), name='GMM_tril_params_1')(decoder_raw_output)
                decoder_raw_output = Lambda(lambda x: tfb.FillScaleTriL()(x), name='GMM_scale_tril')(decoder_raw_output)
            else:
                # n diagonal co-variance matrices
                decoder_raw_output = Dense(
                    units=mixture_components * np.prod(state_shape),
                    activation=None,
                    name='GMM_raw_diag_covariance_0'
                )(decoder)
                decoder_raw_output = Reshape(
                    (mixture_components,) + state_shape, name="GMM_raw_diag_covar")(decoder_raw_output)
            # number of Normal Gaussian forming the mixture model
            decoder_prior = Dense(units=mixture_components, activation='softmax', name="GMM_priors")(decoder)
            self.reconstruction_network = Model(
                inputs=next_latent_state,
                outputs=[decoder_output_mean, decoder_raw_output, decoder_prior],
                name='reconstruction_network')
        else:
            self.encoder_network = encoder_network
            self.transition_network = transition_network
            self.reward_network = reward_network
            self.reconstruction_network = decoder_network

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'state_mse': tf.keras.metrics.MeanSquaredError(name='state_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'cross_entropy_regularizer': tf.keras.metrics.Mean(name='cross_entropy_regularizer'),
        }

    def reset_metrics(self):
        for value in self.loss_metrics.values():
            value.reset_states()
        #  super().reset_metrics()

    def relaxed_encoding(
            self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, state_prime: tf.Tensor, label: tf.Tensor,
            temperature: float
    ) -> tfd.Distribution:
        """
        Encode the sample (s, a, r, l, s') into a into a Binary Concrete probability distribution over relaxed binary
        latent states.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(loc=alpha, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=log alpha / temperature, scale=1. / temperature))
        """
        log_alpha = self.encoder_network([state, action, reward, state_prime])
        # change label = 1 to 100 or label = 0 to -100 so that
        # sigmoid(logistic_z[-1]) ~= 1 if label = 1 and sigmoid(logistic_z[-1]) ~= 0 if label = 0
        logistic_label = (label * 2. - 1.) * 1e2
        log_alpha = tf.concat([log_alpha, logistic_label * temperature], axis=-1)
        return tfd.Logistic(loc=log_alpha / temperature, scale=1. / temperature)

    def binary_encode(
            self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, state_prime: tf.Tensor, label: tf.Tensor
    ) -> tfd.Distribution:
        """
        Encode the sample (s, a, r, l, s') into a Bernoulli probability distribution over binary latent states.
        """
        log_alpha = self.encoder_network([state, action, reward, state_prime])
        return tfd.Bernoulli(logits=tf.concat([log_alpha, (label * 2. - 1.) * 1e2], axis=-1))

    def decode(self, latent_state: tf.Tensor) -> tfd.Distribution:
        """
        Decode a binary latent state into a probability distribution over original states of the MDP, modeled by a GMM
        """
        [reconstruction_mean, reconstruction_raw_covariance, reconstruction_prior_components] = \
            self.reconstruction_network(latent_state)
        if self.mixture_components == 1:
            if self.full_covariance:
                return tfd.MultivariateNormalTriL(
                    loc=reconstruction_mean[0],
                    scale_tril=reconstruction_raw_covariance[0],
                )
            else:
                return tfd.MultivariateNormalDiag(
                    loc=reconstruction_mean[0],
                    scale_diag=self.scale_activation(reconstruction_raw_covariance[0]),
                )
        else:
            if self.full_covariance:
                return tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=reconstruction_prior_components),
                    components_distribution=tfd.MultivariateNormalTriL(
                        loc=reconstruction_mean,
                        scale_tril=reconstruction_raw_covariance,
                    ),
                )
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=reconstruction_prior_components),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=reconstruction_mean,
                        scale_diag=self.scale_activation(reconstruction_raw_covariance),
                    ),
                )

    def relaxed_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor, temperature: float
    ) -> tfd.Distribution:
        """
        Retrieves a Binary Concrete probability distribution P(z'|z, a) over successor latent states, given a relaxed
        binary latent state z and action a.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(loc=alpha, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=log alpha / temperature, scale=1. / temperature))
        """
        log_alpha = self.transition_network([latent_state, action])
        return tfd.Logistic(loc=log_alpha / temperature, scale=1. / temperature)

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor
    ) -> tfd.Distribution:
        """
        Retrieves a Bernoulli probability distribution P(z'|z, a) over successor latent states, given a binary latent
        state z and action a.
        """
        log_alpha = self.transition_network([latent_state, action])
        return tfd.Bernoulli(logits=log_alpha)

    def reward_probability_distribution(
            self, latent_state, action, next_latent_state
    ) -> tfd.Distribution:
        """
        Retrieves a probability distribution P(r|z, a, z') over rewards obtained when the latent transition z -> z' is
        encountered and action a is chosen. Note that rewards are scaled according to the reward scale factor.
        """
        [reward_mean, reward_raw_covariance] = self.reward_network([latent_state, action, next_latent_state])
        return tfd.MultivariateNormalDiag(
            loc=reward_mean,
            scale_diag=self.scale_activation(reward_raw_covariance),
        )

    def anneal(self):
        for var, decay_rate in [
            (self.encoder_temperature, self.encoder_temperature_decay_rate),
            (self.prior_temperature, self.prior_temperature_decay_rate),
            (self.regularizer_scale_factor, self.regularizer_decay_rate),
            (self._decay_kl_scale_factor, self.kl_growth_rate)
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

        if self.kl_growth_rate > 0:
            self.kl_scale_factor.assign(
                self._initial_kl_scale_factor + (1. - self._initial_kl_scale_factor) *
                (1. - self._decay_kl_scale_factor))

    def call(self, inputs, training=None, mask=None, metrics=True):
        # inputs are assumed to have shape
        # [(?, 2, state_shape), (?, 2, action_shape), (?, 2, reward_shape), (?, 2, state_shape), (?, 2, label_shape)]
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        latent_distribution = self.relaxed_encoding(s_0, a_0, r_0, s_1, l_1, self.encoder_temperature)
        latent_distribution_prime = self.relaxed_encoding(s_1, a_1, r_1, s_2, l_2, self.encoder_temperature)

        z = tf.sigmoid(latent_distribution.sample())
        logistic_z_prime = latent_distribution_prime.sample()

        log_q_z_prime = latent_distribution_prime.log_prob(logistic_z_prime)

        log_p_z_prime = self.relaxed_latent_transition_probability_distribution(
            z, a_1, self.prior_temperature
        ).log_prob(logistic_z_prime)

        z_prime = tf.sigmoid(logistic_z_prime)

        # Normal log-probability P(r_1 | z, a_1, z')
        reward_distribution = self.reward_probability_distribution(z, a_1, z_prime)
        log_p_rewards = reward_distribution.log_prob(r_1)

        # Reconstruction P(s_2 | z')
        state_distribution = self.decode(z_prime)
        log_p_state = state_distribution.log_prob(s_2)

        distortion = -1. * (log_p_state + log_p_rewards)
        rate = tf.reduce_sum(log_q_z_prime - log_p_z_prime, axis=1)

        def compute_cross_entropy_regularization():
            log_alpha = self.encoder_network([s_1, a_1, r_1, s_2])
            discrete_latent_distribution = tfd.Bernoulli(logits=log_alpha)
            uniform_distribution = tfd.Bernoulli(probs=0.5 * tf.ones(shape=tf.shape(log_alpha)))
            return tf.reduce_sum(uniform_distribution.kl_divergence(discrete_latent_distribution), axis=1)

        # cross-entropy regularization
        cross_entropy_regularizer = compute_cross_entropy_regularization()

        if metrics:
            self.loss_metrics['ELBO'](-1 * (distortion + rate))
            self.loss_metrics['state_mse'](s_2, state_distribution.sample())
            self.loss_metrics['reward_mse'](r_1, reward_distribution.sample())
            self.loss_metrics['distortion'](distortion)
            self.loss_metrics['rate'](rate)
            self.loss_metrics['annealed_rate'](self.kl_scale_factor * rate)
            self.loss_metrics['cross_entropy_regularizer'](cross_entropy_regularizer)

        if debug:
            tf.print(z, "sampled z")
            tf.print(logistic_z_prime, "sampled (logistic) z'")
            tf.print(self.encoder_network([s_1, a_1, r_1, s_2]), "log locations[:-1] of Q")
            tf.print(log_q_z_prime, "Log Q(logistic z'|s, a, r, s', l')")
            tf.print(self.transition_network([z, a_1]), "log-locations P_transition")
            tf.print(log_p_z_prime, "log P(logistic z'|z, a)")
            tf.print(z_prime, "sampled z'")
            tf.print(tf.exp(log_p_rewards), "P(r | z, a, z')")
            [reconstruction_mean, _, reconstruction_prior_components] = \
                self.reconstruction_network(z_prime)
            tf.print(reconstruction_mean, 'mean(s | z)')
            tf.print(reconstruction_prior_components, 'GMM: prior components')
            tf.print(log_p_state, "log P(s' | z')")
            tf.print(log_q_z_prime - log_p_z_prime, "log Q(z') - log P(z')")

        return [distortion, rate, cross_entropy_regularizer]

    def eval(self, inputs):
        """
        Use binary latent states instead of binary concrete continuous relaxation of the latent states to evaluate
        the VAE.
        """
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        latent_distribution = self.binary_encode(s_0, a_0, r_0, s_1, l_1)
        latent_distribution_prime = self.binary_encode(s_1, a_1, r_1, s_2, l_2)
        z = tf.cast(latent_distribution.sample(), tf.float32)
        z_prime = tf.cast(latent_distribution_prime.sample(), tf.float32)

        transition_distribution = self.discrete_latent_transition_probability_distribution(z_prime, a_1)
        rate = tf.reduce_sum(latent_distribution_prime.kl_divergence(transition_distribution), axis=1)

        reward_distribution = self.reward_probability_distribution(z, a_1, z_prime)
        log_p_rewards = reward_distribution.log_prob(r_1)

        state_distribution = self.decode(z_prime)
        log_p_reconstruction = state_distribution.log_prob(s_2)

        return tf.reduce_mean(log_p_reconstruction + log_p_rewards - rate)

    def mean_latent_bits_used(self, inputs, eps=1e-3):
        """
        Compute the mean number of bits used in the latent space of the vae_mdp for the given dataset batch.
        This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
        """
        mean_bits_used = 0
        for i in (0, 1):
            s, a, r, s_prime, l_prime = (x[:, i, :] for x in inputs)
            mean = tf.reduce_mean(self.binary_encode(s, a, r, s_prime, l_prime).mean(), axis=0)
            check = lambda x: 1 if 1 - eps > x > eps else 0
            mean_bits_used += tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()
        return {'mean_state_bits_used': mean_bits_used / 2}

    @property
    def encoder_temperature(self):
        return self._encoder_temperature

    @encoder_temperature.setter
    def encoder_temperature(self, value):
        self._encoder_temperature = tf.Variable(value, dtype=tf.float32, trainable=False)

    @property
    def encoder_temperature_decay_rate(self):
        return self._encoder_temperature_decay_rate

    @encoder_temperature_decay_rate.setter
    def encoder_temperature_decay_rate(self, value):
        self._encoder_temperature_decay_rate = tf.constant(value, dtype=tf.float32)

    @property
    def prior_temperature(self):
        return self._prior_temperature

    @prior_temperature.setter
    def prior_temperature(self, value):
        self._prior_temperature = tf.Variable(value, dtype=tf.float32, trainable=False)

    @property
    def prior_temperature_decay_rate(self):
        return self._prior_temperature_decay_rate

    @prior_temperature_decay_rate.setter
    def prior_temperature_decay_rate(self, value):
        self._prior_temperature_decay_rate = tf.constant(value, dtype=tf.float32)

    @property
    def regularizer_scale_factor(self):
        return self._regularizer_scale_factor

    @regularizer_scale_factor.setter
    def regularizer_scale_factor(self, value):
        self._regularizer_scale_factor = tf.Variable(value, dtype=tf.float32, trainable=False)

    @property
    def regularizer_decay_rate(self):
        return self._regularizer_decay_rate

    @regularizer_decay_rate.setter
    def regularizer_decay_rate(self, value):
        self._regularizer_decay_rate = tf.constant(value, dtype=tf.float32)

    @property
    def kl_scale_factor(self):
        return self._kl_scale_factor

    @kl_scale_factor.setter
    def kl_scale_factor(self, value):
        self._kl_scale_factor = tf.Variable(value, dtype=tf.float32, trainable=False)

    @property
    def kl_growth_rate(self):
        return self._kl_growth_rate

    @kl_growth_rate.setter
    def kl_growth_rate(self, value):
        self._initial_kl_scale_factor = tf.Variable(self.kl_scale_factor, dtype=tf.float32, trainable=False)
        self._decay_kl_scale_factor = tf.Variable(1., dtype=tf.float32, trainable=False)
        self._kl_growth_rate = tf.constant(value, dtype=tf.float32)


@tf.function
def compute_loss(vae_mdp: VariationalMarkovDecisionProcess, x):
    distortion, rate, cross_entropy_regularizer = vae_mdp(x)
    return tf.reduce_mean(
        distortion + vae_mdp.kl_scale_factor * rate +
        vae_mdp.regularizer_scale_factor * cross_entropy_regularizer
    )


@tf.function
def compute_apply_gradients(vae_mdp: VariationalMarkovDecisionProcess, x, optimizer, train_summary_writer=None):
    with tf.GradientTape() as tape:
        loss = compute_loss(vae_mdp, x)
    gradients = tape.gradient(loss, vae_mdp.trainable_variables)

    if debug_gradients:
        for gradient, variable in zip(gradients, vae_mdp.trainable_variables):
            tf.print(gradient, "Gradient for {}".format(variable.name))

    optimizer.apply_gradients(zip(gradients, vae_mdp.trainable_variables))
    return loss


def load(tf_model_path: str) -> VariationalMarkovDecisionProcess:
    if float(tf.__version__.split('.')[0] + '.' + "".join(tf.__version__.split('.')[1:])) >= 2.2:
        # only with TensorFlow>=2.2.0
        model = tf.keras.models.load_model(tf_model_path)
        label_shape = model.transition_network.output.get_shape()[1] - model.encoder_network.output.get_shape()[1]
        return VariationalMarkovDecisionProcess(
            tuple(model.encoder_network.input[0].shape.as_list()[1:]),
            tuple(model.encoder_network.input[1].shape.as_list()[1:]),
            tuple(model.encoder_network.input[2].shape.as_list()[1:]),
            (label_shape,),
            model.encoder_network,
            model.transition_network,
            model.reward_network,
            model.reconstruction_network,
            model.transition_network.inputs[0].shape[-1],
            encoder_temperature=model._encoder_temperature,
            prior_temperature=model._prior_temperature,
            regularizer_scale_factor=model._regularizer_scale_factor,
            kl_scale_factor=model._kl_scale_factor,
            pre_loaded_model=True)
    else:
        model = tf.saved_model.load(tf_model_path)
        assert model.transition_network.variables[-1].name == 'transition_logistic_locations/bias:0'
        return VariationalMarkovDecisionProcess(
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_1'].shape)[2:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_2'].shape)[2:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_3'].shape)[2:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_5'].shape)[2:],
            model.encoder_network,
            model.transition_network,
            model.reward_network,
            model.reconstruction_network,
            model.transition_network.variables[-1].shape[0],
            encoder_temperature=model._encoder_temperature,
            prior_temperature=model._prior_temperature,
            regularizer_scale_factor=model._regularizer_scale_factor,
            kl_scale_factor=model._kl_scale_factor,
            pre_loaded_model=True)


def train_from_policy(
        vae_mdp: VariationalMarkovDecisionProcess,
        policy: tf_agents.policies.tf_policy.Base,
        environment_suite,
        env_name: str,
        labeling_function: Callable,
        num_iterations: int = int(3e6),
        initial_collect_steps: int = int(1e4),
        collect_steps_per_iteration: Optional[int] = None,
        replay_buffer_capacity: int = int(1e6),
        parallelization: bool = True,
        num_parallel_call: int = 4,
        batch_size: int = 128,
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-4),
        checkpoint: Optional[tf.train.Checkpoint] = None,
        manager: Optional[tf.train.CheckpointManager] = None,
        log_interval: int = 80,
        eval_steps: int = int(1e3),
        save_model_interval: int = int(1e4),
        log_name: str = 'vae_training',
        annealing_period: int = 0,
        start_annealing_step: int = 0,
        logs: bool = True,
        display_progressbar: bool = True,
        save_directory='.'):

    if collect_steps_per_iteration is None:
        collect_steps_per_iteration = batch_size
    if parallelization:
        replay_buffer_capacity = replay_buffer_capacity // num_parallel_call
        collect_steps_per_iteration = collect_steps_per_iteration // num_parallel_call

    save_directory = os.path.join(save_directory, 'saves', env_name)

    # Load checkpoint
    if checkpoint is not None and manager is not None:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    # initialize logs
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('logs', 'gradient_tape', env_name, log_name, current_time)
    if not os.path.exists(train_log_dir) and logs:
        os.makedirs(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir) if logs else None

    # Load step
    global_step = checkpoint.save_counter if checkpoint else tf.Variable(0)
    start_step = global_step.numpy()
    print("Step: {}".format(global_step.numpy()))

    progressbar = Progbar(
        target=None,
        stateful_metrics=list(vae_mdp.loss_metrics.keys()) + [
            'loss', 't_1', 't_2', 'regularizer_scale_factor', 'step', "num_episodes", "env_steps",
            "replay_buffer_frames"],
        interval=0.1) if display_progressbar else None

    if parallelization:
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
            [lambda: environment_suite.load(env_name)] * num_parallel_call))
        tf_env.reset()
    else:
        py_env = environment_suite.load(env_name)
        py_env.reset()
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

    action_spec = tf_env.action_spec()

    # specs
    policy_step_spec = policy_step.PolicyStep(
        action=action_spec,
        state=(),
        info=())
    trajectory_spec = trajectory.from_transition(tf_env.time_step_spec(),
                                                 policy_step_spec,
                                                 tf_env.time_step_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    dataset_generator = lambda: replay_buffer.as_dataset(
        num_parallel_calls=num_parallel_call,
        num_steps=3
    ).map(
        map_func=lambda trajectory, _: map_rl_trajectory_to_vae_input(trajectory, labeling_function),
        num_parallel_calls=num_parallel_call,
        #  deterministic=False  # TF version >= 2.2.0
    )
    dataset = dataset_generator().batch(batch_size=batch_size, drop_remainder=True)
    dataset_iterator = iter(dataset.prefetch(tf.data.experimental.AUTOTUNE))

    num_episodes = tf_metrics.NumberOfEpisodes()
    env_steps = tf_metrics.EnvironmentSteps()
    observers = [num_episodes, env_steps] if not parallelization else []
    observers += [replay_buffer.add_batch]

    driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, policy, observers=observers, num_steps=collect_steps_per_iteration)
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, policy, observers=observers, num_steps=initial_collect_steps)

    driver.run = common.function(driver.run)

    print("Initial collect steps...")
    initial_collect_driver.run()
    print("Start training.")

    for _ in range(global_step.numpy(), num_iterations):
        # Collect a few steps and save them to the replay buffer.
        driver.run()

        training_step(dataset_batch=next(dataset_iterator), batch_size=batch_size, vae_mdp=vae_mdp, optimizer=optimizer,
                      annealing_period=annealing_period, global_step=global_step,
                      dataset_size=replay_buffer.num_frames().numpy(), display_progressbar=display_progressbar,
                      start_step=start_step, epoch=0, progressbar=progressbar, dataset_generator=dataset_generator,
                      save_model_interval=save_model_interval,
                      eval_ratio=eval_steps * batch_size / replay_buffer.num_frames(),
                      save_directory=save_directory, log_name=log_name, train_summary_writer=train_summary_writer,
                      log_interval=log_interval, manager=manager, logs=logs, start_annealing_step=start_annealing_step,
                      additional_metrics={
                "num_episodes": num_episodes.result(),
                "env_steps": env_steps.result(),
                "replay_buffer_frames": replay_buffer.num_frames()} if not parallelization else {
                "replay_buffer_frames": replay_buffer.num_frames()},
                      train_log_dir=train_log_dir)


def train_from_dataset(
        vae_mdp: VariationalMarkovDecisionProcess,
        dataset_generator: Optional[Callable[[], tf.data.Dataset]] = None,
        epochs: int = 16,
        batch_size: int = 128,
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-4),
        checkpoint: Optional[tf.train.Checkpoint] = None,
        manager: Optional[tf.train.CheckpointManager] = None,
        log_interval: int = 80,
        save_model_interval: int = int(1e4),
        dataset_size: Optional[int] = None,
        log_name: str = 'vae',
        annealing_period: int = 0,
        start_annealing_step: int = 0,
        logs: bool = True,
        display_progressbar: bool = True,
        eval_ratio: float = 5e-3,
        max_steps: int = int(1e6),
        save_directory='.'):
    assert 0 < eval_ratio < 1

    # Load checkpoint
    if checkpoint is not None and manager is not None:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    # initialize logs
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('logs/gradient_tape', log_name, current_time)
    if not os.path.exists(train_log_dir) and logs:
        os.makedirs(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir) if logs else None

    # Load step
    global_step = checkpoint.save_counter if checkpoint else tf.Variable(0)
    start_step = global_step.numpy()
    print("Step: {}".format(global_step.numpy()))

    # start training
    for epoch in range(epochs):
        progressbar = Progbar(
            target=dataset_size,
            stateful_metrics=list(vae_mdp.loss_metrics.keys()) + ['loss', 't_1', 't_2', 'regularizer_scale_factor',
                                                                  'step'],
            interval=0.1) if display_progressbar else None
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        dataset = dataset_generator()

        for x in dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE):
            training_step(dataset_batch=x, batch_size=batch_size, vae_mdp=vae_mdp, optimizer=optimizer,
                          annealing_period=annealing_period, global_step=global_step, dataset_size=dataset_size,
                          display_progressbar=display_progressbar, start_step=start_step, epoch=epoch,
                          progressbar=progressbar, dataset_generator=dataset_generator,
                          save_model_interval=save_model_interval, eval_ratio=eval_ratio, save_directory=save_directory,
                          log_name=log_name, train_summary_writer=train_summary_writer, log_interval=log_interval,
                          manager=manager, logs=logs, start_annealing_step=start_annealing_step,
                          train_log_dir=train_log_dir)

        # reset metrics
        vae_mdp.reset_metrics()

        # retrieve the real dataset size
        if epoch == 0:
            dataset_size = (global_step.numpy() - start_step) * batch_size

        if global_step.numpy() > max_steps:
            return


def training_step(dataset_batch, batch_size, vae_mdp, optimizer, annealing_period, global_step, dataset_size,
                  display_progressbar, start_step, epoch, progressbar, dataset_generator, save_model_interval,
                  eval_ratio, save_directory, log_name, train_summary_writer, log_interval, manager, logs,
                  train_log_dir, start_annealing_step,
                  additional_metrics: Optional[Dict[str, tf.Tensor]] = None):
    if additional_metrics is None:
        additional_metrics = {}

    gradients = compute_apply_gradients(vae_mdp, dataset_batch, optimizer)
    loss = gradients

    if annealing_period > 0 and \
            global_step.numpy() % annealing_period == 0 and global_step.numpy() > start_annealing_step:
        vae_mdp.anneal()

    # update progressbar
    metrics_key_values = [('step', global_step.numpy()), ('loss', loss.numpy())] + \
                         [(key, value.result()) for key, value in vae_mdp.loss_metrics.items()] + \
                         [(key, value) for key, value in additional_metrics.items()] + \
                         [(key, value) for key, value in vae_mdp.mean_latent_bits_used(dataset_batch).items()]
    if annealing_period != 0:
        metrics_key_values.append(('t_1', vae_mdp.encoder_temperature.numpy()))
        metrics_key_values.append(('t_2', vae_mdp.prior_temperature.numpy()))
        metrics_key_values.append(('regularizer_scale_factor', vae_mdp.regularizer_scale_factor))
        metrics_key_values.append(('kl_annealing_scale_factor', vae_mdp.kl_scale_factor))
    if progressbar is not None:
        if dataset_size is not None and progressbar.target is not None and display_progressbar and \
                (global_step.numpy() - start_step) * batch_size < dataset_size * (epoch + 1):
            progressbar.add(batch_size, values=metrics_key_values)
        elif (dataset_size is None or progressbar.target is None) and display_progressbar:
            progressbar.add(batch_size, values=metrics_key_values)

    # update step
    global_step.assign_add(1)

    # eval, save and log
    eval_steps = int(1e3) if dataset_size is None else int(dataset_size * eval_ratio) // batch_size
    if global_step.numpy() % save_model_interval == 0:
        eval_and_save(vae_mdp=vae_mdp, dataset=dataset_generator(),
                      batch_size=batch_size, eval_steps=eval_steps,
                      global_step=int(global_step.numpy()), save_directory=save_directory, log_name=log_name,
                      train_summary_writer=train_summary_writer)
    if global_step.numpy() % log_interval == 0:
        if manager is not None:
            manager.save()
        if logs:
            with train_summary_writer.as_default():
                for key, value in metrics_key_values:
                    tf.summary.scalar(key, value, step=global_step.numpy())
        # reset metrics
        vae_mdp.reset_metrics()


def eval_and_save(vae_mdp: VariationalMarkovDecisionProcess,
                  dataset: tf.data.Dataset,
                  batch_size: int,
                  eval_steps: int,
                  global_step: int,
                  save_directory: str,
                  log_name: str,
                  train_summary_writer: Optional[tf.summary.SummaryWriter] = None):
    eval_elbo = tf.metrics.Mean()
    if eval_steps > 0:
        eval_set = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        eval_progressbar = Progbar(target=eval_steps * batch_size, interval=0.1, stateful_metrics=['eval_ELBO'])

        tf.print("\nEvalutation over {} steps".format(eval_steps))

        for step, x in enumerate(eval_set):
            eval_elbo(vae_mdp.eval(x))
            eval_progressbar.add(batch_size, values=[('eval_ELBO', eval_elbo.result())])
            if step > eval_steps:
                break

        del eval_set
        del dataset

        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar('eval_elbo', eval_elbo.result(), step=global_step)
        print('eval ELBO: ', eval_elbo.result().numpy())
        model_name = '{}_step{}_eval_elbo{:.3f}'.format(log_name, global_step, eval_elbo.result())
    else:
        model_name = '{}_step{}'.format(log_name, global_step)
    if check_numerics:
        tf.debugging.disable_check_numerics()
    tf.saved_model.save(vae_mdp, os.path.join(save_directory, 'models', model_name))
    if check_numerics:
        tf.debugging.enable_check_numerics()

    return eval_elbo


def evaluate_encoder_distribution(vae_mdp: VariationalMarkovDecisionProcess,
                                  dataset: tf.data.Dataset,
                                  dataset_size: int,
                                  display_progressbar: bool = True,
                                  batch_size: int = 128):
    from IPython.display import clear_output
    import time

    mean = None

    step = tf.Variable(0., dtype=tf.float32)
    wait_time = 1.
    progressbar = Progbar(target=dataset_size, interval=0.1) if display_progressbar else None
    t = time.time()

    for x in dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE):
        for i in (0, 1):
            s, a, r, s_prime, l_prime = (transition[:, i, :] for transition in x)
            probs = tf.reduce_mean(vae_mdp.binary_encode(s, a, r, s_prime, l_prime).probs_parameter(), axis=0)
            if mean is None:
                mean = probs
            else:
                mean += probs
        step.assign_add(1)
        if step < dataset_size and display_progressbar:
            progressbar.add(batch_size)
        if not display_progressbar and time.time() - t > wait_time:
            b = 'mean=' + str(mean / (2 * (step + 1)))
            c = 'current=' + str(probs / 1.)
            print('\n', b, '\n', c)
            clear_output(wait=True)
            t = time.time()
    return mean / (2 * (step + 1))
