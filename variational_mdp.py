import os
import threading
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
from tf_agents.environments import tf_py_environment, parallel_py_environment, tf_environment
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.utils import common

from util.io.dataset_generator import map_rl_trajectory_to_vae_input, ErgodicMDPTransitionGenerator

tfd = tfp.distributions
tfb = tfp.bijectors

debug = False
debug_verbosity = -1
debug_gradients = False
check_numerics = False

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
                 entropy_regularizer_scale_factor: float = 0.,
                 entropy_regularizer_decay_rate: float = 0.,
                 entropy_regularizer_scale_factor_min_value: float = 0.,
                 marginal_entropy_regularizer_ratio: float = 0.,
                 kl_scale_factor: float = 1.,
                 kl_annealing_growth_rate: float = 0.,
                 mixture_components: int = 3,
                 max_decoder_variance: Optional[float] = None,
                 multivariate_normal_raw_scale_diag_activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.softplus,
                 multivariate_normal_full_covariance: bool = False,
                 action_pre_processing_network: Model = None,
                 state_pre_processing_network: Model = None,
                 state_post_processing_network: Model = None,
                 state_scaler: Callable[[tf.Tensor], tf.Tensor] = lambda x: x,
                 pre_loaded_model: bool = False,
                 reset_state_label: bool = True):

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
        self.entropy_regularizer_scale_factor = (
                entropy_regularizer_scale_factor - entropy_regularizer_scale_factor_min_value)
        self.kl_scale_factor = kl_scale_factor
        self.encoder_temperature_decay_rate = encoder_temperature_decay_rate
        self.prior_temperature_decay_rate = prior_temperature_decay_rate
        self.entropy_regularizer_decay_rate = entropy_regularizer_decay_rate
        self.kl_growth_rate = kl_annealing_growth_rate
        self.max_decoder_variance = max_decoder_variance

        self.scale_activation = multivariate_normal_raw_scale_diag_activation
        self.state_scaler = state_scaler
        self.entropy_regularizer_scale_factor_min_value = tf.constant(entropy_regularizer_scale_factor_min_value)
        self.marginal_entropy_regularizer_ratio = marginal_entropy_regularizer_ratio

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

        scaler = Lambda(self.state_scaler, name='state_scaler')
        state = scaler(state)

        if not pre_loaded_model:
            # Encoder network
            encoder = encoder_network(state)
            logits_layer = Dense(
                units=latent_state_size - np.prod(label_shape) - int(reset_state_label),
                activation=None,
                name='encoder_latent_distribution_logits'
            )(encoder)
            self.encoder_network = Model(
                inputs=input_state,
                outputs=logits_layer,
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
                units=latent_state_size, activation=None, name='latent_transition_distribution_logits')(transition)
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
                outputs=[reward_mean, reward_raw_covar],
                name='reward_network')

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
                    name='GMM_raw_diag_covariance_0',
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
            'entropy_regularizer': tf.keras.metrics.Mean(name='entropy_regularizer'),
            'encoder_entropy': tf.keras.metrics.Mean(name='encoder_entropy'),
            #  'decoder_variance': tf.keras.metrics.Mean(name='decoder_variance')
        }

    def reset_metrics(self):
        for value in self.loss_metrics.values():
            value.reset_states()
        #  super().reset_metrics()

    def relaxed_encoding(self, state: tf.Tensor, label: tf.Tensor, temperature: float) -> tfd.Distribution:
        """
        Embed the input state along with its label into a Binary Concrete probability distribution over
        a relaxed binary latent representation of the latent state space.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(loc=alpha, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=logits/temperature, scale=1./temperature))
        """
        logits = self.encoder_network(state)
        # change label = 1 to 100 and label = 0 to -100 so that
        # sigmoid(logistic_z[-1]) ~= 1 if label = 1 and sigmoid(logistic_z[-1]) ~= 0 if label = 0
        logistic_label = (label * 2. - 1.) * 1e2
        logits = tf.concat([logits, logistic_label], axis=-1)
        return tfd.Logistic(loc=logits / temperature, scale=1. / temperature)

    def binary_encode(self, state: tf.Tensor, label: tf.Tensor) -> tfd.Distribution:
        """
        Embed the input state along with its label into a Bernoulli probability distribution over the binary
        representation of the latent state space.
        """
        logits = self.encoder_network(state)
        return tfd.Bernoulli(
            logits=tf.concat([logits, (label * 2. - 1.) * 1e2], axis=-1),
            name='discrete_state_encoder_distribution'
        )

    def decode(self, latent_state: tf.Tensor) -> tfd.Distribution:
        """
        Decode a binary latent state to a probability distribution over states of the original MDP.
        """
        [
            reconstruction_mean, reconstruction_raw_covariance, reconstruction_prior_components
        ] = self.reconstruction_network(latent_state)
        if self.max_decoder_variance is None:
            reconstruction_raw_covariance = self.scale_activation(reconstruction_raw_covariance)
        else:
            reconstruction_raw_covariance = tfp.bijectors.SoftClip(
                low=epsilon, high=self.max_decoder_variance ** 0.5).forward(reconstruction_raw_covariance)

        if self.mixture_components == 1:
            if self.full_covariance:
                return tfd.MultivariateNormalTriL(
                    loc=reconstruction_mean[0],
                    scale_tril=reconstruction_raw_covariance[0],
                )
            else:
                return tfd.MultivariateNormalDiag(
                    loc=reconstruction_mean[0],
                    scale_diag=reconstruction_raw_covariance[0]
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
                        scale_diag=reconstruction_raw_covariance
                    ),
                )

    def relaxed_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor, temperature: float
    ) -> tfd.Distribution:
        """
        Retrieves a Binary Concrete probability distribution P(z'|z, a) over successor latent states, given a latent
        state z given in relaxed binary representation and an action a.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(loc=alpha, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=log alpha / temperature, scale=1. / temperature))
        """
        logits = self.transition_network([latent_state, action])
        return tfd.Logistic(loc=logits / temperature, scale=1. / temperature)

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor
    ) -> tfd.Distribution:
        """
        Retrieves a Bernoulli probability distribution P(z'|z, a) over successor latent states, given a binary latent
        state z and an action a.
        """
        logits = self.transition_network([latent_state, action])
        return tfd.Bernoulli(logits=logits, name='discrete_state_transition_distribution')

    def reward_probability_distribution(
            self, latent_state, action, next_latent_state
    ) -> tfd.Distribution:
        """
        Retrieves a probability distribution P(r|z, a, z') over rewards obtained when action a is chosen and the latent
        transition z -> z' occurs. Note that rewards are scaled according to the reward scale factor.
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
            (self._entropy_regularizer_scale_factor, self.entropy_regularizer_decay_rate),
            (self._decay_kl_scale_factor, self.kl_growth_rate)
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

        if self.kl_growth_rate > 0:
            self.kl_scale_factor.assign(
                self._initial_kl_scale_factor + (1. - self._initial_kl_scale_factor) *
                (1. - self._decay_kl_scale_factor))

    def call(self, inputs, training=None, mask=None, metrics=True):
        state, label, action, reward, next_state, next_label = inputs

        latent_distribution = self.relaxed_encoding(state, label, self.encoder_temperature)
        next_latent_distribution = self.relaxed_encoding(next_state, next_label, self.encoder_temperature)

        latent_state = tf.sigmoid(latent_distribution.sample())
        next_logistic_latent_state = next_latent_distribution.sample()

        log_q = next_latent_distribution.log_prob(next_logistic_latent_state)

        log_p_prior = self.relaxed_latent_transition_probability_distribution(
            latent_state, action, self.prior_temperature
        ).log_prob(next_logistic_latent_state)

        # retrieve binary concrete sample
        next_latent_state = tf.sigmoid(next_logistic_latent_state)

        # log P(r | z, a_1, z')
        reward_distribution = self.reward_probability_distribution(latent_state, action, next_latent_state)
        log_p_rewards = reward_distribution.log_prob(reward)

        # Reconstruction log P(s' | z')
        state_distribution = self.decode(next_latent_state)
        log_p_state = state_distribution.log_prob(self.state_scaler(next_state))

        distortion = -1. * (log_p_state + log_p_rewards)
        rate = tf.reduce_sum(log_q - log_p_prior, axis=1)
        entropy_regularizer = self.entropy_regularizer(
            next_state,
            enforce_latent_space_spreading=(self.marginal_entropy_regularizer_ratio > 0.),
            latent_states=next_latent_state)

        if metrics:
            self.loss_metrics['ELBO'](-1 * (distortion + rate))
            self.loss_metrics['state_mse'](self.state_scaler(next_state), state_distribution.sample())
            self.loss_metrics['reward_mse'](reward, reward_distribution.sample())
            self.loss_metrics['encoder_entropy'](self.binary_encode(next_state, next_label).entropy())
            #  self.loss_metrics['decoder_variance'](state_distribution.variance())
            self.loss_metrics['distortion'](distortion)
            self.loss_metrics['rate'](rate)
            self.loss_metrics['annealed_rate'](self.kl_scale_factor * rate)
            self.loss_metrics['entropy_regularizer'](self.entropy_regularizer_scale_factor * entropy_regularizer)

        if debug:
            tf.print(latent_state, "sampled z")
            tf.print(next_latent_state, "sampled (logistic) z'")
            tf.print(self.encoder_network([state, label]), "log locations[:-1] of Q")
            tf.print(log_q, "Log Q(logistic z'|s', l')")
            tf.print(self.transition_network([latent_state, action]), "log-locations P_transition")
            tf.print(log_p_prior, "log P(logistic z'|z, a)")
            tf.print(next_latent_state, "sampled z'")
            tf.print(tf.exp(log_p_rewards), "P(r | z, a, z')")
            [reconstruction_mean, _, reconstruction_prior_components] = \
                self.reconstruction_network(next_latent_state)
            tf.print(reconstruction_mean, 'mean(s | z)')
            tf.print(reconstruction_prior_components, 'GMM: prior components')
            tf.print(log_p_state, "log P(s' | z')")
            tf.print(log_q - log_p_prior, "log Q(z') - log P(z')")

        return [distortion, rate, entropy_regularizer]

    @tf.function
    def entropy_regularizer(
            self, state: tf.Tensor,
            enforce_latent_space_spreading: bool = False,
            latent_states: Optional[tf.Tensor] = None
    ):
        logits = self.encoder_network(state)
        regularizer = -1. * tf.reduce_mean(tfd.Bernoulli(logits=logits).entropy(), axis=0)

        if enforce_latent_space_spreading:
            batch_size = tf.shape(logits)[0]
            marginal_encoder = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=tf.ones(shape=batch_size)),
                components_distribution=tfd.RelaxedBernoulli(
                    logits=tf.transpose(logits), temperature=self.encoder_temperature),
                reparameterize=(latent_states is None)
            )
            if latent_states is None:
                latent_states = marginal_encoder.sample(batch_size)
            latent_states = latent_states[..., :tf.shape(logits)[1]]
            latent_states = tf.clip_by_value(latent_states, clip_value_min=1e-7, clip_value_max=1.-1e-7)
            # marginal_entropy_regularizer = -1. * tf.reduce_mean(
            #     -1. * tf.reduce_sum(marginal_encoder.log_prob(latent_states), axis=1))
            marginal_entropy_regularizer = tf.reduce_mean(marginal_encoder.log_prob(latent_states), axis=0)
            regularizer = ((1. - self.marginal_entropy_regularizer_ratio) * regularizer +
                           self.marginal_entropy_regularizer_ratio *
                           (tf.abs(self.entropy_regularizer_scale_factor) / self.entropy_regularizer_scale_factor)
                           * marginal_entropy_regularizer)
        return tf.reduce_mean(regularizer)

    def eval(self, inputs):
        """
        Evaluate the ELBO by making use of a discrete latent space.
        """
        state, label, action, reward, next_state, next_label = inputs

        latent_distribution = self.binary_encode(state, label)
        next_latent_distribution = self.binary_encode(state, next_label)
        latent_state = tf.cast(latent_distribution.sample(), tf.float32)
        next_latent_state = tf.cast(next_latent_distribution.sample(), tf.float32)

        transition_distribution = self.discrete_latent_transition_probability_distribution(next_latent_state, action)
        rate = tf.reduce_sum(next_latent_distribution.kl_divergence(transition_distribution), axis=1)

        reward_distribution = self.reward_probability_distribution(latent_state, action, next_latent_state)
        log_p_rewards = reward_distribution.log_prob(reward)

        state_distribution = self.decode(next_latent_state)
        log_p_reconstruction = state_distribution.log_prob(self.state_scaler(next_state))

        return (
            tf.reduce_mean(log_p_reconstruction + log_p_rewards - rate),
            tf.concat([tf.cast(latent_state, tf.int32), tf.cast(next_latent_state, tf.int32)], axis=0),
            None
        )

    def mean_latent_bits_used(self, inputs, eps=1e-3):
        """
        Compute the mean number of bits used to represent the latent space of the vae_mdp for the given dataset batch.
        This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
        """
        mean_bits_used = 0
        s, l, a, r, s_prime, l_prime = inputs
        mean = tf.reduce_mean(self.binary_encode(s, l).mean(), axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used += tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()
        return {'mean_state_bits_used': mean_bits_used}

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
    def entropy_regularizer_scale_factor(self):
        return self._entropy_regularizer_scale_factor + self.entropy_regularizer_scale_factor_min_value

    @entropy_regularizer_scale_factor.setter
    def entropy_regularizer_scale_factor(self, value):
        self._entropy_regularizer_scale_factor = tf.Variable(value, dtype=tf.float32, trainable=False)

    @property
    def entropy_regularizer_decay_rate(self):
        return self._regularizer_decay_rate

    @entropy_regularizer_decay_rate.setter
    def entropy_regularizer_decay_rate(self, value):
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

    @property
    def inference_variables(self):
        return self.encoder_network.trainable_variables

    @property
    def generator_variables(self):
        variables = []
        for network in [self.reconstruction_network, self.reward_network, self.transition_network]:
            variables += network.trainable_variables
        return variables

    @tf.function
    def compute_loss(self, x):
        distortion, rate, entropy_regularizer = self(x)
        alpha = self.entropy_regularizer_scale_factor
        beta = self.kl_scale_factor
        return tf.reduce_mean(
            distortion + beta * rate + alpha * entropy_regularizer
        )

    def _compute_apply_gradients(self, x, optimizer, trainable_variables):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)

        gradients = tape.gradient(loss, trainable_variables)

        if debug_gradients:
            for gradient, variable in zip(gradients, trainable_variables):
                tf.print(gradient, "Gradient for {}".format(variable.name))

        optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    @tf.function
    def compute_apply_gradients(self, x, optimizer):
        return self._compute_apply_gradients(x, optimizer, self.trainable_variables)

    @tf.function
    def inference_update(self, x, optimizer):
        return self._compute_apply_gradients(x, optimizer, self.inference_variables)

    @tf.function
    def generator_update(self, x, optimizer):
        return self._compute_apply_gradients(x, optimizer, self.generator_variables)

    def train_from_policy(
            self,
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
            save_directory='.',
            get_policy_evaluation: Optional[Callable[[], tf_agents.policies.tf_policy.Base]] = None,
            policy_evaluation_num_episodes: int = 30,
            wrap_eval_tf_env: Optional[Callable[[tf_environment.TFEnvironment], tf_environment.TFEnvironment]] = None,
            aggressive_training: bool = False,
            approximate_convergence_error: float = 5e-1,
            approximate_convergence_steps: int = 10,
            aggressive_training_steps: int = int(2e6)
    ):
        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = batch_size
        if parallelization:
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_call
            collect_steps_per_iteration = max(1, collect_steps_per_iteration // num_parallel_call)

        save_directory = os.path.join(save_directory, 'saves', env_name)

        # Load checkpoint
        if checkpoint is not None and manager is not None:
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

        # initialize logs
        if logs:
            train_log_dir = os.path.join('logs', 'gradient_tape', env_name, log_name)  # , current_time)
            print('logs path:', train_log_dir)
            if not os.path.exists(train_log_dir) and logs:
                os.makedirs(train_log_dir)
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        else:
            train_summary_writer = None

        # Load step
        global_step = checkpoint.save_counter if checkpoint else tf.Variable(0)
        start_step = global_step.numpy()
        print("Step: {}".format(global_step.numpy()))

        progressbar = Progbar(
            target=None,
            stateful_metrics=list(self.loss_metrics.keys()) + [
                'loss', 't_1', 't_2', 'entropy_regularizer_scale_factor', 'step', "num_episodes", "env_steps",
                "replay_buffer_frames", 'kl_annealing_scale_factor', "decoder_jsdiv", 'state_rate',
                "state_distortion", 'action_rate', 'action_distortion', 'mean_state_bits_used'],
            interval=0.1) if display_progressbar else None

        if parallelization:
            tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: environment_suite.load(env_name)] * num_parallel_call))
            tf_env.reset()
        else:
            py_env = environment_suite.load(env_name)
            py_env.reset()
            tf_env = tf_py_environment.TFPyEnvironment(py_env)

        # specs
        action_spec = tf_env.action_spec()
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

        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        observers = [num_episodes, env_steps] if not parallelization else []
        observers += [replay_buffer.add_batch]

        driver = dynamic_step_driver.DynamicStepDriver(
            tf_env, policy, observers=observers, num_steps=collect_steps_per_iteration)
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env, policy, observers=observers, num_steps=initial_collect_steps)

        policy_evaluation_driver = None
        if get_policy_evaluation is not None and wrap_eval_tf_env is not None:
            py_eval_env = environment_suite.load(env_name)
            eval_env = tf_py_environment.TFPyEnvironment(py_eval_env)
            eval_env = wrap_eval_tf_env(eval_env)
            eval_env.reset()
            policy_evaluation_driver = tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
                eval_env, get_policy_evaluation(), num_episodes=policy_evaluation_num_episodes
            )

        driver.run = common.function(driver.run)

        print("Initial collect steps...")
        initial_collect_driver.run()
        print("Start training.")

        def dataset_generator():
            generator = ErgodicMDPTransitionGenerator(labeling_function, replay_buffer)
            return replay_buffer.as_dataset(
                num_parallel_calls=num_parallel_call,
                num_steps=2
            ).map(
                map_func=generator,
                num_parallel_calls=num_parallel_call,
                #  deterministic=False  # TF version >= 2.2.0
            )

        dataset = dataset_generator().batch(batch_size=batch_size, drop_remainder=True)
        dataset_iterator = iter(dataset.prefetch(tf.data.experimental.AUTOTUNE))

        # aggressive training metrics
        best_loss = None
        prev_loss = None
        convergence_error = approximate_convergence_error
        convergence_steps = approximate_convergence_steps
        aggressive_inference_optimization = True
        max_inference_update_steps = int(1e2)
        inference_update_steps = 0

        for _ in range(global_step.numpy(), num_iterations):
            # Collect a few steps and save them to the replay buffer.
            driver.run()

            loss = self.training_step(
                dataset_batch=next(dataset_iterator), batch_size=batch_size, optimizer=optimizer,
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
                eval_policy_driver=policy_evaluation_driver,
                aggressive_training=aggressive_training and global_step.numpy() < aggressive_training_steps,
                aggressive_update=aggressive_inference_optimization
            )

            if aggressive_training and global_step.numpy() < aggressive_training_steps:
                if display_progressbar:
                    progressbar.add(0, [('aggressive_updates_ratio', aggressive_inference_optimization)])
                if best_loss is None:
                    best_loss = loss
                if prev_loss is not None:
                    if tf.abs(loss - prev_loss) > convergence_error:  # and loss < best_loss:
                        convergence_steps = approximate_convergence_steps
                    else:
                        convergence_steps -= 1
                    best_loss = min(loss, best_loss)
                prev_loss = loss
                inference_update_steps += 1
                if convergence_steps == 0 or inference_update_steps == max_inference_update_steps:
                    aggressive_inference_optimization = not aggressive_inference_optimization
                    convergence_steps = approximate_convergence_steps if aggressive_inference_optimization else 0
                    best_loss = None
                    prev_loss = None
                    inference_update_steps = 0

    def train_from_dataset(
            self,
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
                stateful_metrics=list(self.loss_metrics.keys()) + [
                    'loss', 't_1', 't_2', 'entropy_regularizer_scale_factor', 'step'],
                interval=0.1) if display_progressbar else None
            print("Epoch: {}/{}".format(epoch + 1, epochs))

            dataset = dataset_generator()

            for x in dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE):
                self.training_step(dataset_batch=x, batch_size=batch_size, optimizer=optimizer,
                                   annealing_period=annealing_period, global_step=global_step,
                                   dataset_size=dataset_size,
                                   display_progressbar=display_progressbar, start_step=start_step, epoch=epoch,
                                   progressbar=progressbar, dataset_generator=dataset_generator,
                                   save_model_interval=save_model_interval, eval_ratio=eval_ratio,
                                   save_directory=save_directory,
                                   log_name=log_name, train_summary_writer=train_summary_writer,
                                   log_interval=log_interval,
                                   manager=manager, logs=logs, start_annealing_step=start_annealing_step, )

            # reset metrics
            self.reset_metrics()

            # retrieve the real dataset size
            if epoch == 0:
                dataset_size = (global_step.numpy() - start_step) * batch_size

            if global_step.numpy() > max_steps:
                return

    def training_step(
            self, dataset_batch, batch_size, optimizer, annealing_period, global_step, dataset_size,
            display_progressbar, start_step, epoch, progressbar, dataset_generator, save_model_interval,
            eval_ratio, save_directory, log_name, train_summary_writer, log_interval, manager, logs,
            start_annealing_step, additional_metrics: Optional[Dict[str, tf.Tensor]] = None,
            eval_policy_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None,
            aggressive_training=False, aggressive_update=True
    ):
        if additional_metrics is None:
            additional_metrics = {}
        if not aggressive_training:
            gradients = self.compute_apply_gradients(dataset_batch, optimizer)
        elif aggressive_update:
            gradients = self.inference_update(dataset_batch, optimizer)
        else:
            gradients = self.generator_update(dataset_batch, optimizer)
        loss = gradients

        if annealing_period > 0 and \
                global_step.numpy() % annealing_period == 0 and global_step.numpy() > start_annealing_step:
            self.anneal()

        # update progressbar
        metrics_key_values = [('step', global_step.numpy()), ('loss', loss.numpy())] + \
                             [(key, value.result()) for key, value in self.loss_metrics.items()] + \
                             [(key, value) for key, value in additional_metrics.items()] + \
                             [(key, value) for key, value in self.mean_latent_bits_used(dataset_batch).items()]
        if annealing_period != 0:
            metrics_key_values.append(('t_1', self.encoder_temperature))
            metrics_key_values.append(('t_2', self.prior_temperature))
            metrics_key_values.append(('entropy_regularizer_scale_factor', self.entropy_regularizer_scale_factor))
            metrics_key_values.append(('kl_annealing_scale_factor', self.kl_scale_factor))
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
            self.eval_and_save(dataset=dataset_generator(),
                               batch_size=batch_size, eval_steps=eval_steps,
                               global_step=int(global_step.numpy()), save_directory=save_directory, log_name=log_name,
                               train_summary_writer=train_summary_writer,
                               eval_policy_driver=eval_policy_driver)
        if global_step.numpy() % log_interval == 0:
            if manager is not None:
                manager.save()
            if logs:
                with train_summary_writer.as_default():
                    for key, value in metrics_key_values:
                        tf.summary.scalar(key, value, step=global_step.numpy())
            # reset metrics
            self.reset_metrics()

        return loss

    def eval_and_save(
            self,
            dataset: tf.data.Dataset,
            batch_size: int,
            eval_steps: int,
            global_step: int,
            save_directory: str,
            log_name: str,
            train_summary_writer: Optional[tf.summary.SummaryWriter] = None,
            eval_policy_driver: Optional[tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver] = None
    ):
        eval_elbo = tf.metrics.Mean()
        if eval_steps > 0:
            eval_set = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
            eval_progressbar = Progbar(target=eval_steps * batch_size, interval=0.1, stateful_metrics=['eval_ELBO'])

            tf.print("\nEvalutation over {} steps".format(eval_steps))
            data = {'state': None, 'action': None}
            for step, x in enumerate(eval_set):
                elbo, latent_states, latent_actions = self.eval(x)
                for value in ('state', 'action'):
                    latent = latent_states if value == 'state' else latent_actions
                    data[value] = latent if data[value] is None else tf.concat([data[value], latent], axis=0)
                eval_elbo(elbo)
                eval_progressbar.add(batch_size, values=[('eval_ELBO', eval_elbo.result())])
                if step > eval_steps:
                    break

            del eval_set

            if eval_policy_driver is not None:
                eval_policy_thread = threading.Thread(
                    target=eval_policy,
                    args=(eval_policy_driver, train_summary_writer, global_step),
                    daemon=True,
                    name='eval')
                eval_policy_thread.start()

            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar('eval_elbo', eval_elbo.result(), step=global_step)
                    for value in ('state', 'action'):
                        if data[value] is not None:
                            if value == 'state':
                                data[value] = tf.cast(
                                    tf.reduce_sum(data[value] * 2 ** tf.range(self.latent_state_size), axis=-1),
                                    dtype=tf.int64
                                )
                            tf.summary.histogram('{}_frequency'.format(value), data[value], step=global_step)
            print('eval ELBO: ', eval_elbo.result().numpy())
            model_name = os.path.join(log_name, 'step{}'.format(global_step),
                                      'eval_elbo{:.3f}'.format(eval_elbo.result()))
        else:
            model_name = os.path.join(log_name, 'step{}'.format(global_step),
                                      'eval_elbo{:.3f}'.format(eval_elbo.result()))
        if check_numerics:
            tf.debugging.disable_check_numerics()
        tf.saved_model.save(self, os.path.join(save_directory, 'models', model_name))
        if check_numerics:
            tf.debugging.enable_check_numerics()

        del dataset

        return eval_elbo


def load(tf_model_path: str) -> VariationalMarkovDecisionProcess:
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
        entropy_regularizer_scale_factor=model._entropy_regularizer_scale_factor,
        kl_scale_factor=model._kl_scale_factor,
        pre_loaded_model=True)


def eval_policy(eval_policy_driver, train_summary_writer, global_step):
    eval_avg_rewards = tf_agents.metrics.tf_metrics.AverageReturnMetric()
    eval_policy_driver.observers.append(eval_avg_rewards)
    eval_policy_driver.run()
    eval_policy_driver.observers.remove(eval_avg_rewards)
    if train_summary_writer is not None:
        with train_summary_writer.as_default():
            tf.summary.scalar('policy_evaluation_avg_rewards', eval_avg_rewards.result(), step=global_step)
    print('eval policy', eval_avg_rewards.result().numpy())
