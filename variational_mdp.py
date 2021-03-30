import os
import threading
from typing import Tuple, Optional, List, Callable, Dict
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Reshape, Dense, Lambda
from tensorflow.keras.utils import Progbar

import tf_agents.policies.tf_policy
import tf_agents.agents.tf_agent
from tensorflow.python.keras.models import Sequential
from tf_agents import specs, trajectories
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment, tf_environment
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.utils import common

from util.io.dataset_generator import ErgodicMDPTransitionGenerator

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
                 action_shape: Tuple[int, ...],
                 reward_shape: Tuple[int, ...],
                 label_shape: Tuple[int, ...],
                 encoder_network: Model,
                 transition_network: Model,
                 reward_network: Model,
                 decoder_network: Model,
                 label_transition_network: Model = Sequential(
                     [Dense(units=256, activation='relu'),
                      Dense(units=256, activation='relu')],
                     name='label_transition_network_body'),
                 latent_policy_network: Optional[Model] = None,
                 induced_markov_chain_transition_function: bool = False,  # experimental
                 latent_state_size: int = 12,
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
                 pre_loaded_model: bool = False,
                 reset_state_label: bool = True,
                 latent_policy_training_phase: bool = False,
                 full_optimization: bool = True):

        super(VariationalMarkovDecisionProcess, self).__init__()

        self.state_shape = tf.TensorShape(state_shape)
        self.action_shape = tf.TensorShape(action_shape)
        self.reward_shape = tf.TensorShape(reward_shape)
        self.latent_state_size = tf.Variable(
            latent_state_size, dtype=tf.int64, trainable=False, name='latent_state_size')
        self.label_shape = tf.TensorShape(label_shape)
        self.atomic_props_dims = tf.Variable(
            np.prod(label_shape) + int(reset_state_label), dtype=tf.int64, trainable=False, name='atomic_props_dims')
        self.mixture_components = tf.constant(mixture_components)
        self.full_covariance = multivariate_normal_full_covariance
        self.induced_markov_chain_transition_function = (
                induced_markov_chain_transition_function and latent_policy_network is not None)
        self.latent_policy_training_phase = latent_policy_training_phase
        self.full_optimization = full_optimization

        self._entropy_regularizer_scale_factor = None
        self._kl_scale_factor = None
        self._initial_kl_scale_factor = None
        self._decay_kl_scale_factor = None

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
        self.entropy_regularizer_scale_factor_min_value = tf.constant(entropy_regularizer_scale_factor_min_value)
        self.marginal_entropy_regularizer_ratio = marginal_entropy_regularizer_ratio

        self.number_of_discrete_actions = -1  # only used if a latent policy network is provided

        state = Input(shape=state_shape, name="state")
        action = Input(shape=action_shape, name="action")

        if not pre_loaded_model:
            # Encoder network
            encoder = encoder_network(state)
            logits_layer = Dense(
                units=latent_state_size - self.atomic_props_dims.numpy(),
                # allows avoiding exploding logits values and probability errors after applying a sigmoid
                activation=lambda x: tfb.SoftClip(high=10., low=-10.).forward(x),
                name='encoder_latent_distribution_logits'
            )(encoder)
            self.encoder_network = Model(
                inputs=state,
                outputs=logits_layer,
                name='encoder')

            # Latent policy network
            latent_state = Input(shape=(latent_state_size,), name="latent_state")
            if latent_policy_network is not None:
                self.latent_policy_network = latent_policy_network(latent_state)
                # we assume actions to be discrete and given in one hot when using a latent policy network
                assert len(self.action_shape) == 1
                self.number_of_discrete_actions = self.action_shape[0]
                self.latent_policy_network = Dense(
                    units=self.number_of_discrete_actions,
                    activation=None,
                    name='latent_policy_one_hot_logits'
                )(self.latent_policy_network)
                self.latent_policy_network = Model(
                    inputs=latent_state,
                    outputs=self.latent_policy_network,
                    name='latent_policy_network'
                )
            else:
                self.latent_policy_network = None
                self.number_of_discrete_actions = -1

            # Transition network
            # inputs are binary concrete random variables, outputs are locations of logistic distributions
            next_label = Input(shape=(self.atomic_props_dims.numpy(),), name='next_label')
            if self.number_of_discrete_actions != -1:
                transition_network_input = Concatenate(name='transition_network_input')([latent_state, next_label])
                transition = transition_network(transition_network_input)
                no_latent_state_logits = latent_state_size - self.atomic_props_dims.numpy()
                transition_output_layer = Dense(
                    units=no_latent_state_logits * self.number_of_discrete_actions,
                    activation=None,
                    name='transition_raw_output_layer'
                )(transition)
                transition_output_layer = Reshape(
                    target_shape=(no_latent_state_logits, self.number_of_discrete_actions),
                    name='transition_output_layer_reshape'
                )(transition_output_layer)
                _action = tf.keras.layers.RepeatVector(no_latent_state_logits, name='repeat_action')(action)
                transition_output_layer = tf.keras.layers.Multiply(name="multiply_action_transition")(
                    [_action, transition_output_layer])
                transition_output_layer = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1), name='transition_logits_reduce_sum_action_mask_layer'
                )(transition_output_layer)
            else:
                transition_network_input = Concatenate(
                    name="transition_network_input")([latent_state, action, next_label])
                _transition_network = transition_network(transition_network_input)
                transition_output_layer = Dense(
                    units=latent_state_size - self.atomic_props_dims.numpy(),
                    activation=None,
                    name='latent_transition_distribution_logits'
                )(_transition_network)
            self.transition_network = Model(
                inputs=[latent_state, action, next_label], outputs=transition_output_layer, name="transition_network")

            # Label transition network
            # Gives logits of a Bernoulli distribution giving the probability of the next label given the
            # current latent state and the action chosen
            if self.number_of_discrete_actions != -1:
                _label_transition_network = label_transition_network(latent_state)
                _label_transition_network = Dense(
                    units=self.atomic_props_dims.numpy() * self.number_of_discrete_actions,
                    activation=None,
                    name="label_transition_network_raw_output_layer"
                )(_label_transition_network)
                _label_transition_network = Reshape(
                    target_shape=(self.atomic_props_dims.numpy(), self.number_of_discrete_actions),
                    name='reshape_label_transition_output'
                )(_label_transition_network)
                _action = tf.keras.layers.RepeatVector(self.atomic_props_dims.numpy(), name='repeat_action')(action)
                _label_transition_network = tf.keras.layers.Multiply()([_action, _label_transition_network])
                _label_transition_network = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1),
                    name='label_transition_reduce_sum_action_mask_layer'
                )(_label_transition_network)
            else:
                label_transition_network_input = Concatenate(
                    name="label_transition_network_input")([latent_state, action])
                _label_transition_network = label_transition_network(label_transition_network_input)
                _label_transition_network = Dense(
                    units=self.atomic_props_dims.numpy(), activation=None, name='next_label_transition_logits'
                )(_label_transition_network)
            self.label_transition_network = Model(
                inputs=[latent_state, action], outputs=_label_transition_network, name='label_transition_network')

            # Reward network
            next_latent_state = Input(shape=(latent_state_size,), name="next_latent_state")
            if self.number_of_discrete_actions != -1:
                reward_network_input = Concatenate(name="reward_network_input")(
                    [latent_state, next_latent_state])
                _reward_network = reward_network(reward_network_input)
                reward_mean = Dense(
                    units=np.prod(reward_shape) * self.number_of_discrete_actions,
                    activation=None,
                    name='reward_mean_raw_output')(_reward_network)
                reward_mean = Reshape(target_shape=(reward_shape + (self.number_of_discrete_actions,)))(reward_mean)
                _action = tf.keras.layers.RepeatVector(np.prod(reward_shape))(action)
                _action = Reshape(target_shape=(reward_shape + (self.number_of_discrete_actions,)))(_action)
                reward_mean = tf.keras.layers.Multiply(name="multiply_action_reward_stack")([_action, reward_mean])
                reward_mean = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1), name='reward_mean_reduce_sum_action_mask_layer'
                )(reward_mean)
                reward_raw_covar = Dense(
                    units=np.prod(reward_shape) * self.number_of_discrete_actions,
                    activation=None,
                    name='reward_covar_raw_output')(_reward_network)
                reward_raw_covar = Reshape(
                    target_shape=reward_shape + (self.number_of_discrete_actions,))(reward_raw_covar)
                reward_raw_covar = tf.keras.layers.Multiply(
                    name='multiply_action_raw_covar_stack')([_action, reward_raw_covar])
                reward_raw_covar = Lambda(
                    lambda x: tf.reduce_sum(x, axis=-1),
                    name='reward_raw_covar_reduce_sum_action_mask_layer'
                )(reward_raw_covar)
            else:
                reward_network_input = Concatenate(name="reward_network_input")(
                    [latent_state, action, next_latent_state])
                _reward_network = reward_network(reward_network_input)
                reward_mean = Dense(units=np.prod(reward_shape), activation=None, name='reward_mean_0')(_reward_network)
                reward_raw_covar = Dense(
                    units=np.prod(reward_shape),
                    activation=None,
                    name='reward_raw_diag_covariance_0'
                )(_reward_network)
            reward_mean = Reshape(reward_shape, name='reward_mean')(reward_mean)
            reward_raw_covar = Reshape(reward_shape, name='reward_raw_diag_covariance')(reward_raw_covar)
            self.reward_network = Model(
                inputs=[latent_state, action, next_latent_state],
                outputs=[reward_mean, reward_raw_covar],
                name='reward_network')

            # Reconstruction network
            # inputs are latent binary states, outputs are given in parameter
            decoder = decoder_network(next_latent_state)
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
            self.latent_policy_network = latent_policy_network

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'state_mse': tf.keras.metrics.MeanSquaredError(name='state_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'entropy_regularizer': tf.keras.metrics.Mean(name='entropy_regularizer'),
            'encoder_entropy': tf.keras.metrics.Mean(name='encoder_entropy'),
            'marginal_encoder_entropy': tf.keras.metrics.Mean(name='marginal_encoder_entropy'),
            'transition_log_probs': tf.keras.metrics.Mean(name='transition_log_probs'),
            'predicted_next_state_mse': tf.keras.metrics.Mean(name='predicted_next_state_mse'),
            #  'decoder_variance': tf.keras.metrics.Mean(name='decoder_variance')
        }
        if self.latent_policy_network is not None:
            self.loss_metrics['action_mse'] = tf.keras.metrics.Mean(name='action_mse')

    def reset_metrics(self):
        for value in self.loss_metrics.values():
            value.reset_states()
        #  super().reset_metrics()

    def relaxed_encoding(
            self, state: tf.Tensor, temperature: float, label: Optional[tf.Tensor] = None
    ) -> tfd.Distribution:
        """
        Embed the input state along with its label into a Binary Concrete probability distribution over
        a relaxed binary latent representation of the latent state space.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(logits, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=logits/temperature, scale=1./temperature))
        """
        logits = self.encoder_network(state)
        if label is not None:
            # change label = 1 to 100 and label = 0 to -100 so that
            # sigmoid(logistic_z[-1]) ~= 1 if label = 1 and sigmoid(logistic_z[-1]) ~= 0 if label = 0
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(tfd.Logistic(loc=logits / temperature, scale=1. / temperature))

    def binary_encode(self, state: tf.Tensor, label: Optional[tf.Tensor] = None) -> tfd.Distribution:
        """
        Embed the input state along with its label into a Bernoulli probability distribution over the binary
        representation of the latent state space.
        """
        logits = self.encoder_network(state)
        if label is not None:
            logits = tf.concat([(label * 2. - 1.) * 1e2, logits], axis=-1)
        return tfd.Independent(tfd.Bernoulli(logits, name='discrete_state_encoder_distribution'))

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
                    loc=reconstruction_mean[:, 0, ...],
                    scale_tril=reconstruction_raw_covariance[:, 0, ...],
                )
            else:
                return tfd.MultivariateNormalDiag(
                    loc=reconstruction_mean[:, 0, ...],
                    scale_diag=reconstruction_raw_covariance[:, 0, ...]
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
            self, latent_state: tf.Tensor, action: tf.Tensor, next_label: Optional[tf.Tensor] = None,
            temperature: float = 1e-5
    ) -> tfd.Distribution:
        """
        Retrieves a Binary Concrete probability distribution P(z'|z, a) over successor latent states, given a latent
        state z given in relaxed binary representation and an action a.
        Note: the Binary Concrete distribution is replaced by a Logistic distribution to avoid underflow issues:
              z ~ BinaryConcrete(logits, temperature) = sigmoid(z_logistic)
              with z_logistic ~ Logistic(loc=logits / temperature, scale=1. / temperature))
        """
        if next_label is not None:
            next_latent_state_logits = self.transition_network([latent_state, action, next_label])
            return tfd.Independent(tfd.Logistic(loc=next_latent_state_logits / temperature, scale=1. / temperature))
        else:
            return tfd.JointDistributionSequential([
                tfd.Independent(tfd.Bernoulli(logits=self.label_transition_network([latent_state, action]))),
                lambda _next_label: tfd.Independent(tfd.Logistic(
                    loc=self.transition_network([latent_state, action, _next_label]) / temperature,
                    scale=1. / temperature))
            ])

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor, next_label: Optional[tf.Tensor] = None
    ) -> tfd.Distribution:
        """
        Retrieves a Bernoulli probability distribution P(z'|z, a) over successor latent states, given a binary latent
        state z and an action a.
        """
        if next_label is not None:
            next_latent_state_logits = self.transition_network([latent_state, action, next_label])
            return tfd.Independent(tfd.Bernoulli(logits=next_latent_state_logits))
        else:
            return tfd.JointDistributionSequential([
                tfd.Independent(tfd.Bernoulli(logits=self.label_transition_network([latent_state, action]))),
                lambda _next_label: tfd.Independent(
                    tfd.Bernoulli(logits=self.transition_network([latent_state, action, _next_label])))
            ])

    def reward_probability_distribution(
            self, latent_state: tf.Tensor, action: tf.Tensor, next_latent_state: tf.Tensor) -> tfd.Distribution:
        """
        Retrieves a probability distribution P(r|z, a, z') over rewards obtained when action a is chosen in z.
        """
        [reward_mean, reward_raw_covariance] = self.reward_network([latent_state, action, next_latent_state])
        return tfd.MultivariateNormalDiag(
            loc=reward_mean,
            scale_diag=self.scale_activation(reward_raw_covariance),
        )

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        return tfd.OneHotCategorical(logits=self.latent_policy_network(latent_state))

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

    @tf.function
    def call(self, inputs, training=None, mask=None, metrics=True):
        if self.latent_policy_training_phase:
            return self.latent_policy_training(inputs)

        state, label, action, reward, next_state, next_label = inputs

        # Logistic samples
        state_encoder_distribution = self.relaxed_encoding(state, self.encoder_temperature)
        next_state_encoder_distribution = self.relaxed_encoding(next_state, self.encoder_temperature)

        # Sigmoid of Logistic samples with location alpha/t and scale 1/t gives Relaxed Bernoulli
        # samples of location alpha and temperature t
        latent_state = tf.concat([label, tf.sigmoid(state_encoder_distribution.sample())], axis=-1)
        next_logistic_latent_state = next_state_encoder_distribution.sample()

        log_q_encoding = next_state_encoder_distribution.log_prob(next_logistic_latent_state)
        log_p_transition = self.relaxed_latent_transition_probability_distribution(
            latent_state, action, temperature=self.prior_temperature
        ).log_prob(next_label, next_logistic_latent_state)
        rate = log_q_encoding - log_p_transition

        # retrieve Relaxed Bernoulli samples
        next_latent_state = tf.concat([next_label, tf.sigmoid(next_logistic_latent_state)], axis=-1)

        if self.latent_policy_network is not None and self.full_optimization:
            # log P(a, r, s' | z, z') =  log π(a | z) + log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.discrete_latent_policy(latent_state),
                lambda _action: self.reward_probability_distribution(latent_state, _action, next_latent_state),
                self.decode(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)
        else:
            # log P(r, s' | z, a, z') = log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.reward_probability_distribution(latent_state, action, next_latent_state),
                self.decode(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(reward, next_state)

        entropy_regularizer = self.entropy_regularizer(
            next_state,
            enforce_latent_space_spreading=(
                    self.marginal_entropy_regularizer_ratio > 0. and not self.latent_policy_training_phase),
            latent_states=next_latent_state)

        if metrics:
            self.loss_metrics['ELBO'](-1 * (distortion + rate))
            reconstruction_sample = reconstruction_distribution.sample()
            self.loss_metrics['state_mse'](next_state, reconstruction_sample[-1])
            self.loss_metrics['reward_mse'](reward, reconstruction_sample[-2])
            #  self.loss_metrics['decoder_variance'](state_distribution.variance())
            self.loss_metrics['distortion'](distortion)
            self.loss_metrics['rate'](rate)
            self.loss_metrics['annealed_rate'](self.kl_scale_factor * rate)
            self.loss_metrics['entropy_regularizer'](self.entropy_regularizer_scale_factor * entropy_regularizer)
            self.loss_metrics['transition_log_probs'](
                self.discrete_latent_transition_probability_distribution(tf.round(latent_state), action).log_prob(
                    next_label, tf.round(tf.sigmoid(next_logistic_latent_state))))
            predicted_next_label, predicted_next_logistic = self.relaxed_latent_transition_probability_distribution(
                latent_state, action, temperature=self.prior_temperature).sample()
            predicted_next_latent_state = tf.concat(
                [tf.cast(predicted_next_label, dtype=tf.float32), tf.sigmoid(predicted_next_logistic)], axis=-1)
            self.loss_metrics['predicted_next_state_mse'](
                next_state, self.decode(predicted_next_latent_state).sample())

        if 'action_mse' in self.loss_metrics:
            self.loss_metrics['action_mse'](action, reconstruction_sample[0])

        if debug:
            tf.print(latent_state, "sampled z")
            tf.print(next_logistic_latent_state, "sampled (logistic) z'")
            tf.print(next_latent_state, "sampled z'")
            tf.print(self.encoder_network([state, label]), "log locations[:-1] -- logits[:-1] of Q")
            tf.print(log_q_encoding, "Log Q(logistic z'|s', l')")
            tf.print(self.transition_network([latent_state, action]), "log-locations P_transition")
            tf.print(log_p_transition, "log P(logistic z'|z, a)")
            tf.print(self.discrete_latent_transition_probability_distribution(
                tf.round(latent_state), action
            ).prob(tf.round(tf.sigmoid(next_logistic_latent_state))), "P(round(z') | round(z), a)")
            tf.print(next_latent_state, "sampled z'")
            [reconstruction_mean, _, reconstruction_prior_components] = \
                self.reconstruction_network(next_latent_state)
            tf.print(reconstruction_mean, 'mean(s | z)')
            tf.print(reconstruction_prior_components, 'GMM: prior components')
            tf.print(log_q_encoding - log_p_transition, "log Q(z') - log P(z')")

        return [distortion, rate, entropy_regularizer]

    @tf.function
    def entropy_regularizer(
            self, state: tf.Tensor,
            enforce_latent_space_spreading: bool = False,
            latent_states: Optional[tf.Tensor] = None
    ):
        logits = self.encoder_network(state)

        for metric_label in ('encoder_entropy', 'state_encoder_entropy'):
            if metric_label in self.loss_metrics:
                self.loss_metrics[metric_label](tfd.Independent(tfd.Bernoulli(logits=logits)).entropy())

        #  if enforce_latent_space_spreading:
        batch_size = tf.shape(logits)[0]
        marginal_encoder = tfd.Independent(
            tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(logits=tf.ones(shape=batch_size)),
                components_distribution=tfd.RelaxedBernoulli(
                    logits=tf.transpose(logits), temperature=self.encoder_temperature),
                reparameterize=(latent_states is None),
                allow_nan_stats=False
            )
        )
        if latent_states is None:
            latent_states = marginal_encoder.sample(batch_size)
        else:
            latent_states = latent_states[..., self.atomic_props_dims:]
        latent_states = tf.clip_by_value(latent_states, clip_value_min=1e-7, clip_value_max=1. - 1e-7)
        marginal_entropy_regularizer = tf.reduce_mean(marginal_encoder.log_prob(latent_states))
        #  regularizer = ((1. - self.marginal_entropy_regularizer_ratio) * regularizer +
        #                 self.marginal_entropy_regularizer_ratio *
        #                 (tf.abs(self.entropy_regularizer_scale_factor) / self.entropy_regularizer_scale_factor)
        #                 * marginal_entropy_regularizer)

        if 'marginal_encoder_entropy' in self.loss_metrics:
            self.loss_metrics['marginal_encoder_entropy'](-1. * marginal_entropy_regularizer)

        return marginal_entropy_regularizer

    def latent_policy_training(self, inputs):
        state, label, action, _, next_state, next_label = inputs
        latent_distribution = self.relaxed_encoding(state, label, temperature=self.encoder_temperature)
        next_latent_distribution = self.relaxed_encoding(next_state, next_label, temperature=self.encoder_temperature)
        latent_state = latent_distribution.sample()
        next_latent_state = next_latent_distribution.sample()

        latent_policy_distribution = self.discrete_latent_policy(latent_state)
        latent_markov_chain_transition_distribution = self.relaxed_latent_transition_probability_distribution(
            latent_state, action=None, temperature=self.prior_temperature)

        if 'action_mse' in self.loss_metrics:
            self.loss_metrics['action_mse'](action, latent_policy_distribution.sample())

        return [-1. * latent_policy_distribution.log_prob(action),
                -1. * tf.reduce_sum(latent_markov_chain_transition_distribution.log_prob(next_latent_state), axis=1),
                0.]

    def eval(self, inputs):
        """
        Evaluate the ELBO by making use of a discrete latent space.
        """
        state, label, action, reward, next_state, next_label = inputs

        latent_distribution = self.binary_encode(state)
        next_latent_distribution = self.binary_encode(next_state)
        latent_state = tf.concat([label, tf.cast(latent_distribution.sample(), tf.float32)], axis=-1)
        next_latent_state_no_label = tf.cast(next_latent_distribution.sample(), tf.float32)

        transition_distribution = self.discrete_latent_transition_probability_distribution(latent_state, action)
        # rate = next_latent_distribution.kl_divergence(transition_distribution)
        rate = next_latent_distribution.log_prob(next_latent_state_no_label) - transition_distribution.log_prob(
            next_label, next_latent_state_no_label)

        next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        if self.latent_policy_network is not None and self.full_optimization:
            # log P(a, r, s' | z, z') =  log π(a | z) + log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.discrete_latent_policy(latent_state),
                lambda _action: self.reward_probability_distribution(latent_state, _action, next_latent_state),
                self.decode(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)
        else:
            # log P(r, s' | z, a, z') = log P(r | z, a, z') + log P(s' | z')
            reconstruction_distribution = tfd.JointDistributionSequential([
                self.reward_probability_distribution(latent_state, action, next_latent_state),
                self.decode(next_latent_state)
            ])
            distortion = -1. * reconstruction_distribution.log_prob(reward, next_state)

        return (
            tf.reduce_mean(-1. * (distortion + rate)),
            tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            None
        )

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        """
        Compute the mean number of bits used to represent the latent space of the vae_mdp for the given dataset batch.
        This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
        """
        mean_bits_used = 0
        s, l, _, _, _, _ = inputs
        if deterministic:
            mean = tf.reduce_mean(tf.cast(self.binary_encode(s, l).mode(), dtype=tf.float32), axis=0)
        else:
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
        if self._entropy_regularizer_scale_factor is None:
            self._entropy_regularizer_scale_factor = tf.Variable(value, dtype=tf.float32, trainable=False)
        else:
            self._entropy_regularizer_scale_factor.assign(value)

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
        if self._kl_scale_factor is None:
            self._kl_scale_factor = tf.Variable(value, dtype=tf.float32, trainable=False)
        else:
            self._kl_scale_factor.assign(value)

    @property
    def kl_growth_rate(self):
        return self._kl_growth_rate

    @kl_growth_rate.setter
    def kl_growth_rate(self, value):
        if self._initial_kl_scale_factor is None:
            self._initial_kl_scale_factor = tf.Variable(self.kl_scale_factor, dtype=tf.float32, trainable=False)
        else:
            self._initial_kl_scale_factor.assign(self.kl_scale_factor)
        if self._decay_kl_scale_factor is None:
            self._decay_kl_scale_factor = tf.Variable(1., dtype=tf.float32, trainable=False)
        else:
            self._decay_kl_scale_factor.assign(1.)
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
    def latent_policy_update(self, x, optimizer):
        return self._compute_apply_gradients(x, optimizer, self.latent_policy_network.trainable_variables)

    @tf.function
    def generator_update(self, x, optimizer):
        return self._compute_apply_gradients(x, optimizer, self.generator_variables)

    def train_from_policy(
            self,
            policy: tf_agents.policies.tf_policy.Base,
            environment_suite,
            env_name: str,
            labeling_function: Callable,
            epsilon_greedy: Optional[float] = 0.,
            epsilon_greedy_decay_rate: Optional[float] = -1.,
            discrete_action_space: bool = False,
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
            reset_kl_scale_factor: Optional[float] = None,
            reset_entropy_regularizer: Optional[float] = None,
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
        print("Step: {}".format(start_step))

        if start_step < start_annealing_step:
            if reset_kl_scale_factor is not None:
                self.kl_scale_factor = reset_kl_scale_factor
            if reset_entropy_regularizer is not None:
                self.entropy_regularizer_scale_factor = reset_entropy_regularizer

        progressbar = Progbar(
            target=None,
            stateful_metrics=list(self.loss_metrics.keys()) + [
                'loss', 't_1', 't_2', 'entropy_regularizer_scale_factor', 'step', "num_episodes", "env_steps",
                "replay_buffer_frames", 'kl_annealing_scale_factor', "decoder_jsdiv", 'state_rate',
                "state_distortion", 'action_rate', 'action_distortion', 'mean_state_bits_used'],
            interval=0.1) if display_progressbar else None

        discrete_action_space = discrete_action_space and (self.latent_policy_network is not None)
        load_environment = lambda: environment_suite.load(env_name)

        if parallelization:
            tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [load_environment] * num_parallel_call))
            tf_env.reset()
        else:
            py_env = load_environment()
            py_env.reset()
            tf_env = tf_py_environment.TFPyEnvironment(py_env)

        if epsilon_greedy > 0.:
            epsilon_greedy = tf.Variable(epsilon_greedy, trainable=False, dtype=tf.float32)

            if epsilon_greedy_decay_rate == -1:
                epsilon_greedy_decay_rate = 1. - tf.exp((tf.math.log(1e-3) - tf.math.log(epsilon_greedy))
                                                        / (3. * (num_iterations - start_annealing_step) / 5.))
            epsilon_greedy.assign(
                epsilon_greedy * tf.pow(1. - epsilon_greedy_decay_rate,
                                        tf.math.maximum(0.,
                                                        tf.cast(global_step, dtype=tf.float32) - start_annealing_step)))

            @tf.function
            def _epsilon():
                if tf.greater(global_step, start_annealing_step):
                    epsilon_greedy.assign(epsilon_greedy * (1 - epsilon_greedy_decay_rate))
                return epsilon_greedy

            policy = tf_agents.policies.epsilon_greedy_policy.EpsilonGreedyPolicy(policy, _epsilon)

        # specs
        trajectory_spec = trajectory.from_transition(tf_env.time_step_spec(),
                                                     policy.policy_step_spec,
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
                eval_env, get_policy_evaluation(), num_episodes=policy_evaluation_num_episodes)

        driver.run = common.function(driver.run)

        print("Initial collect steps...")
        initial_collect_driver.run()
        print("Start training.")

        def dataset_generator():
            generator = ErgodicMDPTransitionGenerator(
                labeling_function,
                replay_buffer,
                discrete_action=discrete_action_space,
                num_discrete_actions=self.action_shape[0])
            return replay_buffer.as_dataset(
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                num_steps=2
            ).map(
                map_func=generator,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
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

            additional_training_metrics = {
                "num_episodes": num_episodes.result(),
                "env_steps": env_steps.result(),
                "replay_buffer_frames": replay_buffer.num_frames()} if not parallelization else {
                "replay_buffer_frames": replay_buffer.num_frames(),
            }
            if epsilon_greedy > 0.:
                additional_training_metrics['epsilon_greedy'] = epsilon_greedy

            loss = self.training_step(
                dataset_batch=next(dataset_iterator), batch_size=batch_size, optimizer=optimizer,
                annealing_period=annealing_period, global_step=global_step,
                dataset_size=replay_buffer.num_frames().numpy(), display_progressbar=display_progressbar,
                start_step=start_step, epoch=0, progressbar=progressbar, dataset_generator=dataset_generator,
                save_model_interval=save_model_interval,
                eval_ratio=eval_steps * batch_size / replay_buffer.num_frames(),
                save_directory=save_directory, log_name=log_name, train_summary_writer=train_summary_writer,
                log_interval=log_interval, manager=manager, logs=logs, start_annealing_step=start_annealing_step,
                additional_metrics=additional_training_metrics,
                eval_policy_driver=policy_evaluation_driver,
                aggressive_training=aggressive_training and global_step.numpy() < aggressive_training_steps,
                aggressive_update=aggressive_inference_optimization,
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

        tf_env.close()
        del tf_env
        return 0

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
        if not aggressive_training and not self.latent_policy_training_phase:
            gradients = self.compute_apply_gradients(dataset_batch, optimizer)
        elif not aggressive_training and self.latent_policy_training_phase:
            gradients = self.latent_policy_update(dataset_batch, optimizer)
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
                eval_policy(eval_policy_driver, train_summary_writer, global_step)
                #  eval_policy_thread = threading.Thread(
                #      target=eval_policy,
                #      args=(eval_policy_driver, train_summary_writer, global_step),
                #      daemon=True,
                #      name='eval')
                #  eval_policy_thread.start()

            if train_summary_writer is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar('eval_elbo', eval_elbo.result(), step=global_step)
                    for value in ('state', 'action'):
                        if data[value] is not None:
                            if value == 'state':
                                data[value] = tf.reduce_sum(
                                        data[value] * 2 ** tf.range(self.latent_state_size), axis=-1)
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

    def wrap_tf_environment(
            self,
            tf_env: tf_environment.TFEnvironment,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            deterministic_embedding_functions: bool = True
    ) -> tf_environment.TFEnvironment:

        class VariationalTFEnvironmentDiscretizer(tf_environment.TFEnvironment):

            def __init__(
                    self,
                    vae_mdp: VariationalMarkovDecisionProcess,
                    tf_env: tf_environment.TFEnvironment,
                    labeling_function: Callable[[tf.Tensor], tf.Tensor],
                    deterministic_state_embedding: bool = True
            ):
                action_spec = tf_env.action_spec()
                observation_spec = specs.BoundedTensorSpec(
                    shape=(vae_mdp.latent_state_size,),
                    dtype=tf.int32,
                    minimum=0,
                    maximum=1,
                    name='latent_observation'
                )
                time_step_spec = ts.time_step_spec(observation_spec)
                super(VariationalTFEnvironmentDiscretizer, self).__init__(
                    time_step_spec=time_step_spec,
                    action_spec=action_spec,
                    batch_size=tf_env.batch_size
                )

                self.embed_observation = vae_mdp.binary_encode
                self.tf_env = tf_env
                self._labeling_function = common.function(labeling_function)
                self.observation_shape, self.action_shape, self.reward_shape = [
                    vae_mdp.state_shape,
                    vae_mdp.action_shape,
                    vae_mdp.reward_shape
                ]
                self._current_latent_state = None
                if deterministic_state_embedding:
                    self._get_embedding = lambda distribution: distribution.mode()
                else:
                    self._get_embedding = lambda distribution: distribution.sample()
                self.deterministic_state_embedding = deterministic_state_embedding

            def labeling_function(self, state: tf.Tensor):
                label = tf.cast(self._labeling_function(state), dtype=tf.float32)
                # take the reset label into account
                label = tf.cond(
                    tf.rank(label) == 1,
                    lambda: tf.expand_dims(label, axis=-1),
                    lambda: label)
                return tf.concat(
                    [label, tf.zeros(shape=tf.concat([tf.shape(label)[:-1], tf.constant([1], dtype=tf.int32)], axis=-1),
                                     dtype=tf.float32)],
                    axis=-1)

            def _current_time_step(self):
                if self._current_latent_state is None:
                    return self.reset()
                time_step = self.tf_env.current_time_step()
                return trajectories.time_step.TimeStep(
                    time_step.step_type, time_step.reward, time_step.discount, self._current_latent_state)

            def _step(self, action):
                time_step = self.tf_env.step(action)
                label = self.labeling_function(time_step.observation)

                latent_state = self._get_embedding(self.embed_observation(time_step.observation, label))
                self._current_latent_state = latent_state
                return self._current_time_step()

            def _reset(self):
                time_step = self.tf_env.reset()
                label = self.labeling_function(time_step.observation)
                self._current_latent_state = self._get_embedding(self.embed_observation(time_step.observation, label))
                return self._current_time_step()

            def render(self):
                return self.tf_env.render()

        return VariationalTFEnvironmentDiscretizer(self, tf_env, labeling_function, deterministic_embedding_functions)

    def get_latent_policy(self) -> tf_policy.Base:

        assert self.latent_policy_network is not None
        action_spec = specs.BoundedTensorSpec(
            shape=(),
            dtype=tf.int32,
            minimum=0,
            maximum=self.number_of_discrete_actions - 1,
            name='action'
        )
        observation_spec = specs.BoundedTensorSpec(
            shape=(self.latent_state_size.numpy(),),
            dtype=tf.int32,
            minimum=0,
            maximum=1,
            name='observation'
        )
        time_step_spec = ts.time_step_spec(observation_spec)

        class LatentPolicy(tf_policy.Base):

            def __init__(self, time_step_spec, action_spec, discrete_latent_policy):
                super().__init__(time_step_spec, action_spec)
                self.discrete_latent_policy = discrete_latent_policy

            def _distribution(self, time_step, policy_state):
                one_hot_categorical_distribution = self.discrete_latent_policy(
                    tf.cast(time_step.observation, dtype=tf.float32))
                return PolicyStep(tfd.Categorical(logits=one_hot_categorical_distribution.logits_parameter()), (), ())

        return LatentPolicy(time_step_spec, action_spec, self.discrete_latent_policy)


def load(tf_model_path: str, discrete_action=False) -> VariationalMarkovDecisionProcess:
    model = tf.saved_model.load(tf_model_path)
    print(model.signatures)
    if discrete_action:
        return VariationalMarkovDecisionProcess(
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_1'].shape)[2:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_2'].shape)[2:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_3'].shape)[2:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_5'].shape)[2:],
            encoder_network=model.encoder_network,
            transition_network=model.transition_network,
            reward_network=model.reward_network,
            decoder_network=model.reconstruction_network,
            label_transition_network=model.label_transition_network,
            latent_policy_network=model.latent_policy_network,
            latent_state_size=model.latent_state_size,
            encoder_temperature=model._encoder_temperature,
            prior_temperature=model._prior_temperature,
            entropy_regularizer_scale_factor=model._entropy_regularizer_scale_factor,
            kl_scale_factor=model._kl_scale_factor,
            pre_loaded_model=True)
    else:
        return VariationalMarkovDecisionProcess(
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_1'].shape)[1:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_2'].shape)[1:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_3'].shape)[1:],
            tuple(model.signatures['serving_default'].structured_input_signature[1]['input_5'].shape)[1:],
            encoder_network=model.encoder_network,
            transition_network=model.transition_network,
            reward_network=model.reward_network,
            decoder_network=model.reconstruction_network,
            latent_state_size=model.latent_state_size,
            label_transition_network=model.label_transition_network,
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
