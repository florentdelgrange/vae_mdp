import os
from typing import Tuple, Optional, List, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Concatenate, Reshape, Dense, Lambda
from tf_agents import trajectories, specs
from tf_agents.environments import tf_environment, tf_py_environment
from tf_agents.trajectories import time_step as ts

import variational_mdp
from util.io import dataset_generator
from variational_mdp import VariationalMarkovDecisionProcess
from variational_mdp import epsilon
from verification.local_losses import estimate_local_losses_from_samples

tfd = tfp.distributions
tfb = tfp.bijectors


class VariationalActionDiscretizer(VariationalMarkovDecisionProcess):

    def __init__(
            self,
            vae_mdp: VariationalMarkovDecisionProcess,
            number_of_discrete_actions: int,
            action_encoder_network: Model,
            action_decoder_network: Model,
            transition_network: Model,
            reward_network: Model,
            latent_policy_network: Model,
            action_label_transition_network: Model,
            pre_processing_network: Model = Sequential(
                [Dense(units=256, activation='relu'),
                 Dense(units=256, activation='relu')],
                name='variational_action_discretizer_pre_processing_network'),
            branching_action_networks: bool = False,
            encoder_temperature: Optional[float] = None,
            prior_temperature: Optional[float] = None,
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            pre_loaded_model: bool = False,
            one_output_per_action: bool = False,
            relaxed_state_encoding: bool = False,
            full_optimization: bool = True,
            reconstruction_mixture_components: int = 1,
            action_regularizer_scaling: float = 1.,
            importance_sampling_exponent: Optional[float] = None,
            importance_sampling_exponent_growth_rate: Optional[float] = None
    ):

        super().__init__(
            state_shape=vae_mdp.state_shape, action_shape=vae_mdp.action_shape, reward_shape=vae_mdp.reward_shape,
            label_shape=vae_mdp.label_shape, encoder_network=vae_mdp.encoder_network,
            transition_network=vae_mdp.transition_network, label_transition_network=vae_mdp.label_transition_network,
            reward_network=vae_mdp.reward_network,
            decoder_network=vae_mdp.reconstruction_network,
            time_stacked_states=vae_mdp.time_stacked_states,
            latent_state_size=vae_mdp.latent_state_size,
            encoder_temperature=vae_mdp.encoder_temperature.numpy(),
            prior_temperature=vae_mdp.prior_temperature.numpy(),
            encoder_temperature_decay_rate=vae_mdp.encoder_temperature_decay_rate.numpy(),
            prior_temperature_decay_rate=vae_mdp.prior_temperature_decay_rate.numpy(),
            entropy_regularizer_scale_factor=vae_mdp.entropy_regularizer_scale_factor.numpy(),
            entropy_regularizer_decay_rate=vae_mdp.entropy_regularizer_decay_rate.numpy(),
            entropy_regularizer_scale_factor_min_value=vae_mdp.entropy_regularizer_scale_factor_min_value.numpy(),
            marginal_entropy_regularizer_ratio=vae_mdp.marginal_entropy_regularizer_ratio,
            kl_scale_factor=vae_mdp.kl_scale_factor.numpy(),
            kl_annealing_growth_rate=vae_mdp.kl_growth_rate.numpy(),
            mixture_components=vae_mdp.mixture_components,
            multivariate_normal_raw_scale_diag_activation=vae_mdp.scale_activation,
            multivariate_normal_full_covariance=vae_mdp.full_covariance,
            pre_loaded_model=True,
            importance_sampling_exponent=vae_mdp.is_exponent,
            importance_sampling_exponent_growth_rate=vae_mdp.is_exponent_growth_rate,
            optimizer=vae_mdp._optimizer,
            evaluation_window_size=tf.shape(vae_mdp.evaluation_window)[0],
            evaluation_criterion=vae_mdp.evaluation_criterion)

        if encoder_temperature is None:
            encoder_temperature = 1. / (number_of_discrete_actions - 1)
        if prior_temperature is None:
            prior_temperature = encoder_temperature / 1.5

        self.number_of_discrete_actions = number_of_discrete_actions
        self._state_vae = vae_mdp
        self.one_output_per_action = one_output_per_action
        self.relaxed_state_encoding = relaxed_state_encoding or full_optimization
        self.full_optimization = full_optimization
        self.mixture_components = reconstruction_mixture_components

        self.state_encoder_temperature = self.encoder_temperature
        self.state_prior_temperature = self.prior_temperature
        self.state_encoder_temperature_decay_rate = self.encoder_temperature_decay_rate
        self.state_prior_temperature_decay_rate = self.prior_temperature_decay_rate

        self.encoder_temperature = tf.Variable(encoder_temperature, dtype=tf.float32, trainable=False)
        self.prior_temperature = tf.Variable(prior_temperature, dtype=tf.float32, trainable=False)
        self.encoder_temperature_decay_rate = tf.constant(encoder_temperature_decay_rate, dtype=tf.float32)
        self.prior_temperature_decay_rate = tf.constant(prior_temperature_decay_rate, dtype=tf.float32)
        self._action_regularizer_scaling = tf.constant(action_regularizer_scaling, dtype=tf.float32)
        if importance_sampling_exponent is not None:
            self.is_exponent = importance_sampling_exponent
        if importance_sampling_exponent_growth_rate is not None:
            self.is_exponent_growth_rate = importance_sampling_exponent_growth_rate

        def clone_model(model: tf.keras.Model, copy_name: str = ''):
            model = model_from_json(model.to_json(), custom_objects={'leaky_relu': tf.nn.leaky_relu})
            for layer in model.layers:
                layer._name = copy_name + '_' + layer.name
            model._name = copy_name + '_' + model.name
            return model

        if not pre_loaded_model:
            # action encoder network
            latent_state = Input(shape=(self.latent_state_size,))
            action = Input(shape=self.action_shape)
            next_latent_state = Input(shape=(self.latent_state_size,))
            next_label = Input(shape=(self.atomic_props_dims,))
            latent_action = Input(shape=(number_of_discrete_actions,)) if not one_output_per_action else None

            action_encoder = Concatenate(name="variational_action_discretizer_action_encoder_input")(
                [latent_state, action])
            action_encoder = action_encoder_network(action_encoder)
            action_encoder = Dense(
                units=number_of_discrete_actions,
                activation=None,
                name='variational_action_discretizer_action_encoder_exp_one_hot_logits'
            )(action_encoder)
            self.action_encoder = Model(
                inputs=[latent_state, action],
                outputs=action_encoder,
                name="variational_action_discretizer_action_encoder")

            # prior over actions
            self.latent_policy_network = latent_policy_network(latent_state)
            self.latent_policy_network = Dense(
                units=self.number_of_discrete_actions,
                activation=None,
                name='variational_action_discretizer_latent_policy_exp_one_hot_logits'
            )(self.latent_policy_network)
            self.latent_policy_network = Model(
                inputs=latent_state,
                outputs=self.latent_policy_network,
                name='variational_action_discretizer_latent_policy_network')

            # discrete actions transition network
            if not one_output_per_action:
                _transition_network = Concatenate()([latent_state, latent_action, next_label])
                _transition_network = transition_network(_transition_network)
                _transition_network = Dense(
                    units=self.latent_state_size - self.atomic_props_dims,
                    activation=None,
                    name='variational_action_discretizer_discrete_action_transition_next_state_logits'
                )(_transition_network)
                self.action_transition_network = Model(
                    inputs=[latent_state, latent_action, next_label],
                    outputs=_transition_network,
                    name="variational_action_discretizer_transition_network")
            else:
                transition_network_input = Concatenate()([latent_state, next_label])
                transition_network_pre_processing = clone_model(
                    pre_processing_network, 'transition')(transition_network_input)
                transition_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        _transition_network = clone_model(transition_network, str(action))
                    else:
                        _transition_network = transition_network
                    _transition_network = _transition_network(transition_network_pre_processing)
                    _transition_network = Dense(
                        units=self.latent_state_size - self.atomic_props_dims,
                        activation=None,
                        name='variational_action_discretizer_transition_next_state_logits_action{}'.format(action)
                    )(_transition_network)
                    transition_outputs.append(_transition_network)
                self.action_transition_network = Model(
                    inputs=[latent_state, next_label],
                    outputs=Lambda(
                        lambda x: tf.stack(x, axis=1),
                        name="variational_action_discretizer_transition_network_output"
                    )(transition_outputs),
                    name="variational_action_discretizer_transition_network")

            # label transition network
            if not one_output_per_action:
                _label_transition_network = Concatenate()([latent_state, latent_action])
                _label_transition_network = action_label_transition_network(_label_transition_network)
                _label_transition_network = Dense(
                    units=self.atomic_props_dims,
                    name='variational_action_discretizer_next_label_transition_logits')(_label_transition_network)
                self.action_label_transition_network = Model(
                    inputs=[latent_state, latent_action],
                    outputs=_label_transition_network,
                    name='variational_action_discretizer_label_transition_network')
            else:
                if branching_action_networks:
                    _action_label_transition_network = clone_model(action_label_transition_network, str(action))
                else:
                    _action_label_transition_network = action_label_transition_network
                _label_transition_network = _action_label_transition_network(latent_state)
                outputs = []
                for action in range(self.number_of_discrete_actions):
                    _label_transition_network = Dense(
                        units=self.atomic_props_dims,
                        name='variational_action_discretizer_next_label_transition_logits_action{}'.format(action)
                    )(_label_transition_network)
                    outputs.append(_label_transition_network)
                self.action_label_transition_network = Model(
                    inputs=latent_state,
                    outputs=Lambda(
                        lambda x: tf.stack(x, axis=1),
                        name='variational_action_discretizer_label_transition_network_output')(outputs),
                    name='variational_action_discretizer_label_transition_network')

            # discrete actions reward network
            if not one_output_per_action:
                _reward_network = Concatenate()([latent_state, latent_action, next_latent_state])
                _reward_network = reward_network(_reward_network)
                reward_mean = Dense(
                    units=np.prod(self.reward_shape),
                    activation=None,
                    name='variational_action_discretizer_action_reward_mean_0'
                )(_reward_network)
                reward_mean = Reshape(self.reward_shape, name='variational_action_discretizer_action_reward_mean')(
                    reward_mean)
                reward_raw_covar = Dense(
                    units=np.prod(self.reward_shape),
                    activation=None,
                    name='variational_action_discretizer_action_reward_raw_diag_covariance_0'
                )(_reward_network)
                reward_raw_covar = Reshape(
                    self.reward_shape,
                    name='variational_action_discretizer_action_reward_raw_diag_covariance'
                )(reward_raw_covar)
                self.action_reward_network = Model(
                    inputs=[latent_state, latent_action, next_latent_state],
                    outputs=[reward_mean, reward_raw_covar],
                    name="variational_action_discretizer_reward_network")
            else:
                reward_network_input = Concatenate()([latent_state, next_latent_state])
                reward_network_pre_processing = clone_model(pre_processing_network, 'reward')(reward_network_input)
                reward_network_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        _reward_network = clone_model(reward_network, str(action))
                    else:
                        _reward_network = reward_network
                    _reward_network = _reward_network(reward_network_pre_processing)
                    reward_mean = Dense(
                        units=np.prod(self.reward_shape),
                        activation=None,
                        name='variational_action_discretizer_reward_mean_0_action{}'.format(action)
                    )(_reward_network)
                    reward_mean = Reshape(
                        self.reward_shape,
                        name='variational_action_discretizer_action{}_reward_mean'.format(action))(reward_mean)
                    reward_raw_covar = Dense(
                        units=np.prod(self.reward_shape),
                        activation=None,
                        name='variational_action_discretizer_reward_raw_diag_covariance_0_action{}'.format(action)
                    )(_reward_network)
                    reward_raw_covar = Reshape(
                        self.reward_shape,
                        name='variational_action_discretizer_reward_raw_diag_covariance_action{}'.format(action)
                    )(reward_raw_covar)
                    reward_network_outputs.append([reward_mean, reward_raw_covar])
                reward_network_mean = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(mean for mean, covariance in reward_network_outputs))
                reward_network_raw_covariance = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(covariance for mean, covariance in reward_network_outputs))
                self.action_reward_network = Model(
                    inputs=[latent_state, next_latent_state],
                    outputs=[reward_network_mean, reward_network_raw_covariance],
                    name="variational_action_discretizer_discrete_actions_reward_network")

            # discrete actions decoder
            if self.mixture_components > 1:
                action_shape = (self.mixture_components,) + self.action_shape
            else:
                action_shape = self.action_shape
            if not one_output_per_action:
                action_decoder = Concatenate()([latent_state, latent_action])
                action_decoder = action_decoder_network(action_decoder)
                action_decoder_mean = Dense(
                    units=self.mixture_components * np.prod(self.action_shape),
                    name='variational_action_discretizer_action_decoder_mean_raw_output',
                    activation=None
                )(action_decoder)
                action_decoder_mean = Reshape(
                    target_shape=action_shape,
                    name='variational_action_discretizer_action_decoder_mean'
                )(action_decoder_mean)
                action_decoder_raw_covariance = Dense(
                    units=self.mixture_components * np.prod(self.action_shape),
                    name='variational_action_discretizer_action_decoder_raw_covariance_output',
                    activation=None
                )(action_decoder)
                action_decoder_raw_covariance = Reshape(
                    target_shape=action_shape,
                    name='variational_action_discretizer_action_decoder_diag_covariance'
                )(action_decoder_raw_covariance)
                action_decoder_mixture_categorical_logits = Dense(
                    units=self.mixture_components,
                    activation=None,
                    name='variational_action_discretizer_action_decoder_mixture_categorical_logits'
                )(action_decoder)
                self.action_decoder_network = Model(
                    inputs=[latent_state, latent_action],
                    outputs=[
                        action_decoder_mean,
                        action_decoder_raw_covariance,
                        action_decoder_mixture_categorical_logits
                    ],
                    name="variational_action_discretizer_action_decoder_network")
            else:
                action_decoder_pre_processing = clone_model(pre_processing_network, 'action')(latent_state)
                action_decoder_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        action_decoder = clone_model(action_decoder_network, str(action))
                    else:
                        action_decoder = action_decoder_network
                    action_decoder = action_decoder(action_decoder_pre_processing)
                    action_decoder_mean = Dense(
                        units=self.mixture_components * np.prod(self.action_shape),
                        activation=None
                    )(action_decoder)
                    action_decoder_mean = Reshape(
                        target_shape=action_shape,
                        name='variational_action_discretizer_action{}_decoder_mean'.format(action)
                    )(action_decoder_mean)
                    action_decoder_raw_covariance = Dense(
                        units=self.mixture_components * np.prod(self.action_shape),
                        activation=None
                    )(action_decoder)
                    action_decoder_raw_covariance = Reshape(
                        target_shape=action_shape,
                        name='variational_action_discretizer_action{}_decoder_raw_diag_covariance'.format(action)
                    )(action_decoder_raw_covariance)
                    action_decoder_mixture_categorical_logits = Dense(
                        units=self.mixture_components,
                        activation=None,
                        name='variational_action_discretizer_action{}_decoder_mixture_categorical_logits'.format(action)
                    )(action_decoder)
                    action_decoder_outputs.append(
                        (action_decoder_mean, action_decoder_raw_covariance, action_decoder_mixture_categorical_logits)
                    )
                action_decoder_mean = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(mean for mean, covariance, component_logits in action_decoder_outputs))
                action_decoder_raw_covariance = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(covariance for mean, covariance, component_logits in action_decoder_outputs))
                action_decoder_mixture_categorical_logits = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(component_logits for mean, covariance, component_logits in action_decoder_outputs))
                self.action_decoder_network = Model(
                    inputs=latent_state,
                    outputs=[
                        action_decoder_mean,
                        action_decoder_raw_covariance,
                        action_decoder_mixture_categorical_logits
                    ],
                    name="variational_action_discretizer_action_decoder_network")

        else:
            self.action_encoder = action_encoder_network
            self.latent_policy_network = latent_policy_network
            self.action_transition_network = transition_network
            self.action_reward_network = reward_network
            self.action_decoder_network = action_decoder_network
            self.action_label_transition_network = action_label_transition_network

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'action_mse': tf.keras.metrics.MeanSquaredError(name='action_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'entropy_regularizer': tf.keras.metrics.Mean(name='entropy_regularizer'),
            'transition_log_probs': tf.keras.metrics.Mean(name='transition_log_probs'),
            # 'decoder_divergence': tf.keras.metrics.Mean(name='decoder_divergence'),
        }
        if self.full_optimization:
            self.loss_metrics.update({
                'state_mse': tf.keras.metrics.MeanSquaredError(name='state_mse'),
                'state_encoder_entropy': tf.keras.metrics.Mean(name='encoder_entropy'),
                'marginal_encoder_entropy': tf.keras.metrics.Mean(name='marginal_encoder_entropy'),
                'action_encoder_entropy': tf.keras.metrics.Mean(name='action_encoder_entropy'),
                # 'state_decoder_variance': tf.keras.metrics.Mean('decoder_variance'),
                'state_rate': tf.keras.metrics.Mean(name='state_rate'),
                'action_rate': tf.keras.metrics.Mean(name='action_rate'),
                't_1_state': tf.keras.metrics.Mean(name='state_encoder_temperature'),
                't_2_state': tf.keras.metrics.Mean(name='state_prior_temperature')
            })

    def anneal(self):
        super().anneal()
        for var, decay_rate in [
            (self._state_vae.encoder_temperature, self._state_vae.encoder_temperature_decay_rate),
            (self._state_vae.prior_temperature, self._state_vae.prior_temperature_decay_rate),
        ]:
            if decay_rate.numpy().all() > 0:
                var.assign(var * (1. - decay_rate))

    def relaxed_action_encoding(
            self, latent_state: tf.Tensor, action: tf.Tensor, temperature: float
    ) -> tfd.Distribution:
        encoder_logits = self.action_encoder([latent_state, action])
        return tfd.ExpRelaxedOneHotCategorical(temperature=temperature, logits=encoder_logits, allow_nan_stats=False)

    def discrete_action_encoding(self, latent_state: tf.Tensor, action: tf.Tensor) -> tfd.Distribution:
        relaxed_distribution = self.relaxed_action_encoding(latent_state, action, 1e-5)
        log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
        return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor,
            relaxed_state_encoding: bool = False, log_latent_action: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            if relaxed_state_encoding:
                return tfd.JointDistributionSequential([
                    tfd.Independent(
                        tfd.Bernoulli(
                            logits=self.action_label_transition_network([latent_state, latent_action]),
                            allow_nan_stats=False,
                            dtype=tf.float32, )),
                    lambda _next_label: tfd.Independent(
                        tfd.Logistic(
                            loc=(self.action_transition_network([latent_state, latent_action, _next_label])
                                 / self._state_vae.prior_temperature),
                            scale=1. / self._state_vae.prior_temperature,
                            allow_nan_stats=False))])
            else:
                return tfd.JointDistributionSequential([
                    tfd.Independent(
                        tfd.Bernoulli(
                            logits=self.action_label_transition_network([latent_state, latent_action]),
                            allow_nan_stats=False,
                            dtype=tf.float32)),
                    lambda _next_label: tfd.Independent(
                        tfd.Bernoulli(
                            logits=self.action_transition_network([latent_state, latent_action, _next_label]),
                            allow_nan_stats=False))])
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action, allow_nan_stats=False)
            else:
                action_categorical = tfd.Categorical(probs=latent_action, allow_nan_stats=False)

            if relaxed_state_encoding:
                return tfd.JointDistributionSequential([
                    tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Bernoulli(
                                logits=self.action_label_transition_network(latent_state),
                                allow_nan_stats=False,
                                dtype=tf.float32),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False),
                    lambda _next_label: tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Logistic(
                                loc=(self.action_transition_network([latent_state, _next_label])
                                     / self._state_vae.prior_temperature),
                                scale=1. / self._state_vae.prior_temperature,
                                allow_nan_stats=False),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False)])
            else:
                return tfd.JointDistributionSequential([
                    tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Bernoulli(
                                logits=self.action_label_transition_network(latent_state),
                                allow_nan_stats=False,
                                dtype=tf.float32),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False),
                    lambda _next_label: tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.Independent(
                            tfd.Bernoulli(
                                logits=self.action_transition_network([latent_state, _next_label]),
                                allow_nan_stats=False),
                            reinterpreted_batch_ndims=1),
                        allow_nan_stats=False)])

    def reward_probability_distribution(
            self, latent_state, latent_action, next_latent_state, log_latent_action: bool = False,
            disable_mixture_distribution: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            [reward_mean, reward_raw_covariance] = self.action_reward_network(
                [latent_state, latent_action, next_latent_state])

            return tfd.MultivariateNormalDiag(
                loc=reward_mean,
                scale_diag=self.scale_activation(reward_raw_covariance),
                allow_nan_stats=False)
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action, allow_nan_stats=False)
            else:
                action_categorical = tfd.Categorical(probs=latent_action, allow_nan_stats=False)

            [reward_mean, reward_raw_covariance] = self.action_reward_network([latent_state, next_latent_state])

            if not log_latent_action and disable_mixture_distribution:
                # all actions are assumed to be given one-hot
                _latent_action = tf.cast(tf.stop_gradient(tf.argmax(latent_action, axis=-1)), dtype=tf.int32)
                return tfd.MultivariateNormalDiag(
                    loc=tf.stop_gradient(tf.map_fn(
                        fn=lambda i: reward_mean[i, _latent_action[i], ...],
                        elems=tf.range(tf.shape(_latent_action)[0]),
                        fn_output_signature=tf.float32)),
                    scale_diag=tf.stop_gradient(tf.map_fn(
                        fn=lambda i: self.scale_activation(reward_raw_covariance[i, _latent_action[i], ...]),
                        elems=tf.range(tf.shape(_latent_action)[0]),
                        fn_output_signature=tf.float32)),
                    allow_nan_stats=False)
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=action_categorical,
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=reward_mean,
                        scale_diag=self.scale_activation(reward_raw_covariance),
                        allow_nan_stats=False
                    ), allow_nan_stats=False)

    def decode_action(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, log_latent_action: bool = False,
            disable_mixture_distribution: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            [action_mean, action_raw_covariance, cat_logits] = self.action_decoder_network(
                [latent_state, latent_action])
            if self.mixture_components == 1:
                return tfd.MultivariateNormalDiag(
                    loc=action_mean,
                    scale_diag=self.scale_activation(action_raw_covariance),
                    allow_nan_stats=False)
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=cat_logits, allow_nan_stats=False),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=action_mean,
                        scale_diag=self.scale_activation(action_raw_covariance),
                        allow_nan_stats=False),
                    allow_nan_stats=False)
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action, allow_nan_stats=False)
            else:
                action_categorical = tfd.Categorical(probs=latent_action, allow_nan_stats=False)

            [action_mean, action_raw_covariance, cat_logits] = self.action_decoder_network(latent_state)

            if self.mixture_components == 1:
                if not log_latent_action and disable_mixture_distribution:
                    # all actions are assumed to be given one-hot
                    _latent_action = tf.cast(tf.stop_gradient(tf.argmax(latent_action, axis=-1)), dtype=tf.int32)
                    return tfd.MultivariateNormalDiag(
                        loc=tf.stop_gradient(tf.map_fn(
                            fn=lambda i: action_mean[i, _latent_action[i], ...],
                            elems=tf.range(tf.shape(_latent_action)[0]),
                            fn_output_signature=tf.float32)),
                        scale_diag=tf.stop_gradient(tf.map_fn(
                            fn=lambda i: self.scale_activation(action_raw_covariance[i, _latent_action[i], ...]),
                            elems=tf.range(tf.shape(_latent_action)[0]),
                            fn_output_signature=tf.float32)),
                        allow_nan_stats=False)
                else:
                    return tfd.MixtureSameFamily(
                        mixture_distribution=action_categorical,
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=action_mean,
                            scale_diag=self.scale_activation(action_raw_covariance),
                            allow_nan_stats=False),
                        allow_nan_stats=False)
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=action_categorical,
                    components_distribution=tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(logits=cat_logits),
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=action_mean,
                            scale_diag=self.scale_activation(action_raw_covariance),
                            allow_nan_stats=False),
                        allow_nan_stats=False),
                    allow_nan_stats=False)

    def relaxed_latent_policy(self, latent_state: tf.Tensor, temperature: float):
        return tfd.ExpRelaxedOneHotCategorical(
            temperature=temperature, logits=self.latent_policy_network(latent_state), allow_nan_stats=False)

    def discrete_latent_policy(self, latent_state: tf.Tensor):
        relaxed_distribution = self.relaxed_latent_policy(latent_state, temperature=1e-5)
        log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
        return tfd.OneHotCategorical(logits=log_probs, allow_nan_stats=False)

    def action_embedding_function(
            self,
            state: tf.Tensor,
            latent_action: tf.Tensor,
            label: Optional[tf.Tensor] = None,
            labeling_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
    ) -> tf.Tensor:
        if (label is None) == (labeling_function is None):
            raise ValueError("Must either pass a label or a labeling_function")

        if labeling_function is not None:
            label = labeling_function(state)

        return self.decode_action(
            latent_state=tf.cast(self.state_embedding_function(state, label=label), dtype=tf.float32),
            latent_action=tf.cast(tf.one_hot(latent_action, depth=self.number_of_discrete_actions), dtype=tf.float32),
            disable_mixture_distribution=True
        ).mode()

    @tf.function
    def __call__(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
    ):
        if self.full_optimization:
            return self._full_optimization_call(state, label, action, reward, next_state, next_label, sample_key)

        if self.relaxed_state_encoding:
            logistic_latent_state = self.relaxed_encoding(state, self._state_vae.encoder_temperature).sample()
            latent_state = tf.concat([label, tf.sigmoid(logistic_latent_state)], axis=-1)
            next_latent_state_no_label = self._state_vae.relaxed_encoding(
                next_state, self._state_vae.encoder_temperature).sample()
        else:
            latent_state = tf.concat([label, tf.cast(self.binary_encode(state).sample(), tf.float32)])
            next_latent_state_no_label = tf.cast(self.binary_encode(next_state).sample())
        q = self.relaxed_action_encoding(latent_state, action, self.encoder_temperature)
        p = self.relaxed_latent_policy(latent_state, self.prior_temperature)
        log_latent_action = q.sample()

        log_q_log_latent_action = q.log_prob(log_latent_action)
        log_p_log_latent_action = p.log_prob(log_latent_action)

        # transition probability reconstruction
        transition_probability_distribution = \
            self.discrete_latent_transition_probability_distribution(
                latent_state, log_latent_action,
                relaxed_state_encoding=self.relaxed_state_encoding, log_latent_action=True)
        if self.relaxed_state_encoding:
            continuous_action_transition = self._state_vae.relaxed_latent_transition_probability_distribution(
                latent_state, action, self._state_vae.prior_temperature)
        else:
            continuous_action_transition = self._state_vae.discrete_latent_transition_probability_distribution(
                latent_state, action)
        log_p_transition_action = continuous_action_transition.log_prob(next_label, next_latent_state_no_label)
        log_p_transition_latent_action = transition_probability_distribution.log_prob(
            next_label, next_latent_state_no_label)
        log_p_transition = log_p_transition_latent_action - log_p_transition_action

        if self.relaxed_state_encoding:
            next_latent_state = tf.concat([next_label, tf.sigmoid(next_latent_state_no_label)], axis=-1)
        else:
            next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        # rewards reconstruction
        reward_distribution = self.reward_probability_distribution(
            latent_state, log_latent_action, next_latent_state, log_latent_action=True)
        log_p_rewards_action = self._state_vae.reward_probability_distribution(
            latent_state, action, next_latent_state).log_prob(reward)
        log_p_rewards_latent_action = reward_distribution.log_prob(reward)
        log_p_rewards = log_p_rewards_latent_action - log_p_rewards_action

        # action reconstruction
        action_distribution = self.decode_action(latent_state, log_latent_action, log_latent_action=True)
        log_p_action = action_distribution.log_prob(action)

        rate = log_q_log_latent_action - log_p_log_latent_action
        distortion = -1. * (log_p_action + log_p_rewards + log_p_transition)

        entropy_regularizer = self.entropy_regularizer(latent_state, action)

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(distortion + rate)))

        # metrics
        self.loss_metrics['ELBO'](tf.stop_gradient(-1. * (distortion + rate)))
        self.loss_metrics['action_mse'](action, tf.stop_gradient(action_distribution.sample()))
        self.loss_metrics['reward_mse'](reward, tf.stop_gradient(reward_distribution.sample()))
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        self.loss_metrics['annealed_rate'](tf.stop_gradient(self.kl_scale_factor * rate))
        self.loss_metrics['entropy_regularizer'](
            tf.stop_gradient(self.entropy_regularizer_scale_factor * entropy_regularizer))
        # if self.one_output_per_action:
        #     self.loss_metrics['decoder_divergence'](self._compute_decoder_jensen_shannon_divergence(z, a_1))

        if variational_mdp.debug:
            tf.print(latent_state, "sampled z", summarize=variational_mdp.debug_verbosity)
            tf.print(next_latent_state, "sampled z'", summarize=variational_mdp.debug_verbosity)
            tf.print(q.logits, "logits of Q_action", summarize=variational_mdp.debug_verbosity)
            tf.print(p.logits, "logits of P_action", summarize=variational_mdp.debug_verbosity)
            tf.print(log_latent_action, "sampled log action from Q", summarize=variational_mdp.debug_verbosity)
            tf.print(log_q_log_latent_action, "log Q(exp_action)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_log_latent_action, "log P(exp_action)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_rewards, "log P(r | z, â, z')", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_transition, "log P(z' | z, â)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_action, "log P(a | z, â)", summarize=variational_mdp.debug_verbosity)

        return {'distortion': distortion, 'rate': rate, 'entropy_regularizer': entropy_regularizer}

    def _full_optimization_call(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None
    ):
        # sampling from encoder distributions
        latent_state_encoder = self._state_vae.relaxed_encoding(state, self._state_vae.encoder_temperature)
        next_latent_state_encoder = self._state_vae.relaxed_encoding(next_state, self._state_vae.encoder_temperature)
        latent_state = tf.concat([label, tf.sigmoid(latent_state_encoder.sample())], axis=-1)
        next_logistic_latent_state = next_latent_state_encoder.sample()
        latent_action_encoder = self.relaxed_action_encoding(latent_state, action, self.encoder_temperature)
        latent_policy = self.relaxed_latent_policy(latent_state, self.prior_temperature)
        log_latent_action = latent_action_encoder.sample()

        # action encoder rate
        log_q_log_latent_action = latent_action_encoder.log_prob(log_latent_action)
        log_pi_log_latent_action = latent_policy.log_prob(log_latent_action)
        action_encoder_rate = log_q_log_latent_action - log_pi_log_latent_action

        # transitions
        transition_probability_distribution = self.discrete_latent_transition_probability_distribution(
            latent_state, log_latent_action, relaxed_state_encoding=True, log_latent_action=True)
        log_p_next_latent_state = transition_probability_distribution.log_prob(next_label, next_logistic_latent_state)

        # state encoder rate
        log_q_next_latent_state = next_latent_state_encoder.log_prob(next_logistic_latent_state)
        state_encoder_rate = log_q_next_latent_state - log_p_next_latent_state

        rate = state_encoder_rate + action_encoder_rate

        next_latent_state = tf.concat([next_label, tf.sigmoid(next_logistic_latent_state)], axis=-1)

        # Reconstruction
        # log P(a, r, s' | z, â, z') = log P(a | z, â) + log P(r | z, â, z') + log P(s' | z')
        reconstruction_distribution = tfd.JointDistributionSequential([
            self.decode_action(latent_state, log_latent_action, log_latent_action=True),
            self.reward_probability_distribution(
                latent_state, log_latent_action, next_latent_state, log_latent_action=True),
            self.decode(next_latent_state)
        ])

        distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)

        entropy_regularizer = self.entropy_regularizer(
            latent_state, action, state=state,
            use_marginal_entropy=True)

        # priority support
        if self.priority_handler is not None and sample_key is not None:
            tf.stop_gradient(
                self.priority_handler.update_priority(
                    keys=sample_key,
                    latent_states=tf.stop_gradient(tf.cast(tf.round(latent_state), tf.int32)),
                    loss=tf.stop_gradient(distortion + rate)))

        # metrics
        action_sample, reward_sample, next_state_sample = reconstruction_distribution.sample()
        self.loss_metrics['ELBO'](tf.stop_gradient(-1. * (distortion + rate)))
        self.loss_metrics['action_mse'](action, action_sample)
        self.loss_metrics['reward_mse'](reward, reward_sample)
        self.loss_metrics['state_mse'](next_state, next_state_sample)
        self.loss_metrics['state_rate'](state_encoder_rate)
        # self.loss_metrics['state_encoder_entropy'](self._state_vae.binary_encode(next_state, next_label).entropy())
        self.loss_metrics['action_encoder_entropy'](
            tf.stop_gradient(self.discrete_action_encoding(latent_state, action).entropy()))
        #  self.loss_metrics['state_decoder_variance'](state_distribution.variance())
        self.loss_metrics['action_rate'](action_encoder_rate)
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        self.loss_metrics['annealed_rate'](tf.stop_gradient(self.kl_scale_factor * rate))
        self.loss_metrics['entropy_regularizer'](
            tf.stop_gradient(self.entropy_regularizer_scale_factor * entropy_regularizer))
        self.loss_metrics['t_1_state'].reset_states()
        self.loss_metrics['t_1_state'](self._state_vae.encoder_temperature)
        self.loss_metrics['t_2_state'].reset_states()
        self.loss_metrics['t_2_state'](self._state_vae.prior_temperature)
        self.loss_metrics['transition_log_probs'](
            tf.stop_gradient(
                self.discrete_latent_transition_probability_distribution(
                    tf.stop_gradient(tf.round(latent_state)),
                    tf.stop_gradient(tf.math.log(tf.round(tf.exp(log_latent_action)) + epsilon)),
                    log_latent_action=True
                ).log_prob(next_label, tf.stop_gradient(tf.round(tf.sigmoid(next_logistic_latent_state))))))

        return {'distortion': distortion, 'rate': rate, 'entropy_regularizer': entropy_regularizer}

    @tf.function
    def entropy_regularizer(
            self,
            latent_state: tf.Tensor,
            action: tf.Tensor,
            state: Optional[tf.Tensor] = None,
            enforce_deterministic_action_encoder: bool = False,
            use_marginal_entropy: bool = False
    ):
        if state is not None:
            state_regularizer = super().entropy_regularizer(
                state=state,
                use_marginal_entropy=use_marginal_entropy,
                latent_states=latent_state)
        else:
            state_regularizer = 0.
        if self.entropy_regularizer_scale_factor < 0. and not enforce_deterministic_action_encoder:
            action_regularizer = 0.
        else:
            action_regularizer = -1. * self._action_regularizer_scaling * tf.reduce_mean(
                self.discrete_action_encoding(latent_state, action).entropy(), axis=0)

        return state_regularizer + action_regularizer

    def eval(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor
    ):
        latent_distribution = self.binary_encode(state)
        next_latent_distribution = self.binary_encode(next_state)
        latent_state = tf.concat([label, tf.cast(latent_distribution.sample(), tf.float32)], axis=-1)
        next_latent_state_no_label = tf.cast(next_latent_distribution.sample(), tf.float32)

        latent_action_encoder = self.discrete_action_encoding(latent_state, action)
        latent_policy = self.discrete_latent_policy(latent_state)
        latent_action = tf.cast(latent_action_encoder.sample(), tf.float32)
        try:
            rate = latent_action_encoder.kl_divergence(latent_policy)
        except tf.errors.InvalidArgumentError:
            rate = latent_action_encoder.log_prob(latent_action) - latent_policy.log_prob(latent_action)

        # transition probability reconstruction
        transition_distribution = self.discrete_latent_transition_probability_distribution(
            latent_state, tf.math.log(latent_action + epsilon), log_latent_action=True, relaxed_state_encoding=False)
        log_q_encoding_latent_state = next_latent_distribution.log_prob(next_latent_state_no_label)
        log_p_transition = transition_distribution.log_prob(next_label, next_latent_state_no_label)
        rate += log_q_encoding_latent_state - log_p_transition

        next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        reconstruction_distribution = tfd.JointDistributionSequential([
            self.decode_action(latent_state, tf.math.log(latent_action + epsilon), log_latent_action=True),
            self.reward_probability_distribution(
                latent_state, tf.math.log(latent_action + epsilon), next_latent_state, log_latent_action=True),
            self.decode(next_latent_state)
        ])

        distortion = -1. * reconstruction_distribution.log_prob(action, reward, next_state)

        return {
            'distortion': distortion,
            'rate': rate,
            'elbo': -1. * (distortion + rate),
            'latent_states': tf.concat([tf.cast(latent_state, tf.int64), tf.cast(next_latent_state, tf.int64)], axis=0),
            'latent_actions': tf.cast(tf.argmax(latent_action, axis=1), tf.int64)
        }

    def mean_latent_bits_used(self, inputs, eps=1e-3, deterministic=True):
        state, label, action, reward, next_state, next_label = inputs[:6]
        latent_state = tf.cast(self.binary_encode(state, label).sample(), tf.float32)
        mean = tf.reduce_mean(self.discrete_action_encoding(latent_state, action).probs_parameter(), axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

        mbu = {'mean_action_bits_used': mean_bits_used}
        mbu.update(self._state_vae.mean_latent_bits_used(inputs, eps, deterministic))
        return mbu

    def get_state_vae(self) -> VariationalMarkovDecisionProcess:
        return self._state_vae

    def wrap_tf_environment(
            self,
            tf_env: tf_environment.TFEnvironment,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            deterministic_embedding_functions: bool = True
    ) -> tf_environment.TFEnvironment:

        class VariationalTFEnvironmentDiscretizer(tf_environment.TFEnvironment):

            def __init__(
                    self,
                    variational_action_discretizer: VariationalActionDiscretizer,
                    tf_env: tf_environment.TFEnvironment,
                    labeling_function: Callable[[tf.Tensor], tf.Tensor],
                    deterministic_embedding_functions: bool = True
            ):
                action_spec = specs.BoundedTensorSpec(
                    shape=(),
                    dtype=tf.int32,
                    minimum=0,
                    maximum=variational_action_discretizer.number_of_discrete_actions - 1,
                    name='action'
                )
                observation_spec = specs.BoundedTensorSpec(
                    shape=(variational_action_discretizer.latent_state_size,),
                    dtype=tf.int32,
                    minimum=0,
                    maximum=1,
                    name='observation'
                )
                time_step_spec = ts.time_step_spec(observation_spec)
                super(VariationalTFEnvironmentDiscretizer, self).__init__(
                    time_step_spec=time_step_spec,
                    action_spec=action_spec,
                    batch_size=tf_env.batch_size
                )

                self.embed_observation = variational_action_discretizer.binary_encode
                self.embed_latent_action = (
                    lambda latent_state, latent_action: variational_action_discretizer.decode_action(
                        latent_state, latent_action, disable_mixture_distribution=True))
                self.tf_env = tf_env
                self._labeling_function = labeling_function
                self.observation_shape, self.action_shape, self.reward_shape = [
                    variational_action_discretizer.state_shape,
                    variational_action_discretizer.action_shape,
                    variational_action_discretizer.reward_shape
                ]
                self._current_latent_state = None
                if deterministic_embedding_functions:
                    self._get_embedding = lambda distribution: distribution.mode()
                else:
                    self._get_embedding = lambda distribution: distribution.sample()
                self.deterministic_embedding_functions = deterministic_embedding_functions
                self.labeling_function = dataset_generator.ergodic_batched_labeling_function(labeling_function)

            def _current_time_step(self):
                if self._current_latent_state is None:
                    return self.reset()
                time_step = self.tf_env.current_time_step()
                return trajectories.time_step.TimeStep(
                    time_step.step_type, time_step.reward, time_step.discount, self._current_latent_state)

            def _step(self, action):
                real_action = self._get_embedding(
                    self.embed_latent_action(
                        tf.cast(self._current_latent_state, tf.float32),
                        tf.one_hot(indices=action, depth=self.action_spec().maximum + 1, axis=-1, dtype=tf.float32)))

                time_step = self.tf_env.step(real_action)
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

    @property
    def inference_variables(self):
        variables = []
        for network in [self.action_encoder, self.encoder_network]:
            variables += network.trainable_variables
        return variables

    @property
    def generator_variables(self):
        variables = []
        for network in [self.reconstruction_network, self.action_transition_network, self.action_reward_network]:
            variables += network.trainable_variables
        return variables

    @property
    def action_discretizer_variables(self):
        variables = []
        for network in [
            self.action_encoder,
            self.action_transition_network,
            self.latent_policy_network,
            self.action_reward_network,
            self.action_decoder_network
        ]:
            variables += network.trainable_variables
        return variables

    @tf.function
    def compute_apply_gradients(
            self,
            state: tf.Tensor,
            label: tf.Tensor,
            action: tf.Tensor,
            reward: tf.Tensor,
            next_state: tf.Tensor,
            next_label: tf.Tensor,
            sample_key: Optional[tf.Tensor] = None,
            sample_probability: Optional[tf.Tensor] = None,
    ):
        if self.full_optimization:
            return self._compute_apply_gradients(
                state, label, action, reward, next_state, next_label,
                self.trainable_variables, sample_key, sample_probability)
        else:
            return self._compute_apply_gradients(
                state, label, action, reward, next_state, next_label,
                self.action_discretizer_variables, sample_key, sample_probability)

    def estimate_local_losses_from_samples(
            self,
            environment: tf_py_environment.TFPyEnvironment,
            steps: int,
            labeling_function: Callable[[tf.Tensor], tf.Tensor],
            estimate_transition_function_from_samples: bool = False,
            assert_estimated_transition_function_distribution: bool = False,
            replay_buffer_max_frames: Optional[int] = int(1e5),
            reward_scaling: Optional[float] = 1.,
    ):
        if self.latent_policy_network is None:
            raise ValueError('This VAE is not built for policy abstraction.')

        return estimate_local_losses_from_samples(
            environment=environment, latent_policy=self.get_latent_policy(),
            steps=steps,
            number_of_discrete_actions=self.number_of_discrete_actions,
            state_embedding_function=self.state_embedding_function,
            action_embedding_function=lambda state, latent_action: self.action_embedding_function(
                state, latent_action, labeling_function=latent_action),
            latent_reward_function=lambda latent_state, latent_action,
                                          next_latent_state: (
                self.reward_probability_distribution(
                    latent_state=tf.cast(latent_state, dtype=tf.float32),
                    latent_action=tf.cast(latent_action, dtype=tf.float32),
                    next_latent_state=tf.cast(next_latent_state,
                                              dtype=tf.float32),
                    disable_mixture_distribution=True).mode()),
            labeling_function=labeling_function, latent_transition_function=(
                lambda latent_state, latent_action:
                self.discrete_latent_transition_probability_distribution(
                    latent_state=tf.cast(latent_state, tf.float32),
                    latent_action=tf.math.log(latent_action + epsilon),
                    log_latent_action=True)),
            estimate_transition_function_from_samples=estimate_transition_function_from_samples,
            replay_buffer_max_frames=replay_buffer_max_frames,
            reward_scaling=reward_scaling)


def load(tf_model_path: str, full_optimization: bool = False,
         step: Optional[int] = None) -> VariationalActionDiscretizer:
    tf_model = tf.saved_model.load(tf_model_path)
    state_model = tf_model._state_vae
    state_vae = VariationalMarkovDecisionProcess(
        state_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['state'].shape)[1:],
        label_shape=(tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1] - 1,),
        action_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['action'].shape)[1:],
        reward_shape=tuple(tf_model.signatures['serving_default'].structured_input_signature[1]['reward'].shape)[1:],
        encoder_network=state_model.encoder_network,
        transition_network=state_model.transition_network,
        reward_network=state_model.reward_network,
        label_transition_network=state_model.label_transition_network,
        decoder_network=state_model.reconstruction_network,
        latent_state_size=(tf_model.encoder_network.variables[-1].shape[0] +
                           tf_model.signatures['serving_default'].structured_input_signature[1]['label'].shape[-1]),
        encoder_temperature=state_model._encoder_temperature,
        prior_temperature=state_model._prior_temperature,
        mixture_components=tf.shape(state_model.reconstruction_network.variables[-1])[-1],
        evaluation_window_size=tf.shape(tf_model.evaluation_window)[0],
        pre_loaded_model=True)
    model = VariationalActionDiscretizer(
        vae_mdp=state_vae,
        number_of_discrete_actions=tf_model.action_encoder.variables[-1].shape[0],
        action_encoder_network=tf_model.action_encoder,
        action_decoder_network=tf_model.action_decoder_network,
        transition_network=tf_model.action_transition_network,
        reward_network=tf_model.action_reward_network,
        action_label_transition_network=tf_model.action_label_transition_network,
        latent_policy_network=tf_model.latent_policy_network,
        one_output_per_action=tf_model.action_decoder_network.variables[0].shape[0] == state_vae.latent_state_size,
        encoder_temperature=tf_model._encoder_temperature,
        prior_temperature=tf_model._prior_temperature,
        reconstruction_mixture_components=tf_model.action_decoder_network.variables[-1].shape[0],
        pre_loaded_model=True,
        full_optimization=full_optimization)

    if step is not None:
        path_list = tf_model_path.split(os.sep)
        path_list[path_list.index('models')] = 'training_checkpoints'
        while not os.path.isdir(os.path.join(*path_list)) and len(path_list) > 0:
            path_list.pop()
        if not path_list:
            raise FileNotFoundError('No training checkpoint found for model', model)
        else:
            path_list.append('ckpt-{:d}-1'.format(step))
            checkpoint_path = os.path.join(*path_list)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_path)

    return model
