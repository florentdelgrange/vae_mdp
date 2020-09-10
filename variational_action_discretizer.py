from typing import Tuple, Optional, List, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Concatenate, Reshape, Dense, Lambda
from tf_agents import trajectories
from tf_agents.networks import network
from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.policies import tf_policy, actor_policy
from tf_agents.environments import tf_environment, py_environment
from tf_agents.trajectories import time_step as ts

import variational_mdp
from variational_mdp import VariationalMarkovDecisionProcess
from variational_mdp import epsilon

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
            branching_action_networks: bool = False,
            pre_processing_network: Model = Sequential(
                [Dense(units=256, activation=tf.nn.leaky_relu),
                 Dense(units=256, activation=tf.nn.leaky_relu)],
                name='pre_processing_network'),
            encoder_temperature: Optional[float] = None,
            prior_temperature: Optional[float] = None,
            encoder_temperature_decay_rate: float = 0.,
            prior_temperature_decay_rate: float = 0.,
            pre_loaded_model: bool = False,
            one_output_per_action: bool = False,
            relaxed_state_encoding: bool = False,
            full_optimization: bool = False,
            reconstruction_mixture_components: int = 1,
    ):

        super().__init__(vae_mdp.state_shape, vae_mdp.action_shape, vae_mdp.reward_shape, vae_mdp.label_shape,
                         vae_mdp.encoder_network, vae_mdp.transition_network, vae_mdp.reward_network,
                         vae_mdp.reconstruction_network, vae_mdp.latent_state_size,
                         vae_mdp.encoder_temperature.numpy(), vae_mdp.prior_temperature.numpy(),
                         vae_mdp.encoder_temperature_decay_rate.numpy(), vae_mdp.prior_temperature_decay_rate.numpy(),
                         vae_mdp.regularizer_scale_factor.numpy(), vae_mdp.regularizer_decay_rate.numpy(),
                         vae_mdp.kl_scale_factor.numpy(), vae_mdp.kl_growth_rate.numpy(), vae_mdp.mixture_components,
                         vae_mdp.scale_activation, vae_mdp.full_covariance, pre_loaded_model=True)

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
            latent_action = Input(shape=(number_of_discrete_actions,)) if not one_output_per_action else None

            action_encoder = Concatenate(name="action_encoder_input")([latent_state, action])
            action_encoder = action_encoder_network(action_encoder)
            action_encoder = Dense(
                units=number_of_discrete_actions,
                activation=None,
                name='action_encoder_exp_one_hot_logits'
            )(action_encoder)
            self.action_encoder = Model(
                inputs=[latent_state, action],
                outputs=action_encoder,
                name="action_encoder")

            # prior over actions
            self.action_prior_logits = tf.compat.v1.get_variable(
                shape=(number_of_discrete_actions,), name='action_prior_exp_one_hot_logits')

            # discrete actions transition network
            if not one_output_per_action:
                _transition_network = Concatenate()([latent_state, latent_action])
                _transition_network = transition_network(_transition_network)
                _transition_network = Dense(
                    units=self.latent_state_size,
                    activation=None,
                    name='discrete_action_transition_next_state_logits'
                )(_transition_network)
                self.action_transition_network = Model(
                    inputs=[latent_state, latent_action],
                    outputs=_transition_network,
                    name="action_transition_network"
                )
            else:
                transition_network_pre_processing = clone_model(pre_processing_network, 'transition')(latent_state)
                transition_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        _transition_network = clone_model(transition_network, str(action))
                    else:
                        _transition_network = transition_network
                    _transition_network = _transition_network(transition_network_pre_processing)
                    _transition_network = Dense(
                        units=self.latent_state_size,
                        activation=None,
                        name='action{}_transition_next_state_logits'.format(action)
                    )(_transition_network)
                    transition_outputs.append(_transition_network)
                self.action_transition_network = Model(
                    inputs=latent_state,
                    outputs=transition_outputs,
                    name="action_transition_network")

            # discrete actions reward network
            if not one_output_per_action:
                _reward_network = Concatenate()([latent_state, latent_action, next_latent_state])
                _reward_network = reward_network(_reward_network)
                reward_mean = Dense(
                    units=np.prod(self.reward_shape),
                    activation=None,
                    name='action_reward_mean_0'
                )(_reward_network)
                reward_mean = Reshape(self.reward_shape, name='action_reward_mean')(reward_mean)
                reward_raw_covar = Dense(
                    units=np.prod(self.reward_shape),
                    activation=None,
                    name='action_reward_raw_diag_covariance_0'
                )(_reward_network)
                reward_raw_covar = Reshape(
                    self.reward_shape,
                    name='action_reward_raw_diag_covariance'
                )(reward_raw_covar)
                self.action_reward_network = Model(
                    inputs=[latent_state, latent_action, next_latent_state],
                    outputs=[reward_mean, reward_raw_covar],
                    name="discrete_actions_reward_network"
                )
            else:
                reward_network_input = Concatenate()([latent_state, next_latent_state])
                reward_network_pre_processing = clone_model(pre_processing_network, 'reward')(reward_network_input)
                reward_network_outputs = []
                for action in range(number_of_discrete_actions):
                    if branching_action_networks:
                        _reward_network = clone_model(reward_network, str(action))
                    else:
                        _reward_network = reward_network
                    _reward_network = reward_network(reward_network_pre_processing)
                    reward_mean = Dense(
                        units=np.prod(self.reward_shape),
                        activation=None,
                        name='action{}_reward_mean_0'.format(action)
                    )(_reward_network)
                    reward_mean = Reshape(self.reward_shape, name='action{}_reward_mean'.format(action))(reward_mean)
                    reward_raw_covar = Dense(
                        units=np.prod(self.reward_shape),
                        activation=None,
                        name='action{}_reward_raw_diag_covariance_0'.format(action)
                    )(_reward_network)
                    reward_raw_covar = Reshape(
                        self.reward_shape,
                        name='action{}_reward_raw_diag_covariance'.format(action)
                    )(reward_raw_covar)
                    reward_network_outputs.append([reward_mean, reward_raw_covar])
                reward_network_mean = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(mean for mean, covariance in reward_network_outputs))
                reward_network_raw_covariance = Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                    list(covariance for mean, covariance in reward_network_outputs))
                self.action_reward_network = Model(
                    inputs=[latent_state, next_latent_state],
                    outputs=[reward_network_mean, reward_network_raw_covariance],
                    name="discrete_actions_reward_network"
                )

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
                    activation=None
                )(action_decoder)
                action_decoder_mean = Reshape(
                    target_shape=action_shape,
                    name='action_decoder_mean'
                )(action_decoder_mean)
                action_decoder_raw_covariance = Dense(
                    units=self.mixture_components * np.prod(self.action_shape),
                    activation=None
                )(action_decoder)
                action_decoder_raw_covariance = Reshape(
                    target_shape=action_shape,
                    name='action_decoder_raw_diag_covariance'
                )(action_decoder_raw_covariance)
                action_decoder_mixture_categorical_logits = Dense(
                    units=self.mixture_components,
                    activation=None,
                    name='action_decoder_mixture_categorical_logits'
                )(action_decoder)
                self.action_decoder = Model(
                    inputs=[latent_state, latent_action],
                    outputs=[
                        action_decoder_mean,
                        action_decoder_raw_covariance,
                        action_decoder_mixture_categorical_logits
                    ],
                    name="action_decoder_network")
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
                        name='action{}_decoder_mean'.format(action)
                    )(action_decoder_mean)
                    action_decoder_raw_covariance = Dense(
                        units=self.mixture_components * np.prod(self.action_shape),
                        activation=None
                    )(action_decoder)
                    action_decoder_raw_covariance = Reshape(
                        target_shape=action_shape,
                        name='action{}_decoder_raw_diag_covariance'.format(action)
                    )(action_decoder_raw_covariance)
                    action_decoder_mixture_categorical_logits = Dense(
                        units=self.mixture_components,
                        activation=None,
                        name='action{}_decoder_mixture_categorical_logits'.format(action)
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
                self.action_decoder = Model(
                    inputs=latent_state,
                    outputs=[
                        action_decoder_mean,
                        action_decoder_raw_covariance,
                        action_decoder_mixture_categorical_logits
                    ],
                    name="action_decoder_network")

        else:
            self.action_encoder = action_encoder_network
            self.action_transition_network = transition_network
            self.action_reward_network = reward_network
            self.action_decoder = action_decoder_network

        try:
            state_layers = (self.encoder_network.layers,
                            self.transition_network.layers,
                            self.reward_network.layers,
                            self.reconstruction_network.layers)
        except AttributeError:  # tensorflow backward compatibility
            state_layers = (self.encoder_network.keras_api.layers,
                            self.transition_network.keras_api.layers,
                            self.reward_network.keras_api.layers,
                            self.reconstruction_network.keras_api.layers)

        if not self.full_optimization:
            # freeze all latent states related layers
            for layers in state_layers:
                for layer in layers:
                    layer.trainable = False
                    assert not layer.trainable

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'action_mse': tf.keras.metrics.MeanSquaredError(name='action_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'cross_entropy_regularizer': tf.keras.metrics.Mean(name='cross_entropy_regularizer'),
            'decoder_divergence': tf.keras.metrics.Mean(name='cross_entropy_regularizer'),
        }
        if self.full_optimization:
            self.loss_metrics.update({
                'state_mse': tf.keras.metrics.MeanSquaredError(name='state_mse'),
                'state_reward_mse': tf.keras.metrics.MeanSquaredError(name='state_reward_mse'),
                'state_distortion': tf.keras.metrics.Mean(name='state_distortion'),
                'state_rate': tf.keras.metrics.Mean(name='state_rate'),
                'action_distortion': tf.keras.metrics.Mean(name='action_distortion'),
                'action_rate': tf.keras.metrics.Mean(name='action_rate')
            })

    def relaxed_action_encoding(
            self, latent_state: tf.Tensor, action: tf.Tensor, temperature: float
    ) -> tfd.Distribution:
        encoder_logits = self.action_encoder([latent_state, action])
        return tfd.ExpRelaxedOneHotCategorical(temperature=temperature, logits=encoder_logits)

    def discrete_action_encoding(self, latent_state: tf.Tensor, action: tf.Tensor) -> tfd.Distribution:
        relaxed_distribution = self.relaxed_action_encoding(latent_state, action, 1e-5)
        log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
        return tfd.OneHotCategorical(logits=log_probs)

    def relaxed_action_prior(self, temperature: float):
        return tfd.ExpRelaxedOneHotCategorical(
            temperature=temperature, logits=self.action_prior_logits)

    def discrete_action_prior(self):
        relaxed_distribution = self.relaxed_action_prior(temperature=1e-5)
        log_probs = tf.math.log(relaxed_distribution.probs_parameter() + epsilon)
        return tfd.OneHotCategorical(logits=log_probs)

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor,
            relaxed_state_encoding: bool = False, log_latent_action: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            next_state_logits = self.action_transition_network([latent_state, latent_action])

            if relaxed_state_encoding:
                return tfd.Logistic(
                    loc=next_state_logits / self._state_vae.prior_temperature,
                    scale=1. / self._state_vae.prior_temperature
                )
            else:
                return tfd.Bernoulli(logits=next_state_logits)
        else:
            transition_output = self.action_transition_network(latent_state)
            latent_action = tf.stack([latent_action for _ in range(self.latent_state_size)], axis=1)

            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action)
            else:
                action_categorical = tfd.Categorical(probs=latent_action)

            if relaxed_state_encoding:
                components = [tfd.Logistic(
                    loc=logits / self._state_vae.prior_temperature,
                    scale=1. / self._state_vae.prior_temperature)
                    for logits in transition_output]
            else:
                components = [tfd.Bernoulli(logits=logits) for logits in transition_output]

            return tfd.Mixture(cat=action_categorical, components=components)

    def reward_probability_distribution(
            self, latent_state, latent_action, next_latent_state, log_latent_action: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            [reward_mean, reward_raw_covariance] = self.action_reward_network(
                [latent_state, latent_action, next_latent_state])

            return tfd.MultivariateNormalDiag(loc=reward_mean, scale_diag=self.scale_activation(reward_raw_covariance))
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action)
            else:
                action_categorical = tfd.Categorical(probs=latent_action)

            [reward_mean, reward_raw_covariance] = self.action_reward_network([latent_state, next_latent_state])

            return tfd.MixtureSameFamily(
                mixture_distribution=action_categorical,
                components_distribution=tfd.MultivariateNormalDiag(
                    loc=reward_mean,
                    scale_diag=self.scale_activation(reward_raw_covariance)),
            )

    def decode_action(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor, log_latent_action: bool = False
    ) -> tfd.Distribution:

        if not self.one_output_per_action:
            if log_latent_action:
                latent_action = tf.exp(latent_action)

            [action_mean, action_raw_covariance, cat_logits] = self.action_decoder([latent_state, latent_action])
            if self.mixture_components == 1:
                return tfd.MultivariateNormalDiag(
                    loc=action_mean,
                    scale_diag=self.scale_activation(action_raw_covariance)
                )
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=cat_logits),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=action_mean,
                        scale_diag=self.scale_activation(action_raw_covariance)
                    )
                )
        else:
            if log_latent_action:
                action_categorical = tfd.Categorical(logits=latent_action)
            else:
                action_categorical = tfd.Categorical(probs=latent_action)

            [action_mean, action_raw_covariance, cat_logits] = self.action_decoder(latent_state)

            if self.mixture_components == 1:
                return tfd.MixtureSameFamily(
                    mixture_distribution=action_categorical,
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=action_mean,
                        scale_diag=self.scale_activation(action_raw_covariance),
                    )
                )
            else:
                return tfd.MixtureSameFamily(
                    mixture_distribution=action_categorical,
                    components_distribution=tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(logits=cat_logits),
                        components_distribution=tfd.MultivariateNormalDiag(
                            loc=action_mean,
                            scale_diag=self.scale_activation(action_raw_covariance)
                        )
                    )
                )

    def call(self, inputs, training=None, mask=None, **kwargs):
        # inputs are assumed to have shape
        # [(?, 2, state_shape), (?, 2, action_shape), (?, 2, reward_shape), (?, 2, state_shape), (?, 2, label_shape)]
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        if self.relaxed_state_encoding:
            z = self._state_vae.relaxed_encoding(
                s_0, a_0, r_0, s_1, l_1, self._state_vae.encoder_temperature).sample()
            z = tf.sigmoid(z)
            z_prime = self._state_vae.relaxed_encoding(
                s_1, a_1, r_1, s_2, l_2, self._state_vae.encoder_temperature).sample()
        else:
            z = tf.cast(self.binary_encode(s_0, a_0, r_0, s_1, l_1).sample(), tf.float32)
            z_prime = tf.cast(self.binary_encode(s_1, a_1, r_1, s_2, l_2).sample(), tf.float32)
        q = self.relaxed_action_encoding(z, a_1, self.encoder_temperature)
        p = self.relaxed_action_prior(self.prior_temperature)
        latent_action = q.sample()

        log_q_latent_action = q.log_prob(latent_action)
        log_p_latent_action = p.log_prob(latent_action)

        # transition probability reconstruction
        transition_probability_distribution = \
            self.discrete_latent_transition_probability_distribution(
                z, latent_action, relaxed_state_encoding=self.relaxed_state_encoding, log_latent_action=True)
        if self.relaxed_state_encoding:
            continuous_action_transition = self._state_vae.relaxed_latent_transition_probability_distribution(
                z, a_1, self._state_vae.prior_temperature)
        else:
            continuous_action_transition = self._state_vae.discrete_latent_transition_probability_distribution(z, a_1)
        log_p_transition_action = continuous_action_transition.log_prob(z_prime)
        log_p_transition_latent_action = transition_probability_distribution.log_prob(z_prime)
        log_p_transition = tf.reduce_sum(log_p_transition_latent_action - log_p_transition_action, axis=1)

        if self.relaxed_state_encoding:
            z_prime = tf.sigmoid(z_prime)

        # rewards reconstruction
        reward_distribution = self.reward_probability_distribution(z, latent_action, z_prime, log_latent_action=True)
        log_p_rewards_action = self._state_vae.reward_probability_distribution(
            z, a_1, z_prime).log_prob(r_1)
        log_p_rewards_latent_action = reward_distribution.log_prob(r_1)
        log_p_rewards = log_p_rewards_latent_action - log_p_rewards_action

        # action reconstruction
        action_distribution = self.decode_action(z, latent_action, log_latent_action=True)
        log_p_action = action_distribution.log_prob(a_1)

        rate = log_q_latent_action - log_p_latent_action
        distortion = -1. * (log_p_action + log_p_rewards + log_p_transition)

        def compute_encoder_uniform_cross_entropy():
            discrete_action_posterior = self.discrete_action_encoding(z, a_1)
            prior_uniform_distribution = tfd.OneHotCategorical(
                logits=tf.math.log(
                    [1. / self.number_of_discrete_actions for _ in range(self.number_of_discrete_actions)])
            )
            return prior_uniform_distribution.kl_divergence(discrete_action_posterior)

        cross_entropy_regularizer = compute_encoder_uniform_cross_entropy()

        if self.full_optimization:
            state_distortion, state_rate, state_cer = self._state_vae(inputs, metrics=False)
            distortion += state_distortion
            rate += state_rate
            cross_entropy_regularizer += state_cer

        # metrics
        def compute_decoder_jensen_shannon_divergence():
            [action_mean, action_raw_covariance, categorical_logits] = self.action_decoder(z)
            action_means = tf.unstack(action_mean, axis=1)
            action_raw_covariances = tf.unstack(action_raw_covariance, axis=1)
            cat_logits = tf.unstack(categorical_logits, axis=1)
            posterior_distributions = [
                tfd.MultivariateNormalDiag(loc=mean, scale_diag=self.scale_activation(raw_covariance))
                for mean, raw_covariance in zip(action_means, action_raw_covariances)
            ] if self.mixture_components == 1 else [
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(logits=logits),
                    components_distribution=tfd.MultivariateNormalDiag(
                        loc=mean,
                        scale_diag=self.scale_activation(raw_covariance)
                    ),
                ) for logits, mean, raw_covariance in
                zip(cat_logits, action_means, action_raw_covariances)
            ]
            weighted_distribution = tfd.Mixture(
                cat=tfd.Categorical(
                    logits=tf.ones(shape=(tf.shape(categorical_logits)[0], self.number_of_discrete_actions))),
                components=posterior_distributions
            )
            weighted_distribution_entropy = -1. * weighted_distribution.prob(a_1) * weighted_distribution.log_prob(a_1)

            if self.mixture_components == 1:
                weighted_entropy = tf.reduce_sum(
                    [
                        1. / self.number_of_discrete_actions * posterior_distributions[action].entropy()
                        for action in range(self.number_of_discrete_actions)
                    ],
                )
            else:
                weighted_entropy = tf.reduce_sum(
                    [
                        - 1. * 1. / self.number_of_discrete_actions *
                        posterior_distributions[action].prob(a_1) * posterior_distributions[action].log_prob(a_1)
                        for action in range(self.number_of_discrete_actions)
                    ], axis=0
                )

            return weighted_distribution_entropy - weighted_entropy

        self.loss_metrics['ELBO'](-1. * (distortion + rate))
        self.loss_metrics['action_mse'](a_1, action_distribution.sample())
        self.loss_metrics['reward_mse'](r_1, reward_distribution.sample())
        if self.full_optimization:
            self.loss_metrics['state_mse'](s_2, self._state_vae.decode(z_prime).sample())
            self.loss_metrics['state_reward_mse'](
                r_1, self._state_vae.reward_probability_distribution(z, a_1, z_prime).sample())
            self.loss_metrics['state_distortion'](state_distortion)
            self.loss_metrics['state_rate'](state_rate)
            self.loss_metrics['action_distortion'](distortion - state_distortion)
            self.loss_metrics['action_rate'](rate - state_rate)
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        self.loss_metrics['annealed_rate'](self.kl_scale_factor * rate)
        self.loss_metrics['cross_entropy_regularizer'](cross_entropy_regularizer)
        if self.one_output_per_action:
            self.loss_metrics['decoder_divergence'](compute_decoder_jensen_shannon_divergence())

        if variational_mdp.debug:
            tf.print(z, "sampled z", summarize=variational_mdp.debug_verbosity)
            tf.print(z_prime, "sampled z'", summarize=variational_mdp.debug_verbosity)
            tf.print(q.logits, "logits of Q_action", summarize=variational_mdp.debug_verbosity)
            tf.print(p.logits, "logits of P_action", summarize=variational_mdp.debug_verbosity)
            tf.print(latent_action, "sampled log action from Q", summarize=variational_mdp.debug_verbosity)
            tf.print(log_q_latent_action, "log Q(exp_action)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_latent_action, "log P(exp_action)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_rewards, "log P(r | z, â, z')", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_transition, "log P(z' | z, â)", summarize=variational_mdp.debug_verbosity)
            tf.print(log_p_action, "log P(a | z, â)", summarize=variational_mdp.debug_verbosity)

        return [distortion, rate, cross_entropy_regularizer]

    def eval(self, inputs):
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        z = tf.cast(self.binary_encode(s_0, a_0, r_0, s_1, l_1).sample(), tf.float32)
        z_prime = tf.cast(self.binary_encode(s_1, a_1, r_1, s_2, l_2).sample(), tf.float32)
        q = self.discrete_action_encoding(z, a_1)
        p = self.discrete_action_prior()
        latent_action = tf.cast(q.sample(), tf.float32)

        # transition probability reconstruction
        log_p_transition_action = \
            self._state_vae.discrete_latent_transition_probability_distribution(z, a_1).log_prob(z_prime)
        log_p_transition_latent_action = self.discrete_latent_transition_probability_distribution(
            z, tf.math.log(latent_action + epsilon), log_latent_action=True).log_prob(z_prime)
        log_p_transition = tf.reduce_sum(log_p_transition_latent_action - log_p_transition_action, axis=1)

        # rewards reconstruction
        log_p_rewards_action = self._state_vae.reward_probability_distribution(
            z, a_1, z_prime).log_prob(r_1)
        log_p_rewards_latent_action = self.reward_probability_distribution(
            z, tf.math.log(latent_action + epsilon), z_prime, log_latent_action=True).log_prob(r_1)
        log_p_rewards = log_p_rewards_latent_action - log_p_rewards_action

        # action reconstruction
        action_distribution = self.decode_action(z, tf.math.log(latent_action + epsilon), log_latent_action=True)
        log_p_action = action_distribution.log_prob(a_1)

        rate = q.kl_divergence(p)
        distortion = -1. * (log_p_action + log_p_rewards + log_p_transition)

        return -1. * (distortion + rate)

    def mean_latent_bits_used(self, inputs, eps=1e-3):
        """
        Compute the mean number of bits used in the latent space of the vae_mdp for the given dataset batch.
        This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
        """
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)
        z = tf.cast(self.binary_encode(s_0, a_0, r_0, s_1, l_1).sample(), tf.float32)
        mean = tf.reduce_mean(self.discrete_action_encoding(z, a_1).probs_parameter(), axis=0)
        check = lambda x: 1 if 1 - eps > x > eps else 0
        mean_bits_used = tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

        mbu = {'mean_action_bits_used': mean_bits_used}
        mbu.update(self._state_vae.mean_latent_bits_used(inputs, eps))
        return mbu

    def generate_random_policy(
            self, tf_env: tf_environment.TFEnvironment, labeling_function: Callable[[tf.Tensor], tf.Tensor]
    ) -> tf_policy.Base:
        """
        Generates a uniform random policy from discrete states of this VAE-MDP to continuous actions of the input
        environment.
        """

        time_step_spec = tf_env.time_step_spec()
        # continuous actions
        action_spec = tf_env.action_spec()

        # uniform random discrete action
        uniform_action_distribution = tfd.OneHotCategorical(
            logits=tf.math.log(
                1. / self.number_of_discrete_actions * tf.ones(shape=(self.number_of_discrete_actions,))
            ),
            dtype=tf.float32
        )

        class RandomDiscreteActorNetwork(network.Network):

            def __init__(self, vae_mdp: VariationalActionDiscretizer):
                super().__init__(
                    time_step_spec,
                    state_spec=(tf_env.time_step_spec().observation, tf_env.action_spec()),
                    name='RandomDiscreteActorNetwork')
                self._vae_mdp = vae_mdp

            def call(self, observations, step_type, network_state):

                tf.print(observations, step_type, network_state)

                if step_type == ts.StepType.FIRST:
                    initial_state = tf.zeros(shape=self._vae_mdp.state_shape, dtype=tf.float32)
                    initial_action = tf.zeros(shape=self._vae_mdp.action_shape, dtype=tf.float32)
                    initial_reward = tf.zeros(shape=self._vae_mdp.reward_shape, dtype=tf.float32)
                    state, action, reward = [
                        tf.stack([initial_state for _ in range(tf.shape(observations.observation)[0])]),
                        tf.stack([initial_action for _ in range(tf.shape(observations.observation)[0])]),
                        tf.stack([initial_reward for _ in range(tf.shape(observations.observation)[0])]),
                    ]
                else:
                    state, action = network_state
                    reward = tf.reshape(observations.reward, self._vae_mdp.reward_shape)

                next_state = observations.observation
                next_label = tf.reshape(labeling_function(next_state), self._vae_mdp.label_shape)

                z = self._vae_mdp.binary_encode(state, action, reward, next_state, next_label).sample()
                z = tf.cast(z, dtype=tf.float32)
                if tf.shape(z).get_shape() == 1:
                    discrete_action = uniform_action_distribution.sample()
                else:
                    discrete_action = uniform_action_distribution.sample(
                        sample_shape=(tf.shape(z)[0])
                    )
                action = self._vae_mdp.decode_action(z, discrete_action).sample()

                return action, (next_state, action)

        return actor_policy.ActorPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=RandomDiscreteActorNetwork(vae_mdp=self)
        )

    def get_state_vae(self) -> VariationalMarkovDecisionProcess:
        return self._state_vae


def load(tf_model_path: str) -> VariationalActionDiscretizer:
    model = tf.saved_model.load(tf_model_path)
    state_model = model._state_vae
    state_vae = VariationalMarkovDecisionProcess(
        state_shape=tuple(model.signatures['serving_default'].structured_input_signature[1]['input_1'].shape)[2:],
        action_shape=tuple(model.signatures['serving_default'].structured_input_signature[1]['input_2'].shape)[2:],
        reward_shape=tuple(model.signatures['serving_default'].structured_input_signature[1]['input_3'].shape)[2:],
        label_shape=tuple(model.signatures['serving_default'].structured_input_signature[1]['input_5'].shape)[2:],
        encoder_network=state_model.encoder_network,
        transition_network=state_model.transition_network,
        reward_network=state_model.reward_network,
        decoder_network=state_model.reconstruction_network,
        latent_state_size=state_model.transition_network.variables[-1].shape[0],
        encoder_temperature=state_model._encoder_temperature,
        prior_temperature=state_model._prior_temperature,
        pre_loaded_model=True)
    return VariationalActionDiscretizer(
        vae_mdp=state_vae,
        number_of_discrete_actions=model.action_encoder.variables[-1].shape[0],
        action_encoder_network=model.action_encoder,
        action_decoder_network=model.action_decoder,
        transition_network=model.action_transition_network,
        reward_network=model.action_reward_network,
        one_output_per_action=model.action_decoder.variables[0].shape[0] == state_vae.latent_state_size,
        encoder_temperature=model._encoder_temperature,
        prior_temperature=model._prior_temperature,
        reconstruction_mixture_components=2,  # mixture components > 1 is sufficient
        pre_loaded_model=True
    )


class VariationalPyEnvironmentDiscretizer(py_environment.PyEnvironment):

    def __init__(
            self,
            variational_mdp: VariationalMarkovDecisionProcess,
            variational_action_discretizer: VariationalActionDiscretizer,
            py_env: py_environment.PyEnvironment,
            labeling_function: Callable[[tf.Tensor], tf.Tensor]
    ):

        super(VariationalPyEnvironmentDiscretizer, self).__init__()

        self.encode_observation = variational_mdp.binary_encode
        self.decode_action = variational_action_discretizer.decode_action
        self.py_env = py_env
        self.labeling_function = labeling_function
        self._array_spec = array_spec.BoundedArraySpec(
            shape=(variational_action_discretizer.number_of_discrete_actions, ),
            dtype=tf.int32,
            minimum=0,
            maximum=1,
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, variational_mdp.latent_state_size, ),
            dtype=tf.int32,
            minimum=0,
            maximum=1,
            name='observation'
        )


    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._array_spec

    def get_info(self):
        return self.py_env.get_info()

    def get_state(self):
        return self.py_env.get_state()

    def set_state(self, state):
        self.py_env.set_state(state)

    def _step(self, action):
        real_action = self.decode_action(self._current_time_step, action).sample()
        time_step = self.py_env.step(real_action)
        self._previous_reward = time_step.reward
        latent_state = self.encode_observation(
            self._current_state,
            real_action,
            time_step.reward,
            time_step.observation,
            self.labeling_function(time_step.observation)
        ).sample()
        self._previous_state = time_step.observation
        self._previous_action = real_action
        self._previous_reward = time_step.reward
        self._current_time_step = trajectories.time_step.TimeStep(
            time_step.step_type, time_step.reward, time_step.discount, latent_state
        )
        return self._current_time_step

    def _reset(self):
        initial_state = tf.zeros(shape=self._vae_mdp.state_shape, dtype=tf.float32)
        initial_action = tf.zeros(shape=self._vae_mdp.action_shape, dtype=tf.float32)
        initial_reward = tf.zeros(shape=self._vae_mdp.reward_shape, dtype=tf.float32)
        time_step = self.py_env.reset()
        latent_state = self.encode_observation(
            initial_state,
            initial_action,
            initial_reward,
            time_step.observation,
            self.labeling_function(time_step.observation)
        ).sample()
        self._current_time_step = trajectories.time_step.TimeStep(
            time_step.step_type, time_step.reward, time_step.discount, latent_state)
        self._current_state = latent_state
        return self._current_time_step

    def render(self, **kwargs):
        self.py_env.render(**kwargs)
