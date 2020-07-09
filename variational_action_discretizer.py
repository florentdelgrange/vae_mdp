import os
from typing import Tuple, Optional, List, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Concatenate, Reshape, Dense, Lambda

from variational_mdp import VariationalMarkovDecisionProcess

tfd = tfp.distributions
tfb = tfp.bijectors


class VariationalActionDiscretizer(VariationalMarkovDecisionProcess):

    def __init__(self,
                 vae_mdp: VariationalMarkovDecisionProcess,
                 number_of_discrete_actions: int,
                 action_encoder_network: Model,
                 action_decoder_network: Model,
                 transition_network: Model,
                 reward_network: Model,
                 pre_processing_network: Model = Sequential([Dense(units=256, activation=tf.nn.leaky_relu)]),
                 encoder_temperature: Optional[float] = None,
                 prior_temperature: Optional[float] = None,
                 encoder_temperature_decay_rate: float = 0.,
                 prior_temperature_decay_rate: float = 0.):

        super().__init__(vae_mdp.state_shape, vae_mdp.action_shape, vae_mdp.reward_shape, vae_mdp.label_shape,
                         vae_mdp.encoder_network, vae_mdp.transition_network,
                         vae_mdp.reward_network, vae_mdp.reconstruction_network, vae_mdp.latent_state_size,
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
        self.state_encoder_temperature = self.encoder_temperature
        self.state_prior_temperature = self.prior_temperature
        self.state_encoder_temperature_decay_rate = self.encoder_temperature_decay_rate
        self.state_prior_temperature_decay_rate = self.prior_temperature_decay_rate

        self.encoder_temperature = tf.Variable(encoder_temperature, dtype=tf.float32, trainable=False)
        self.prior_temperature = tf.Variable(prior_temperature, dtype=tf.float32, trainable=False)
        self.encoder_temperature_decay_rate = tf.constant(encoder_temperature_decay_rate, dtype=tf.float32)
        self.prior_temperature_decay_rate = tf.constant(prior_temperature_decay_rate, dtype=tf.float32)

        # action encoder network
        latent_state = Input(shape=(self.latent_state_size,))
        action = Input(shape=self.action_shape)
        next_latent_state = Input(shape=(self.latent_state_size,))
        action_encoder = Concatenate(name="action_encoder_input")([latent_state, action])
        action_encoder = action_encoder_network(action_encoder)
        action_encoder = Dense(units=number_of_discrete_actions, activation=None,
                               name='encoder_exp_one_hot_logits')(action_encoder)
        self.action_encoder = Model(inputs=[latent_state, action], outputs=action_encoder,
                                    name="action_encoder")

        # prior over actions
        self.action_prior_logits = tf.compat.v1.get_variable(
            shape=(number_of_discrete_actions,), name='prior_action_logits')

        def clone_model(model: tf.keras.Model, copy_name: str = ''):
            model = model_from_json(model.to_json(), custom_objects={'leaky_relu': tf.nn.leaky_relu})
            for layer in model.layers:
                layer._name = copy_name + '_' + layer.name
            model._name = copy_name + model.name
            return model

        # discrete actions transition network
        transition_outputs = []
        transition_network_pre_processing = clone_model(pre_processing_network, 'transition')(latent_state)
        for action in range(number_of_discrete_actions):  # branching-action network
            _transition_network = clone_model(transition_network, str(action))
            _transition_network = _transition_network(transition_network_pre_processing)
            _transition_network = Dense(units=self.latent_state_size, activation=None)(_transition_network)
            transition_outputs.append(_transition_network)
        self.discrete_actions_transition_network = Model(inputs=latent_state, outputs=transition_outputs,
                                                         name="discrete_actions_transition_network")

        # discrete actions reward network
        reward_network_input = Concatenate()([latent_state, next_latent_state])
        reward_network_pre_processing = clone_model(pre_processing_network, 'reward')(reward_network_input)
        reward_network_outputs = []
        for action in range(number_of_discrete_actions):  # branching-action network
            _reward_network = clone_model(reward_network, str(action))
            _reward_network = reward_network(reward_network_pre_processing)
            reward_mean = Dense(units=np.prod(self.reward_shape),
                                activation=None,
                                name='action{}_reward_mean_0'.format(action))(_reward_network)
            reward_mean = Reshape(self.reward_shape, name='action{}_reward_mean'.format(action))(reward_mean)
            reward_raw_covar = Dense(units=np.prod(self.reward_shape),
                                     activation=None,
                                     name='action{}_reward_raw_diag_covariance_0'.format(action))(_reward_network)
            reward_raw_covar = Reshape(self.reward_shape,
                                       name='action{}_reward_raw_diag_covariance'.format(action))(reward_raw_covar)
            reward_network_outputs.append([reward_mean, reward_raw_covar])
        reward_network_mean = \
            Lambda(lambda outputs: tf.stack(outputs, axis=1))(list(mean for mean, covariance in reward_network_outputs))
        reward_network_raw_covariance = \
            Lambda(lambda outputs: tf.stack(outputs, axis=1))(list(covariance for mean, covariance in reward_network_outputs))
        self.discrete_actions_reward_network = Model(inputs=[latent_state, next_latent_state],
                                                     outputs=[reward_network_mean, reward_network_raw_covariance],
                                                     name="discrete_actions_reward_network")

        # discrete actions decoder
        action_decoder_pre_processing = clone_model(pre_processing_network, 'action')(latent_state)
        action_decoder_outputs = []
        for action in range(number_of_discrete_actions):  # branching-action network
            action_decoder = clone_model(action_decoder_network, str(action))
            action_decoder = action_decoder(action_decoder_pre_processing)
            action_decoder_mean = Dense(units=np.prod(self.action_shape), activation=None)(action_decoder)
            action_decoder_mean = \
                Reshape(target_shape=self.action_shape,
                        name='action{}_decoder_mean'.format(action))(action_decoder_mean)
            action_decoder_raw_covariance = Dense(units=np.prod(self.action_shape), activation=None)(action_decoder)
            action_decoder_raw_covariance = \
                Reshape(target_shape=self.action_shape,
                        name='action{}_decoder_raw_diag_covariance'.format(action))(action_decoder_raw_covariance)
            action_decoder_outputs.append((action_decoder_mean, action_decoder_raw_covariance))
        action_decoder_mean = \
            Lambda(lambda outputs: tf.stack(outputs, axis=1))(list(mean for mean, covariance in action_decoder_outputs))
        action_decoder_raw_covariance = \
            Lambda(lambda outputs: tf.stack(outputs, axis=1))(
                list(covariance for mean, covariance in action_decoder_outputs))
        self.action_decoder_network = Model(inputs=latent_state,
                                            outputs=[action_decoder_mean, action_decoder_raw_covariance],
                                            name="action_decoder_network")

        state_layers = (self.encoder_network.layers,
                        self.transition_network.layers,
                        self.reward_network.layers,
                        self.reconstruction_network.layers)

        # freeze all latent states related layers
        for layers in state_layers:
            for layer in layers:
                layer.trainable = False
                assert layer.trainable == False

        self.loss_metrics = {
            'ELBO': tf.keras.metrics.Mean(name='ELBO'),
            'action_mse': tf.keras.metrics.MeanSquaredError(name='action_mse'),
            'reward_mse': tf.keras.metrics.MeanSquaredError(name='reward_mse'),
            'distortion': tf.keras.metrics.Mean(name='distortion'),
            'rate': tf.keras.metrics.Mean(name='rate'),
            'annealed_rate': tf.keras.metrics.Mean(name='annealed_rate'),
            'cross_entropy_regularizer': tf.keras.metrics.Mean(name='cross_entropy_regularizer'),
        }

    def relaxed_action_encoding(
            self, latent_state: tf.Tensor, action: tf.Tensor, temperature: float
    ) -> tfd.Distribution:
        encoder_logits = self.action_encoder([latent_state, action])
        return tfd.ExpRelaxedOneHotCategorical(temperature=temperature, logits=encoder_logits, allow_nan_stats=False)

    def discrete_action_encoding(self, latent_state: tf.Tensor, action: tf.Tensor) -> tfd.Distribution:
        relaxed_distribution = self.relaxed_action_encoding(latent_state, action, 1e-5)
        probs = relaxed_distribution.probs_parameter()
        return tfd.OneHotCategorical(probs=probs)

    def relaxed_action_prior(self, temperature: float):
        return tfd.ExpRelaxedOneHotCategorical(
            temperature=temperature, logits=self.action_prior_logits, allow_nan_stats=False)

    def discrete_action_prior(self):
        relaxed_distribution = self.relaxed_action_prior(1e-5)
        probs = relaxed_distribution.probs_parameter()
        return tfd.OneHotCategorical(probs=probs)

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor
    ) -> tfd.Distribution:
        transition_output = self.discrete_actions_transition_network(latent_state)
        probs = tf.stack([latent_action for _ in range(self.latent_state_size)], axis=1)
        cat = tfd.Categorical(probs=probs)
        components = [tfd.Bernoulli(logits=logits) for logits in transition_output]
        return tfd.Mixture(cat=cat, components=components)

    def reward_probability_distribution(
            self, latent_state, latent_action, next_latent_state
    ) -> tfd.Distribution:
        [reward_mean, reward_raw_covariance] = self.discrete_actions_reward_network([latent_state, next_latent_state])
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=latent_action),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=reward_mean,
                scale_diag=self.scale_activation(reward_raw_covariance),
                allow_nan_stats=False)
        )

    def decode_action(self, latent_state: tf.Tensor, latent_action: tf.Tensor) -> tfd.Distribution:
        [action_mean, action_raw_covariance] = self.action_decoder_network(latent_state)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=latent_action),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=action_mean,
                scale_diag=self.scale_activation(action_raw_covariance),
                allow_nan_stats=False
            )
        )

    def call(self, inputs, training=None, mask=None):
        # inputs are assumed to have shape
        # [(?, 2, state_shape), (?, 2, action_shape), (?, 2, reward_shape), (?, 2, state_shape), (?, 2, label_shape)]
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        z = tf.cast(self.binary_encode(s_0, a_0, r_0, s_1, l_1).sample(), tf.float32)
        z_prime = tf.cast(self.binary_encode(s_1, a_1, r_1, s_2, l_2).sample(), tf.float32)
        q = self.relaxed_action_encoding(z, a_1, self.encoder_temperature)
        p = self.relaxed_action_prior(self.prior_temperature)
        exp_latent_action = q.sample()
        latent_action = tf.exp(exp_latent_action)

        log_q_latent_action = q.log_prob(exp_latent_action)
        log_p_latent_action = p.log_prob(exp_latent_action)

        # rewards reconstruction
        reward_distribution = self.reward_probability_distribution(z, latent_action, z_prime)
        log_p_rewards_action = self._state_vae.reward_probability_distribution(z, a_1, z_prime).log_prob(r_1)
        log_p_rewards_latent_action = reward_distribution.log_prob(r_1)
        log_p_rewards = log_p_rewards_latent_action - log_p_rewards_action

        # transition probability reconstruction
        transition_probability_distribution = self.discrete_latent_transition_probability_distribution(z, latent_action)
        log_p_transition_action = \
            self._state_vae.discrete_latent_transition_probability_distribution(z, a_1).log_prob(z_prime)
        log_p_transition_latent_action = transition_probability_distribution.log_prob(z_prime)
        log_p_transition = tf.reduce_sum(log_p_transition_latent_action - log_p_transition_action, axis=1)

        # action reconstruction
        action_distribution = self.decode_action(z, latent_action)
        log_p_action = action_distribution.log_prob(a_1)

        rate = log_q_latent_action - log_p_latent_action
        distortion = -1. * (log_p_action + log_p_rewards + log_p_transition)

        self.loss_metrics['ELBO'](-1. * (distortion + rate))
        self.loss_metrics['action_mse'](a_1, action_distribution.sample())
        self.loss_metrics['reward_mse'](r_1, reward_distribution.sample())
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        self.loss_metrics['annealed_rate'](self.kl_scale_factor * rate)
        # self.loss_metrics['cross_entropy_regularizer'](cross_entropy_regularizer)

        return [distortion, rate, 0.]

    def eval(self, inputs):
        s_0, a_0, r_0, _, l_1 = (x[:, 0, :] for x in inputs)
        s_1, a_1, r_1, s_2, l_2 = (x[:, 1, :] for x in inputs)

        z = tf.cast(self.binary_encode(s_0, a_0, r_0, s_1, l_1).sample(), tf.float32)
        z_prime = tf.cast(self.binary_encode(s_1, a_1, r_1, s_2, l_2).sample(), tf.float32)
        q = self.discrete_action_encoding(z, a_1)
        p = self.discrete_action_prior()
        latent_action = tf.cast(q.sample(), tf.float32)

        # rewards reconstruction
        log_p_rewards_action = self._state_vae.reward_probability_distribution(z, a_1, z_prime).log_prob(r_1)
        log_p_rewards_latent_action = \
            self.reward_probability_distribution(z, latent_action, z_prime).reward_distribution.log_prob(r_1)
        log_p_rewards = log_p_rewards_latent_action - log_p_rewards_action

        # transition probability reconstruction
        log_p_transition_action = \
            self._state_vae.discrete_latent_transition_probability_distribution(z, a_1).log_prob(z_prime)
        log_p_transition_latent_action = \
            self.discrete_latent_transition_probability_distribution(z, latent_action).log_prob(z_prime)
        log_p_transition = tf.reduce_sum(log_p_transition_latent_action - log_p_transition_action, axis=1)

        # action reconstruction
        action_distribution = self.decode_action(z, latent_action)
        log_p_action = action_distribution.log_prob(a_1)

        rate = q.kl_divergence(p)
        distortion = -1. * (log_p_action + log_p_rewards + log_p_transition)

        return -1. * (distortion + rate)

    def mean_latent_bits_used(self, inputs, eps=1e-3):
        """
        Compute the mean number of bits used in the latent space of the vae_mdp for the given dataset batch.
        This allows monitoring if the latent space is effectively used by the VAE or if posterior collapse happens.
        """
        mean_bits_used = 0
        for i in (0, 1):
            s, a, r, s_prime, l_prime = (x[:, i, :] for x in inputs)
            z = tf.cast(self.binary_encode(s, a, r, s_prime, l_prime).sample(), tf.float32)
            mean = tf.reduce_mean(self.discrete_action_encoding(z, a).probs_parameter(), axis=0)
            check = lambda x: 1 if 1 - eps > x > eps else 0
            mean_bits_used += tf.reduce_sum(tf.map_fn(check, mean), axis=0).numpy()

        return mean_bits_used / 2
