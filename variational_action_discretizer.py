import os
from typing import Tuple, Optional, List, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model, Sequential
from tensorflow.keras.layers import Input, Concatenate, Reshape, Dense, Lambda
from tensorflow.keras.utils import Progbar

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
                 encoder_temperature: float = 1.,
                 prior_temperature: float = 1.,
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

        self.number_of_discrete_actions = number_of_discrete_actions
        self._states_vae = vae_mdp

        self.action_encoder_temperature = tf.Variable(encoder_temperature, dtype=tf.float32, trainable=False)
        self.action_prior_temperature = tf.Variable(prior_temperature, dtype=tf.float32, trainable=False)
        self.action_encoder_temperature_decay_rate = tf.constant(encoder_temperature_decay_rate, dtype=tf.float32)
        self.action_prior_temperature_decay_rate = tf.constant(prior_temperature_decay_rate, dtype=tf.float32)
        self.annealing_pairs.append([(self.action_encoder_temperature, self.action_encoder_temperature_decay_rate,
                                      self.action_prior_temperature, self.action_prior_temperature_decay_rate)])
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
        self.action_prior_logits = tf.Variable(shape=(number_of_discrete_actions,), name='prior_action_logits')

        # discrete actions transition network
        transition_outputs = []
        transition_network_pre_processing = clone_model(pre_processing_network)(latent_state)
        for _ in range(number_of_discrete_actions):  # branching-action network
            _transition_network = clone_model(transition_network)
            _transition_network.layers.pop(0)  # remove the old input
            _transition_network = _transition_network(transition_network_pre_processing)
            transition_outputs.append(_transition_network)
        next_latent_state_logits = Lambda(lambda outputs: tf.stack(outputs))(latent_state)
        self.discrete_actions_transition_network = Model(input=latent_state, outputs=next_latent_state_logits,
                                                         name="discrete_actions_transition_network")

        # discrete actions reward network
        reward_network_input = Concatenate()([latent_state, next_latent_state])
        reward_network_pre_processing = clone_model(pre_processing_network)(reward_network_input)
        reward_network_outputs = []
        for _ in range(number_of_discrete_actions):  # branching-action network
            _reward_network = clone_model(reward_network)
            _reward_network.layers.pop(0)  # remove the old input
            _reward_network = reward_network(reward_network_pre_processing)
            reward_network_outputs.append(_reward_network)
        reward_network_mean = \
            Lambda(lambda outputs: tf.stack(outputs))(list(mean for mean, covariance in reward_network_outputs))
        reward_network_raw_covariance = \
            Lambda(lambda outputs: tf.stack(outputs))(list(covariance for mean, covariance in reward_network_outputs))
        self.discrete_actions_reward_network = Model(inputs=[latent_state, next_latent_state],
                                                     outputs=[reward_network_mean, reward_network_raw_covariance],
                                                     name="discrete_actions_reward_network")

        # discrete actions decoder
        action_decoder_pre_processing = clone_model(pre_processing_network)(latent_state)
        action_decoder_outputs = []
        for action in range(number_of_discrete_actions):  # branching-action network
            action_decoder = clone_model(action_decoder_network)
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
            Lambda(lambda outputs: tf.stack(outputs))(list(mean for mean, covariance in action_decoder_outputs))
        action_decoder_raw_covariance = \
            Lambda(lambda outputs: tf.stack(outputs))(list(covariance for mean, covariance in action_decoder_outputs))
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

    def relaxed_action_prior(self, temperature: float):
        return tfd.ExpRelaxedOneHotCategorical(
            temperature=temperature, logits=self.action_prior_logits, allow_nan_stats=False)

    def discrete_latent_transition_probability_distribution(
            self, latent_state: tf.Tensor, latent_action: tf.Tensor
    ) -> tfd.Distribution:
        next_latent_state_logits = self.discrete_actions_transition_network(latent_state)
        return tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=latent_action),
                                     components_distribution=tfd.Bernoulli(logits=next_latent_state_logits),
                                     allow_nan_stats=False)

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

        z = self.binary_encode(s_0, a_0, r_0, s_1, l_1).sample()
        z_prime = self.binary_encode(s_1, a_1, r_1, s_2, l_2).sample()
        q = self.relaxed_action_encoding(z, a_1, self.action_encoder_temperature)
        p = self.relaxed_action_prior(self.action_prior_temperature)
        exp_latent_action = q.sample()
        latent_action = tf.exp(exp_latent_action)

        log_q_latent_action = q.log_prob(exp_latent_action)
        log_p_latent_action = p.log_prob(exp_latent_action)

        # rewards reconstruction
        reward_distribution = self.reward_probability_distribution(z, latent_action, z_prime)
        log_p_rewards_action = self._states_vae.reward_probability_distribution(z, a_1, z_prime).log_prob(r_1)
        log_p_rewards_latent_action = reward_distribution.log_prob(r_1)
        log_p_rewards = log_p_rewards_action - log_p_rewards_latent_action

        # transition probability reconstruction
        transition_probability_distribution = self.discrete_latent_transition_probability_distribution(z, latent_action)
        log_p_transition_action = \
            self._states_vae.discrete_latent_transition_probability_distribution(z, a_1).log_prob(z_prime)
        log_p_transition_latent_action = transition_probability_distribution.log_prob(z_prime)
        log_p_transition = tf.reduce_sum(log_p_transition_action - log_p_transition_latent_action, axis=1)

        # action reconstruction
        action_distribution = self.decode_action(z, latent_action)
        log_p_action = action_distribution.log_prob(a_1)

        rate = log_q_latent_action - log_p_latent_action
        distortion = log_p_action + log_p_rewards + log_p_transition

        self.loss_metrics['ELBO'](-1 * (distortion + rate))
        self.loss_metrics['action_mse'](a_1, action_distribution.sample())
        self.loss_metrics['reward_mse'](r_1, reward_distribution.sample())
        self.loss_metrics['distortion'](distortion)
        self.loss_metrics['rate'](rate)
        self.loss_metrics['annealed_rate'](self.kl_scale_factor * rate)
        # self.loss_metrics['cross_entropy_regularizer'](cross_entropy_regularizer)

        return [distortion, rate]
