import os
from typing import Tuple, Optional, List, Callable
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, Concatenate, TimeDistributed, Reshape, Dense, Lambda
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
        self.action_encoder_temperature = tf.Variable(encoder_temperature, dtype=tf.float32, trainable=False)
        self.action_prior_temperature = tf.Variable(prior_temperature, dtype=tf.float32, trainable=False)
        self.action_encoder_temperature_decay_rate = tf.constant(encoder_temperature_decay_rate, dtype=tf.float32)
        self.action_prior_temperature_decay_rate = tf.constant(prior_temperature_decay_rate, dtype=tf.float32)
        self.annealing_pairs.append([(self.action_encoder_temperature, self.action_encoder_temperature_decay_rate,
                                      self.action_prior_temperature, self.action_prior_temperature_decay_rate)])

        # action encoder network
        latent_state = Input(shape=(self.latent_state_size,))
        action = Input(shape=self.action_shape)
        reward = Input(shape=self.reward_shape)
        next_latent_state = Input(shape=(self.latent_state_size,))
        encoder_input = Concatenate(name="action_encoder_input")([latent_state, action, reward, next_latent_state])
        action_encoder = action_encoder_network(encoder_input)
        action_encoder = Dense(units=number_of_discrete_actions, activation=None,
                               name='encoder_exp_one_hot_relaxation_logits')(action_encoder)
        self.action_encoder = Model(inputs=[latent_state, action, reward, next_latent_state], outputs=action_encoder,
                                    name="action_encoder")

        # prior over actions
        self.action_prior_logits = tf.Variable(shape=(number_of_discrete_actions,), name='prior_action_logits')

        # discrete actions transition network
        transition_network_input = Input(self.latent_state_size)
        transition_outputs = []
        for _ in range(number_of_discrete_actions):  # branching-actions network
            transition_network = clone_model(self.transition_network)
            transition_network.layers.pop(0)  # remove the old input
            transition_outputs.append(transition_network(transition_network_input))
        transition_output = Lambda(lambda outputs: tf.stack(outputs))(transition_outputs)
        self.discrete_actions_transition_network = Model(input=transition_network_input, outputs=transition_output,
                                                         name="discrete_actions_transition_network")

        # discrete actions reward network
        reward_network_input = Concatenate()([latent_state, next_latent_state])
        reward_network_outputs = []
        for _ in range(number_of_discrete_actions):  # branching-actions network
            reward_network = clone_model(self.reward_network)
            reward_network.layers.pop(0)  # remove the old input
            reward_network_outputs.append(reward_network(reward_network_input))
        reward_network_mean = Lambda(lambda outputs: tf.stack)(list(output[0] for output in reward_network_outputs))
        reward_network_raw_covariance = \
            Lambda(lambda outputs: tf.stack)(list(output[1] for output in reward_network_outputs))
        self.discrete_actions_reward_network = Model(inputs=[latent_state, next_latent_state],
                                                     outputs=[reward_network_mean, reward_network_raw_covariance],
                                                     name="discrete_actions_reward_network")

        # discrete actions decoder
        action_decoder_outputs = []
        for _ in range(number_of_discrete_actions):  # branching-actions network
            action_decoder = clone_model(action_decoder_network)
            action_decoder = action_decoder(next_latent_state)
            action_decoder = Dense(units=np.prod(self.action_shape), activation=None)(action_decoder)
            action_decoder = Reshape(target_shape=self.action_shape)(action_decoder)
            action_decoder_outputs.append(action_decoder)
        action_decoder_output = Lambda(lambda outputs: tf.stack)(action_decoder_outputs)
        self.action_decoder_network = Model(inputs=next_latent_state, outputs=action_decoder_output,
                                            name="action_decoder_network")

        state_layers = (self.encoder_network.layers,
                        self.transition_network.layers,
                        self.reward_network.layers,
                        self.reconstruction_network.layers)

        # freeze all latent states training related layers
        for layers in state_layers:
            for layer in layers:
                layer.trainable = False
