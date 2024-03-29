from collections import namedtuple
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class TransitionFrequencyEstimator:
    def __init__(
            self,
            latent_states: tf.Tensor,
            latent_actions: tf.Tensor,
            next_latent_states: tf.Tensor,
            backup_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
            assert_distribution: bool = False
    ):
        self.latent_state_size = tf.shape(latent_states)[1]  # first axis is batch, second is latent state size
        self.num_states = 2 ** self.latent_state_size
        self.num_actions = tf.shape(latent_actions)[1]  # first axis is batch, second is a one-hot vector
        self.backup_transition_function = backup_transition_function

        @tf.function
        def compute_transition_counter():
            states = tf.reduce_sum(latent_states * 2 ** tf.range(self.latent_state_size), axis=-1)
            actions = tf.cast(tf.argmax(latent_actions, axis=-1), dtype=tf.int32)
            next_states = tf.reduce_sum(next_latent_states * 2 ** tf.range(self.latent_state_size), axis=-1)

            # flat transition indices
            transitions = states * self.num_actions * self.num_states + actions * self.num_states + next_states
            transitions, _, count = tf.unique_with_counts(transitions)
            transitions = tf.stack([transitions // (self.num_states * self.num_actions),  # state index
                                    (transitions // self.num_states) % self.num_actions,  # action index
                                    transitions % self.num_states],  # next state index
                                   axis=-1)
            transitions = tf.cast(transitions, dtype=tf.int64)
            transition_counter = tf.sparse.SparseTensor(
                indices=transitions,
                values=tf.cast(count, tf.float32),
                dense_shape=(self.num_states, self.num_actions, self.num_states))
            return tf.sparse.reorder(transition_counter)

        transition_counter = compute_transition_counter()
        probs = tf.Variable(tf.cast(transition_counter.values, dtype=tf.float32), trainable=False)
        i = tf.Variable(0, trainable=False)
        j = tf.Variable(0, trainable=False)

        @tf.function
        def compute_transition_probabilities(
                transition_counter: tf.sparse.SparseTensor,
                probs: tf.Variable,
                i: tf.Variable,
                j: tf.Variable):
            state_action_pair_counter = tf.sparse.reduce_sum(transition_counter, axis=-1, output_is_sparse=True)
            indices = transition_counter.indices[..., :-1]
            while i < tf.shape(probs)[0]:
                if tf.reduce_all(indices[i] == state_action_pair_counter.indices[j], axis=-1):
                    probs[i].assign(tf.cast(transition_counter.values[i], dtype=tf.float32) /
                                    tf.cast(state_action_pair_counter.values[j], dtype=tf.float32))
                    i.assign_add(1)
                else:
                    j.assign_add(1)  # works only if indices are ordered

            transition_tensor = tf.sparse.SparseTensor(
                indices=transition_counter.indices,
                values=probs,
                dense_shape=(self.num_states, self.num_actions, self.num_states))

            return tf.sparse.reorder(transition_tensor)

            # probs = tf.reduce_sum(
            #     tf.map_fn(
            #         fn=lambda x: tf.where(
            #             condition=tf.reduce_all(transition_counter.indices[..., :-1] == x[0], axis=-1),
            #             x=transition_counter.values / x[1],
            #             y=tf.zeros(tf.shape(transition_counter.values))),
            #         elems=(state_action_pair_counter.indices,
            #                state_action_pair_counter.values),
            #         fn_output_signature=tf.float32),
            #     axis=0)

        self.transitions = compute_transition_probabilities(transition_counter, probs, i, j)
        self.enabled_actions = tf.cast(
            tf.sparse.reduce_sum(self.transitions, axis=-1, output_is_sparse=True),
            dtype=tf.bool)
        if assert_distribution:
            epsilon = 1e-6
            state_action_pairs = tf.sparse.reduce_sum(self.transitions, axis=-1, output_is_sparse=True)
            tf.assert_less(tf.abs(1. - state_action_pairs.values), epsilon)

    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor):
        state = tf.reduce_sum(latent_state * 2 ** tf.range(self.latent_state_size), axis=-1)
        action = tf.argmax(latent_action, axis=-1)

        @tf.function
        def _get_prob_value(transition):
            latent_state, state, action, next_latent_state_no_label, next_label = transition
            next_state = tf.cast(tf.concat([next_label, next_latent_state_no_label], axis=-1), tf.int32)
            next_state = tf.reduce_sum(next_state * 2 ** tf.range(self.latent_state_size), axis=-1)

            # check if the action has been visited in the given state during the transition sparse tensor construction
            action_is_enabled = tf.squeeze(tf.sparse.slice(self.enabled_actions, [state, action], [1, 1]).values)
            if tf.size(action_is_enabled) > 0 and action_is_enabled:
                probs = tf.squeeze(tf.sparse.slice(self.transitions, [state, action, next_state], [1, 1, 1]).values)
                # if the entry of the slice is empty, then the probability is 0
                return 0. if tf.equal(tf.size(probs), 0) else probs
            # if not, then use a backup transition function to retrieve the probability distribution for [state, action]
            else:
                return tf.squeeze(
                    self.backup_transition_function(
                        tf.expand_dims(latent_state, axis=0),
                        tf.expand_dims(tf.one_hot(action, depth=self.num_actions), axis=0)
                    ).prob(tf.expand_dims(next_label, axis=0),
                           tf.expand_dims(next_latent_state_no_label, axis=0)))

        @tf.function
        def _prob(*value):
            next_label, next_latent_state_no_label = value
            return tf.map_fn(
                fn=_get_prob_value,
                elems=(latent_state, state, action, next_latent_state_no_label, next_label),
                fn_output_signature=tf.float32)

        return namedtuple('next_state_transition_distribution', ['prob'])(_prob)
