from collections import namedtuple

import tensorflow as tf


class TransitionFrequencyEstimator:
    def __init__(self, latent_states: tf.Tensor, latent_actions: tf.Tensor, next_latent_states: tf.Tensor):
        self.num_states = tf.shape(latent_states)[1]  # first axis is batch, second is latent state size
        self.num_actions = tf.shape(latent_actions)[1]  # first axis is batch, second is a one-hot vector

        @tf.function
        def estimate_transition_tensor():
            states = tf.reduce_sum(latent_states * 2 ** tf.range(self.num_states), axis=-1)
            actions = tf.argmax(latent_actions, axis=-1)
            next_states = tf.reduce_sum(next_latent_states * 2 ** tf.range(self.num_states), axis=-1)

            # flat transition indices
            transitions = states * self.num_actions * self.num_states + actions * self.num_states + next_states
            transitions, _, count = tf.unique_with_counts(transitions)
            transitions = tf.stack([transitions // (self.num_states * self.num_actions),  # state index
                                    (transitions // self.num_states) % self.num_actions,  # action index
                                    transitions % self.num_states],  # next state index
                                   axis=-1)
            transition_counter = tf.sparse.SparseTensor(
                indices=transitions, values=count, dense_shape=(self.num_states, self.num_actions, self.num_states))
            transition_counter = tf.sparse.reorder(transition_counter)
            state_action_pair_counter = tf.sparse.reduce_sum(transition_counter, axis=-1, output_is_sparse=True)

            probs = tf.Variable(transition_counter.values)
            indices = transition_counter.indices[..., :-1]
            j = tf.Variable(0)
            for i in tf.range(tf.shape(probs)[0]):
                if tf.reduce_all(indices[i] == state_action_pair_counter.indices[j], axis=-1):
                    probs[i].assign(probs[i] / state_action_pair_counter.values[j])
                else:
                    j.assign_add(1)  # works by assuming ordered indices

            #  probs = tf.reduce_sum(
            #      tf.map_fn(
            #          fn=lambda x: tf.where(
            #              condition=tf.reduce_all(transition_counter.indices[..., :-1] == x[0], axis=-1),
            #              x=transition_counter.values / x[1],
            #              y=tf.zeros(tf.shape(transition_counter.values))),
            #          elems=(state_action_pair_counter.indices, state_action_pair_counter.values)),
            #      axis=0)

            return tf.sparse.SparseTensor(
                indices=transitions, values=probs, dense_shape=(self.num_states, self.num_actions, self.num_states))

        self.transition_tensor = estimate_transition_tensor()

    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor, *next_latent_state):
        state = tf.reduce_sum(latent_state * 2 ** tf.range(self.num_states), axis=-1)
        action = tf.argmax(latent_action, axis=-1)

        @tf.function
        def _get_prob_value(next_state: tf.Tensor):
            probs = tf.squeeze(tf.sparse.slice(self.transition_tensor, [state, action, next_state], [1, 1, 1]))
            return 0. if tf.equal(tf.size(probs), 0) else probs

        @tf.function
        def _prob(*value):
            next_latent_state = tf.concat(value, axis=-1)
            next_state = tf.reduce_sum(next_latent_state * 2 ** tf.range(self.num_states), axis=-1)
            return tf.map_fn(fn=_get_prob_value, elems=next_state)

        return namedtuple('next_state_transition_distribution', ['prob'])(_prob)
