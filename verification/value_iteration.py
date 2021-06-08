from typing import Callable, Optional

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

error = 1e-7

def sparse_value_iteration(
        sparse_transition_matrix: tf.SparseTensor,
        backup_transition_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        num_actions: int,
        gamma: float = 0.99,
        epsilon: float = 1e-4,
        policy: Optional[Callable[[tf.Tensor], tfd.Categorical]] = None,
        latent_state_size: Optional[int] = None,
        number_of_latent_states: Optional[int] = None,
):
    if (latent_state_size is None) == (number_of_latent_states is None):
        raise ValueError('Must either pass the latent space in binary (via latent_state_size) or in unary '
                         '(via number_of_latent_states)')

    if number_of_latent_states is None:
        number_of_latent_states = 2 ** latent_state_size - 1

    values = tf.ones(shape=(number_of_latent_states,), dtype=tf.float32)

    @tf.function
    def values_update(
            sparse_transition_matrix: tf.SparseTensor,
            values: tf.Tensor,
    ):
        #  i = 0
        #  while i < tf.shape(sparse_transition_matrix.indices)[0]:
        #      state = sparse_transition_matrix.indices[i, 0]
        #      _action = sparse_transition_matrix.indices[i, 1]
        states = tf.sparse.reduce_sum(
            sp_input=sparse_transition_matrix,
            axis=[1, 2],
            output_is_sparse=True)
        for state in states.indices:
            enabled_actions = policy(state).probs(tf.range(num_actions)) > error
            for action in tf.where(enabled_actions):
                transition_values = tf.sparse.slice(
                    sp_input=sparse_transition_matrix,
                    start=[tf.squeeze(state), tf.squeeze(action), 0],
                    size=[1, 1, number_of_latent_states])
                if tf.size(transition_values.values) > 0:
                    while sparse_transition_matrix.indices[i, 1] == action:
                        next_state = sparse_transition_matrix.indices[i, 2]
                        if transition_values is None:
                            transition_values =
