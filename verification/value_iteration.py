import tensorflow as tf

def sparse_value_iteration(sparse_transition_matrix: tf.SparseTensor, gamma: float = 0.99, epsilon: float = 1e-4):
    V = tf.sparse.reduce_sum