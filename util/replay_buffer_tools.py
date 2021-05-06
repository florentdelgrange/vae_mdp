from typing import Optional

import tensorflow as tf


class PriorityBuckets:

    def __init__(self, replay_buffer, latent_state_size: int):
        self.replay_buffer = replay_buffer
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        size = 2 ** latent_state_size
        self.latent_state_size = latent_state_size
        self.buckets = tf.Variable(initial_value=tf.zeros(shape=(size,), dtype=tf.int32), trainable=False)
        self.max_priority = tf.Variable(initial_value=1., dtype=tf.float64)

    def update_priority(self, keys: tf.Tensor, latent_states: tf.Tensor, value: Optional = None):
        keys = keys[keys < tf.uint64.max]
        latent_states = latent_states[keys < tf.uint64.max]
        batch_size = tf.shape(latent_states)[0]
        self.step_counter.assign_add(batch_size)

        indices = tf.reduce_sum(latent_states * 2 ** tf.range(self.latent_state_size), axis=-1)
        for index in indices:
            self.buckets[index].assign(self.buckets[index] + 1)
        priorities = tf.map_fn(
            fn=lambda index: self.step_counter / self.buckets[index],
            elems=indices,
            parallel_iterations=10,
            fn_output_signature=tf.float64)
        batch_max_priority = tf.reduce_max(priorities)

        if self.max_priority < batch_max_priority:
            self.max_priority.assign(batch_max_priority)

        self.replay_buffer.update_priorities(keys, priorities)
