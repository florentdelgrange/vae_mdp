from abc import ABC, abstractmethod
import tensorflow as tf
import os


class PriorityHandler(ABC):

    def __init__(self):
        self.step_counter = None
        self.max_priority = None

    @abstractmethod
    def update_priority(self, key, value):
        return NotImplemented
        pass

    @abstractmethod
    def load_or_initialize_checkpoint(self, dir_path: str):
        raise NotImplementedError()
        pass

    @abstractmethod
    def checkpoint(self, *args, **kwargs):
        raise NotImplementedError()
        pass


class LossPriority(PriorityHandler):

    def __init__(self, replay_buffer, epoch_steps: int = 10000, max_priority: tf.float64 = 10.,
                 smoothness: tf.float32 = 1.):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.max_priority = tf.Variable(initial_value=max_priority, dtype=tf.float64, name='max_priority')
        self.epoch_steps = epoch_steps
        self.smoothness = tf.constant(smoothness, dtype=tf.float64)
        self.max_loss = tf.Variable(0., trainable=False, dtype=tf.float64)
        self.min_loss = tf.Variable(0., trainable=False, dtype=tf.float64)

        self._checkpointer = None
        self._manager = None

    def update_priority(self, keys: tf.Tensor, loss: tf.Tensor):

        batch_size = tf.shape(loss)[0]
        self.step_counter.assign_add(batch_size)
        loss = tf.cast(loss, tf.float64)

        def update_loss_bounds():
            max_loss = tf.reduce_max(loss)
            min_loss = tf.reduce_min(loss)
            if self.step_counter < self.epoch_steps:
                if self.max_loss < max_loss:
                    self.max_loss.assign(max_loss)
                if self.min_loss < min_loss:
                    self.min_loss.assign(min_loss)
            elif self.step_counter % self.epoch_steps < batch_size:
                n = tf.cast(self.step_counter // self.epoch_steps, tf.float64)
                self.max_loss.assign((self.max_loss * (n - 1) + max_loss) / n)
                self.min_loss.assign((self.min_loss * (n - 1) + min_loss) / n)

        if self.step_counter < self.epoch_steps or self.step_counter % self.epoch_steps < batch_size:
            update_loss_bounds()

        L = self.max_priority
        x0 = (self.max_loss - self.min_loss) / 2.
        k = self.smoothness

        self.replay_buffer.update_priorities(keys=keys, priorities=L / (1. + tf.exp(-k * (loss - x0))))

    def load_or_initialize_checkpoint(self, dir_path: str):
        checkpoint_path = os.path.join(dir_path, 'loss_priority')
        self._checkpointer = tf.train.Checkpoint(
            max_loss=self.max_loss,
            min_loss=self.min_loss,
            step_counter=self.step_counter)
        self._manager = tf.train.CheckpointManager(
            checkpoint=self._checkpointer, directory=checkpoint_path, max_to_keep=1)
        self._checkpointer.restore(self._manager.latest_checkpoint)

    def checkpoint(self, *args, **kwargs):
        assert self._checkpointer is not None and self._manager is not None
        self._manager.save(*args, **kwargs)


class PriorityBuckets(PriorityHandler):

    def __init__(self, replay_buffer, latent_state_size: int):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        size = 2 ** latent_state_size
        self.latent_state_size = latent_state_size
        self._buckets = tf.Variable(
            initial_value=tf.zeros(shape=(size,), dtype=tf.int32), trainable=False, name='bucket')
        self.max_priority = tf.Variable(initial_value=0., dtype=tf.float64, name='max_priority')

        self._checkpointer = None
        self._manager = None

    def update_priority(self, keys: tf.Tensor, latent_states: tf.Tensor):
        batch_size = tf.shape(latent_states)[0]
        self.step_counter.assign_add(batch_size)

        indices = tf.reduce_sum(latent_states * 2 ** tf.range(self.latent_state_size), axis=-1)

        for index in indices:
            self._buckets[index].assign(self._buckets[index] + 1)
        priorities = tf.map_fn(
            fn=lambda index: self.step_counter / self._buckets[index],
            elems=indices,
            parallel_iterations=10,
            fn_output_signature=tf.float64)
        batch_max_priority = tf.reduce_max(priorities)

        if self.max_priority < batch_max_priority:
            self.max_priority.assign(batch_max_priority)

        self.replay_buffer.update_priorities(keys, priorities)

    def load_or_initialize_checkpoint(self, dir_path: str):
        checkpoint_path = os.path.join(dir_path, 'priority_buckets')
        self._checkpointer = tf.train.Checkpoint(
            buckets=self._buckets,
            step_counter=self.step_counter,
            max_priority=self.max_priority)
        self._manager = tf.train.CheckpointManager(
            checkpoint=self._checkpointer, directory=checkpoint_path, max_to_keep=1)
        self._checkpointer.restore(self._manager.latest_checkpoint)

    def checkpoint(self, *args, **kwargs):
        assert self._checkpointer is not None and self._manager is not None
        self._manager.save(*args, **kwargs)
