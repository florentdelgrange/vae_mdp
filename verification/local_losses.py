from collections import Callable

import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.utils import common
import tensorflow_probability as tfp
tfd = tfp.distributions


def compute_local_losses_from_samples(
        environment: TFPyEnvironment, latent_policy: tf_policy.Base, steps: int, latent_state_size: int):
    # create environment wrapper
    # create dataset
    # replay_buffer
    replay_buffer = ReplayBuffer()
    # create driver
    driver = DynamicStepDriver(environment, latent_policy, num_steps=steps, observers=[replay_buffer.add_batch])
    driver.run = common.function(driver.run)

    state, label, latent_action, reward, next_latent_state, next_label = next(dataset_iterator)


def estimate_local_probability_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        embed_state: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        latent_probability_transition_function: tfd.Distribution,
        latent_state_size: int,
):
    all_latent_states = tf.range(2 ** tf.cast(latent_state_size))
    all_latent_states = tf.map_fn(lambda n: (n // 2 ** tf.range(latent_state_size)) % 2, all_latent_states)

    def total_variation(state, label, latent_action, next_state, next_label):
        latent_state = embed_state(state, label)
        next_latent_state = embed_state(next_state, next_label)
        next_latent_state = tf.map_fn(lambda s: s == next_latent_state, all_latent_states)

    tf.map_fn(total_variation, inputs)