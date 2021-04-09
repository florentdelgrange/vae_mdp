from collections import Callable
from typing import Optional

import tensorflow as tf
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.utils import common
import tensorflow_probability as tfp

from util.io.dataset_generator import ErgodicMDPTransitionGenerator
from verification.latent_environment import LatentPolicyOverRealStateSpace, DiscreteActionTFEnvironmentWrapper

tfd = tfp.distributions


def frequency_estimation_probability_function(
        latent_state: tf.Tensor,
        latent_action: tf.Tensor,
        next_latent_state: tf.Tensor,  # Optional[tf.Tensor] = None,
        # next_latent_state_no_label: Optional[tf.Tensor] = None,
        # next_label: Optional[tf.Tensor] = None,
        # label_size: Optional[int] = None,
):
    #  if (next_latent_state is None) == (next_latent_state_no_label is None):
    #      raise ValueError('Must either pass next latent states or next latent states without labels')
    #  if next_latent_state_no_label is not None:
    #      assert next_label is not None
    #  else:
    #      assert label_size is not None

    #  if next_latent_state_no_label is None:
    #  next_label,  = next_latent_state[..., :label_size]
    #  next_latent_state_no_label = next_latent_state[...: label_size:]

    latent_state_size = tf.shape(latent_state)[1]  # first axis is batch, second is latent state size
    number_of_discrete_actions = tf.shape(latent_action)[1]  # first axis is batch, second is a one-hot vector
    latent_state_indices = tf.reduce_sum(latent_state * 2 ** tf.range(latent_state_size), axis=-1)
    latent_action_indices = tf.argmax(latent_action, axis=-1)
    next_latent_state_indices = tf.reduce_sum(latent_state * 2 ** tf.range(latent_state_size), axis=-1)


def compute_local_losses_from_samples(
        environment: TFPyEnvironment,
        latent_policy: tf_policy.Base,
        steps: int,
        latent_state_size: int,
        number_of_discrete_actions: int,
        state_embedding_function: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        labeling_function: Callable[[tf.Tensor], tf.Tensor],
):
    # generate environment wrapper for discrete actions
    latent_environment = DiscreteActionTFEnvironmentWrapper(
        tf_env=environment,
        action_embedding_function=action_embedding_function,
        number_of_discrete_actions=number_of_discrete_actions)
    # set the latent policy over real states
    policy = LatentPolicyOverRealStateSpace(
        time_step_spec=latent_environment.time_step_spec(),
        labeling_function=labeling_function,
        latent_policy=latent_policy,
        state_embedding_function=state_embedding_function)
    policy_step_spec = policy_step.PolicyStep(action=latent_environment.action_spec(), state=(), info=())
    trajectory_spec = trajectory.from_transition(
        latent_environment.time_step_spec(), policy_step_spec, latent_environment.time_step_spec())
    # replay_buffer
    replay_buffer = TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=latent_environment.batch_size,
        max_length=steps,
        dataset_drop_remainder=True,
        # set the window shift is one to gather all transitions when the batch size corresponds to max_length
        dataset_window_shift=1)
    # retrieve dataset from the replay buffer
    generator = ErgodicMDPTransitionGenerator(
        labeling_function,
        replay_buffer,
        discrete_action=True,
        num_discrete_actions=number_of_discrete_actions)
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        num_steps=2,
        single_deterministic_pass=True  # gather transitions only once
    ).map(
        map_func=generator,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(
        batch_size=replay_buffer.num_frames(),
        drop_remainder=False)
    dataset_iterator = iter(dataset)
    # initialize driver
    driver = DynamicStepDriver(latent_environment, policy, num_steps=steps, observers=[replay_buffer.add_batch])
    driver.run = common.function(driver.run)
    # collect environment steps
    driver.run()

    state, label, latent_action, reward, next_state, next_label = next(dataset_iterator)
    latent_state = state_embedding_function(state, label)
    next_latent_state_no_label = state_embedding_function(next_state)
    next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)
    local_reward_loss = estimate_local_reward_loss(
        state, label, latent_action, reward, next_state, next_label,
        latent_reward_function, latent_state, next_latent_state)
    local_probability_loss = estimate_local_probability_loss(
        state, label, latent_action, next_state, next_label, latent_transition_function,
        latent_state_size, latent_state, next_latent_state_no_label, next_latent_state)
    return local_reward_loss, local_probability_loss


def generate_binary_latent_state_space(latent_state_size):
    all_latent_states = tf.range(2 ** latent_state_size)
    return tf.map_fn(lambda n: (n // 2 ** tf.range(latent_state_size)) % 2, all_latent_states)


def estimate_local_reward_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        reward: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
):
    if latent_state is None:
        latent_state = state_embedding_function(state, label)
    if next_latent_state is None:
        next_latent_state = state_embedding_function(next_state, next_label)

    return tf.reduce_mean(tf.abs(reward - latent_reward_function(latent_state, latent_action, next_latent_state)))


def estimate_local_probability_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        latent_state_size: int,
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state_no_label: Optional[tf.Tensor] = None,
        next_latent_state: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
        all_latent_states: Optional[tf.Tensor] = None
):
    if all_latent_states is None:
        all_latent_states = generate_binary_latent_state_space(latent_state_size)

    def total_variation(
            state, label, latent_action, next_state, next_label,
            latent_state=None, next_latent_state_no_label=None, next_latent_state=None):
        if latent_state is None:
            latent_state = state_embedding_function(state, label)
        if next_latent_state_no_label is None:
            next_latent_state_no_label = state_embedding_function(next_state)
        if next_latent_state is None:
            next_latent_state = tf.concat([next_label, next_latent_state_no_label], axis=-1)

        p_next_latent_state = tf.cast(tf.reduce_all(next_latent_state == all_latent_states, axis=-1), dtype=tf.float32)
        latent_transition_distribution = latent_transition_function(latent_state, latent_action)
        return tf.reduce_sum(tf.abs(
            p_next_latent_state - latent_transition_distribution.prob(next_label, next_latent_state_no_label)),
            axis=-1)

    return tf.reduce_mean(tf.map_fn(total_variation, (
        state, label, latent_action, next_state, next_label,
        latent_state, next_latent_state_no_label, next_latent_state)))
