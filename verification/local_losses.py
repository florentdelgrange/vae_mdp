from typing import Optional, Callable

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.utils import common
import tensorflow_probability as tfp

from util.io.dataset_generator import ErgodicMDPTransitionGenerator
from verification.latent_environment import LatentPolicyOverRealStateSpace, DiscreteActionTFEnvironmentWrapper
from verification.transition_function import TransitionFrequencyEstimator

tfd = tfp.distributions


def estimate_local_losses_from_samples(
        environment: TFPyEnvironment,
        latent_policy: tf_policy.TFPolicy,
        steps: int,
        latent_state_size: int,
        number_of_discrete_actions: int,
        state_embedding_function: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        latent_reward_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        labeling_function: Callable[[tf.Tensor], tf.Tensor],
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution] = None,
        estimate_transition_function_from_samples: bool = False
):
    if latent_transition_function is None and not estimate_transition_function_from_samples:
        raise ValueError('no latent transition function provided')
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
    # policy_step_spec = policy_step.PolicyStep(action=latent_environment.action_spec(), state=(), info=())
    trajectory_spec = trajectory.from_transition(
        time_step=latent_environment.time_step_spec(),
        action_step=policy.policy_step_spec,
        next_time_step=latent_environment.time_step_spec())
    # replay_buffer
    replay_buffer = TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=latent_environment.batch_size,
        max_length=steps,
        dataset_drop_remainder=True,
        # set the window shift to one to gather all transitions when the batch size corresponds to max_length
        dataset_window_shift=1)
    # initialize driver
    driver = DynamicStepDriver(latent_environment, policy, num_steps=steps, observers=[replay_buffer.add_batch])
    driver.run = common.function(driver.run)
    # collect environment steps
    driver.run()

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

    state, label, latent_action, reward, next_state, next_label = next(dataset_iterator)
    latent_state = state_embedding_function(state, label)
    next_latent_state_no_label = state_embedding_function(next_state, None)
    next_latent_state = tf.concat([tf.cast(next_label, dtype=tf.int32), next_latent_state_no_label], axis=-1)
    local_reward_loss = estimate_local_reward_loss(
        state, label, latent_action, reward, next_state, next_label,
        latent_reward_function, latent_state, next_latent_state)

    if estimate_transition_function_from_samples:
        latent_transition_function = TransitionFrequencyEstimator(
            latent_state, latent_action, next_latent_state, backup_transition_function=latent_transition_function)
        driver.run()

    local_probability_loss = estimate_local_probability_loss(
        state, label, latent_action, next_state, next_label,
        latent_transition_function, latent_state, next_latent_state_no_label)

    return {'local_reward_loss': local_reward_loss, 'local_probability_loss': local_probability_loss}


def generate_binary_latent_state_space(latent_state_size):
    all_latent_states = tf.range(2 ** latent_state_size)
    return tf.map_fn(lambda n: (n // 2 ** tf.range(latent_state_size)) % 2, all_latent_states)


@tf.function
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


@tf.function
def estimate_local_probability_loss(
        state: tf.Tensor,
        label: tf.Tensor,
        latent_action: tf.Tensor,
        next_state: tf.Tensor,
        next_label: tf.Tensor,
        latent_transition_function: Callable[[tf.Tensor, tf.Tensor], tfd.Distribution],
        latent_state: Optional[tf.Tensor] = None,
        next_latent_state_no_label: Optional[tf.Tensor] = None,
        state_embedding_function: Optional[Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]] = None,
):

    if latent_state is None:
        latent_state = state_embedding_function(state, label)
    if next_latent_state_no_label is None:
        next_latent_state_no_label = state_embedding_function(next_state, None)

    next_label = tf.cast(next_label, tf.float32)
    next_latent_state_no_label = tf.cast(next_latent_state_no_label, tf.float32)
    latent_transition_distribution = latent_transition_function(latent_state, latent_action)
    return tf.reduce_mean(1. - latent_transition_distribution.prob(next_label, next_latent_state_no_label))
