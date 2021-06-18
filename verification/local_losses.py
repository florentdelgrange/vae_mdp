from collections import namedtuple
from typing import Optional, Callable
import time

import tensorflow as tf
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.utils import common
import tensorflow_probability as tfp

from util.io.dataset_generator import ErgodicMDPTransitionGenerator
from verification.latent_environment import DiscreteActionTFEnvironmentWrapper
from policies.latent_policy import LatentPolicyOverRealStateSpace
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
        estimate_transition_function_from_samples: bool = False,
        assert_transition_distribution: bool = False,
        fill_in_replay_buffer: bool = True,
        replay_buffer_max_frames: int = int(1e5)
):
    """
    Estimates reward and probability local losses from samples.
    :param environment: batched TFPyEnvironment
    :param latent_policy: a policy sets over latent states and producing (one-hot) latent actions
    :param steps: number of environment steps used to perform the estimation
    :param latent_state_size: binary size of the latent space
    :param number_of_discrete_actions: number of discrete actions
    :param state_embedding_function: mapping from real states and labels to binary states (of type int32)
    :param action_embedding_function: mapping from real states and discrete actions (given in base 10 with type int32)
                                      to real actions.
                                      Important: latent actions are here assumed to be given in base 10 and not in
                                      one-hot since environments generally process discrete actions represented in
                                      base 10 by convention.
    :param latent_reward_function: mapping from (binary) latent state, (one-hot) latent actions and (binary) next latent
                                   state to rewards. Latent states and actions are assumed to be of type int32 while
                                   returned reward type is float32.
    :param labeling_function: labeling function (mapping from real states to subset of atomic propositions)
    :param latent_transition_function: mapping from (binary) latent states and (one-hot) actions (given in int32) to
                                       a probability distribution over latent states.
                                       A probability distribution object is assumed to have a prob method (e.g.,
                                       tensorflow probability distribution objects) such that
                                       distribution.prob(label, latent_state_no_label) is the probability of the
                                       latent state given by tf.concat([label, latent_state_no_label], axis=-1)
    :param estimate_transition_function_from_samples: whether to estimate the latent transition function from samples
                                                      or not. If True, the latent transition function estimated this
                                                      way stores the probability matrix into a sparse tensor.
    :param assert_transition_distribution: whether to assert the transition function is correctly computed or not
    :return: a dictionary containing an estimation of the local reward and probability losses
    :param fill_in_replay_buffer: whether to fill in the replay buffer or not before the loss estimation.
    :param replay_buffer_max_frames: maximum number of frames to be contained in the replay buffer.
    """
    start_time = time.time()

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
        next_time_step=latent_environment.time_step_spec(),
    )
    # replay_buffer
    replay_buffer = TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=latent_environment.batch_size,
        max_length=replay_buffer_max_frames,
        # to retrieve all the transitions when single_deterministic_pass is True
        dataset_window_shift=1,
        dataset_drop_remainder=True)
    # initialize driver
    driver = DynamicStepDriver(
        env=latent_environment,
        policy=policy,
        num_steps=replay_buffer_max_frames if fill_in_replay_buffer else steps,
        observers=[replay_buffer.add_batch])
    driver.run = common.function(driver.run)
    # collect environment steps
    driver.run()

    collect_time = time.time() - start_time

    # retrieve dataset from the replay buffer
    generator = ErgodicMDPTransitionGenerator(
        labeling_function,
        replay_buffer,
        discrete_action=True,
        num_discrete_actions=number_of_discrete_actions)

    def sample_from_replay_buffer(num_transitions: Optional[int] = None, single_deterministic_pass=False):
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_steps=2,
            # whether to gather transitions only once or not
            single_deterministic_pass=single_deterministic_pass,
        ).map(
            map_func=generator,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            batch_size=num_transitions if num_transitions else replay_buffer.num_frames(),
            drop_remainder=False)
        dataset_iterator = iter(dataset)

        state, label, latent_action, reward, next_state, next_label = next(dataset_iterator)
        latent_state = state_embedding_function(state, label)
        next_latent_state_no_label = state_embedding_function(next_state, None)
        next_latent_state = tf.concat([tf.cast(next_label, dtype=tf.int32), next_latent_state_no_label], axis=-1)

        return namedtuple(
            'ErgodicMDPTransitionSample',
            ['state', 'label', 'latent_state', 'latent_action', 'reward', 'next_state', 'next_label',
             'next_latent_state_no_label', 'next_latent_state', 'batch_size'])(
            state, label, latent_state, latent_action, reward, next_state, next_label, next_latent_state_no_label,
            next_latent_state, tf.shape(state)[0])

    local_reward_loss_time = time.time()

    samples = sample_from_replay_buffer(num_transitions=steps)
    (state, label, latent_state, latent_action, reward, next_state, next_label, next_latent_state_no_label,
     next_latent_state) = (
        samples.state, samples.label, samples.latent_state, samples.latent_action, samples.reward,
        samples.next_state, samples.next_label, samples.next_latent_state_no_label, samples.next_latent_state)

    local_reward_loss = estimate_local_reward_loss(
        state, label, latent_action, reward, next_state, next_label,
        latent_reward_function, latent_state, next_latent_state)

    local_reward_loss_time = time.time() - local_reward_loss_time

    if estimate_transition_function_from_samples:

        transition_function_estimation_time = time.time()
        _samples = sample_from_replay_buffer(single_deterministic_pass=True)
        _transition_function_estimation_num_frames = _samples.batch_size
        latent_transition_function = TransitionFrequencyEstimator(
            _samples.latent_state, _samples.latent_action, _samples.next_latent_state,
            backup_transition_function=latent_transition_function,
            assert_distribution=assert_transition_distribution)

        transition_function_estimation_time = time.time() - transition_function_estimation_time
    else:
        _transition_function_estimation_num_frames = 0
        transition_function_estimation_time = 0

    local_probability_loss_time = time.time()

    local_probability_loss = estimate_local_probability_loss(
        state, label, latent_action, next_state, next_label,
        latent_transition_function, latent_state, next_latent_state_no_label)

    local_probability_loss_time = time.time() - local_probability_loss_time

    def print_time_metrics():
        print("Time to fill in the Replay Buffer ({:d} frames): {:.3f}".format(replay_buffer_max_frames, collect_time))
        print("Time to estimate the local reward loss function (from {:d} transitions):"
              " {:.3f}".format(steps, local_reward_loss_time))
        if estimate_transition_function_from_samples:
            print("Time to estimate the probability transition function (from {:d} transitions): {:3f}".format(
                _transition_function_estimation_num_frames, transition_function_estimation_time))
        print("Time to estimate the local probability loss function (from {:d} transitions):"
              " {:.3f}".format(steps, local_probability_loss_time))

    return namedtuple(
        'LocalLossesEstimationMetrics',
        ['local_reward_loss', 'local_probability_loss', 'print_time_metrics'])(
          local_reward_loss,   local_probability_loss,   print_time_metrics)


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
