from collections import Callable, namedtuple
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


class FrequencyEstimationTransitionFunction:
    def __init__(self, latent_states: tf.Tensor, latent_actions: tf.Tensor, next_latent_states: tf.Tensor):
        self.num_states = tf.shape(latent_states)[1]  # first axis is batch, second is latent state size
        self.num_actions = tf.shape(latent_actions)[1]  # first axis is batch, second is a one-hot vector
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
        state_action_pair_counter = tf.sparse.reduce_sum(transition_counter, axis=-1, output_is_sparse=True)
        probs = tf.reduce_sum(
            tf.map_fn(
                fn=lambda x: tf.where(
                    condition=tf.reduce_all(transition_counter.indices[..., :-1] == x[0], axis=-1),
                    x=transition_counter.values / x[1],
                    y=tf.zeros(tf.shape(transition_counter.values))),
                elems=(state_action_pair_counter.indices, state_action_pair_counter.values)),
            axis=0)

        self.transition_tensor = tf.sparse.SparseTensor(
            indices=transitions, values=probs, dense_shape=(self.num_states, self.num_actions, self.num_states))

    def __call__(self, latent_state: tf.Tensor, latent_action: tf.Tensor, *next_latent_state):
        state = tf.reduce_sum(latent_state * 2 ** tf.range(self.num_states), axis=-1)
        action = tf.argmax(latent_action, axis=-1)

        def _get_prob_value(*args):
            state, action, next_state = args
            probs = tf.squeeze(tf.sparse.slice(self.transition_tensor, [state, action, next_state], [1, 1, 1]))
            return 0. if tf.equal(tf.size(probs), 0) else probs

        def _prob(*value):
            next_latent_state = tf.concat(value, axis=-1)
            next_state = tf.reduce_sum(next_latent_state * 2 ** tf.range(self.num_states), axis=-1)
            return tf.map_fn(fn=_get_prob_value, elems=(state, action, next_state))

        return namedtuple('next_state_transition_distribution', ['prob'])(_prob)


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
