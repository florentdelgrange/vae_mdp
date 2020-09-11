from typing import Callable, Optional, Tuple
import time

import tensorflow as tf
import tf_agents.policies
from scipy.sparse import dok_matrix
from scipy.sparse import spmatrix
import numpy as np
from tensorflow.python.keras.utils.generic_utils import Progbar
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment, py_environment, tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.utils import common

from util.io.dataset_generator import map_rl_trajectory_to_vae_input
from variational_mdp import VariationalMarkovDecisionProcess
from variational_action_discretizer import VariationalActionDiscretizer

import variational_action_discretizer

def learn_empirical_mdp(
        environment_suite,
        labeling_function: Callable,
        environment_name: str,
        vae_mdp: VariationalActionDiscretizer,
        num_parallel_calls: int = 1,
        batch_size: int = 512,
        replay_buffer_capacity: int = int(1e6),
        collect_steps_per_iteration: Optional[int] = None,
        policy: Optional[tf_agents.policies.tf_policy.Base] = None,
        initial_collect_steps: int = int(1e4),
        num_iterations: int = int(1e6),
        use_sparse_tensors: bool = True
) -> spmatrix:
    if collect_steps_per_iteration is None:
        collect_steps_per_iteration = batch_size
    replay_buffer_capacity = replay_buffer_capacity // num_parallel_calls
    collect_steps_per_iteration = collect_steps_per_iteration // num_parallel_calls

    number_of_states = tf.pow(tf.constant(2, dtype=tf.int64), tf.constant(vae_mdp.latent_state_size, dtype=tf.int64))

    if use_sparse_tensors:
        mdp_matrix = tf.SparseTensor(
            indices=tf.zeros((0, 2), tf.int64),
            values=tf.cast([], dtype=tf.int32),
            dense_shape=[number_of_states * vae_mdp.number_of_discrete_actions, number_of_states]
        )
    else:
        mdp_matrix = dok_matrix(
            (number_of_states * vae_mdp.number_of_discrete_actions, number_of_states),
            dtype=np.int32
        )

    if num_parallel_calls > 1:
        tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
            [lambda: environment_suite.load(environment_name)] * num_parallel_calls))
        tf_env.reset()
    else:
        py_env = environment_suite.load(environment_name)
        py_env.reset()
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

    tf_env = vae_mdp.wrap_tf_environment(tf_env, labeling_function)
    if policy is None:
        policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=tf_env.action_spec()
        )

    action_spec = tf_env.action_spec()

    # specs
    policy_step_spec = policy_step.PolicyStep(
        action=action_spec,
        state=(),
        info=())
    trajectory_spec = trajectory.from_transition(
        tf_env.time_step_spec(),
        policy_step_spec,
        tf_env.time_step_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=trajectory_spec,
        batch_size=tf_env.batch_size,
        max_length=batch_size // num_parallel_calls  # to remove transitions already considered
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=num_parallel_calls,
        num_steps=2,
        single_deterministic_pass=True  # gather transitions only once
    ).map(
        map_func=retrieve_states_and_actions_indices,
        num_parallel_calls=num_parallel_calls,
        #  deterministic=False  # TF version >= 2.2.0
    ).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    num_episodes = tf_metrics.NumberOfEpisodes()
    env_steps = tf_metrics.EnvironmentSteps()
    observers = [num_episodes, env_steps] if parallel_py_environment == 1 else []
    observers += [replay_buffer.add_batch]

    driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, policy, observers=observers, num_steps=collect_steps_per_iteration)
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, policy, observers=observers, num_steps=initial_collect_steps)

    driver.run = common.function(driver.run)

    progressbar = Progbar(target=num_iterations, interval=0.1)

    # print("Initial collect steps...")
    # initial_collect_driver.run()
    # print("Start training.")

    for _ in range(num_iterations):
        driver.run()
        dataset_iterator = iter(dataset)
        states, actions, next_states = next(dataset_iterator)

        if use_sparse_tensors:
            sparse_transitions = retrieve_sparse_transitions(
                states,
                actions,
                next_states,
                number_of_states,
                vae_mdp.number_of_discrete_actions)
            mdp_matrix = update_mdp_sparse_tensor(mdp_matrix, sparse_transitions)
        else:
            update_mdp_sparse_matrix(
                states=states,
                actions=actions,
                next_states=next_states,
                num_parallel_calls=num_parallel_calls,
                mdp_matrix=mdp_matrix)
        progressbar.add(1)

    return mdp_matrix


@tf.function
def retrieve_states_and_actions_indices(trajectory, info) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    tf.print(trajectory.observation, 'observation', summarize=-1)

    latent_state = tf.cast(
        tf.reduce_sum(trajectory.observation * 2 ** tf.range(vae_mdp.latent_state_size), axis=-1),
        dtype=tf.int64
    )
    tf.print(latent_state, 'observation', summarize=-1)
    state, next_state = latent_state[0], latent_state[1]
    tf.print(trajectory.action[0], 'action_played', summarize=-1)

    action = tf.cast(trajectory.action[0], tf.int64)
    tf.print(action, 'action in one hot', summarize=-1)

    return state, action, next_state


@tf.function
def retrieve_sparse_transitions(
        states: tf.Tensor,
        actions: tf.Tensor,
        next_states: tf.Tensor,
        num_states: int,
        num_actions: int
) -> tf.SparseTensor:
    transitions = (states * num_actions * num_states +  # index of the state group
                   actions * num_states +  # index of the action group
                   next_states)  # index of the next state
    transitions, _, count = tf.unique_with_counts(transitions)

    transitions = tf.stack([transitions // num_states, transitions % num_states], axis=-1)
    sparse_transitions = tf.sparse.SparseTensor(
        indices=transitions,
        values=count,
        dense_shape=[num_states * num_actions, num_states]
    )
    return tf.sparse.reorder(sparse_transitions)


@tf.function
def update_mdp_sparse_tensor(
        mdp_matrix: tf.sparse.SparseTensor, sparse_transitions: tf.sparse.SparseTensor
) -> tf.SparseTensor:
    return tf.sparse.add(mdp_matrix, sparse_transitions)


# @tf.function  # cannot use tf.function due to scipy mdp_matrix instance
def update_mdp_sparse_matrix(states, actions, next_states, num_parallel_calls, mdp_matrix):
    def increment_mdp_matrix_with_transition(transition):
        state, action, next_state = transition[..., 0], transition[..., 1], transition[..., 2]
        mdp_matrix[state * vae_mdp.number_of_discrete_actions + action, next_state] += 1
        return transition

    transitions = tf.stack([states, actions, next_states], axis=-1)
    tf.map_fn(increment_mdp_matrix_with_transition, transitions, parallel_iterations=num_parallel_calls)


if __name__ == '__main__':
    dir = "saves/BipedalWalker-v2/models/vae_LS14_MC5_CER10.0_KLA0.0_TD1.00-0.90_1e-06-2e-06_step410000_eval_elbo52.650/policy/action_discretizer/LA6_MC16_CER1.0-decay=0.001_KLA0.0-growth=5e-06_TD0.20-0.13_1e-06-2e-06_params=one_output_per_action_base/step200000/eval_elbo0.426"
    vae_mdp = variational_action_discretizer.load(dir)

    from tf_agents.environments import suite_gym
    from reinforcement_learning import labeling_functions

    environment_name = 'BipedalWalker-v2'
    num_iterations = int(1e4)

    start_time = time.time()
    sparse_tensor_mdp = learn_empirical_mdp(
        environment_suite=suite_gym,
        labeling_function=labeling_functions[environment_name],
        environment_name=environment_name,
        vae_mdp=vae_mdp,
        num_parallel_calls=1,
        num_iterations=num_iterations
    )

    print('Time to create an MDP via sparse tensors in {} iterations: {} sec'.format(num_iterations,
                                                                                     time.time() - start_time))

    start_time = time.time()
    dict_sparse_matrix_mdp = learn_empirical_mdp(
        environment_suite=suite_gym,
        labeling_function=labeling_functions[environment_name],
        environment_name=environment_name,
        vae_mdp=vae_mdp,
        num_parallel_calls=1,
        num_iterations=num_iterations,
        use_sparse_tensors=False
    )

    print('Time to create an MDP via a scipy dict sparse matrix in {} iterations:'
          '{} sec'.format(num_iterations, time.time() - start_time))
