import os
from typing import Callable, Optional, Tuple
import time

import tensorflow as tf
import tf_agents.policies
from scipy.sparse import dok_matrix
from scipy.sparse import spmatrix
import numpy as np
from tensorflow.python.keras.utils.generic_utils import Progbar
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
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
        episode_per_iteration: int = 1,
        num_episodes: int = int(2 ** 14),
) -> tf.SparseTensor:
    if collect_steps_per_iteration is None:
        collect_steps_per_iteration = batch_size
    replay_buffer_capacity = replay_buffer_capacity // num_parallel_calls
    collect_steps_per_iteration = collect_steps_per_iteration // num_parallel_calls

    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join('logs', environment_name, 'model_generation', current_time)
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    number_of_states = tf.pow(tf.constant(2, dtype=tf.int64), tf.constant(vae_mdp.latent_state_size, dtype=tf.int64))
    number_of_actions = tf.cast(vae_mdp.number_of_discrete_actions, tf.int64)

    mdp_matrix = tf.SparseTensor(
        indices=tf.zeros((0, 2), tf.int64),
        values=tf.cast([], dtype=tf.int32),
        dense_shape=[number_of_states * vae_mdp.number_of_discrete_actions, number_of_states]
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
        max_length=replay_buffer_capacity,
        dataset_drop_remainder=True,
        dataset_window_shift=1  # to retrieve all transitions
    )

    episode_counter = tf_metrics.NumberOfEpisodes()
    env_steps = tf_metrics.EnvironmentSteps()
    observers = [episode_counter, env_steps] if parallel_py_environment == 1 else []
    observers += [replay_buffer.add_batch]

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env, policy, observers=observers, num_episodes=episode_per_iteration
    )

    driver.run = common.function(driver.run)

    progressbar = Progbar(target=num_episodes, interval=0.1)

    for episode in tf.range(tf.cast(tf.math.floor(num_episodes / episode_per_iteration), tf.int64)):
        driver.run()
        states, actions, next_states = next(iter(
            replay_buffer.as_dataset(
                num_parallel_calls=num_parallel_calls,
                num_steps=2,
                single_deterministic_pass=True  # gather transitions only once
            ).map(
                map_func=retrieve_states_and_actions_indices,
                num_parallel_calls=num_parallel_calls,
                #  deterministic=False  # TF version >= 2.2.0
            ).batch(
                batch_size=replay_buffer.num_frames(),
                drop_remainder=False)
        ))

        with train_summary_writer.as_default():
            tf.summary.histogram('state_frequency', states, episode * episode_per_iteration)

        sparse_transitions = retrieve_sparse_transitions(
            states,
            actions,
            next_states,
            number_of_states,
            number_of_actions)
        mdp_matrix = update_mdp_sparse_tensor(mdp_matrix, sparse_transitions)

        replay_buffer.clear()
        progressbar.add(episode_per_iteration)

    return mdp_matrix


@tf.function
def retrieve_states_and_actions_indices(trajectory, info) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    latent_state = tf.cast(
        tf.reduce_sum(trajectory.observation * 2 ** tf.range(vae_mdp.latent_state_size), axis=-1),
        dtype=tf.int64
    )
    return latent_state[0], tf.cast(trajectory.action[0], tf.int64), latent_state[1]


@tf.function(experimental_relax_shapes=True)
def retrieve_sparse_transitions(
        states: tf.Tensor,
        actions: tf.Tensor,
        next_states: tf.Tensor,
        num_states: tf.int64,
        num_actions: tf.int64
) -> tf.SparseTensor:
    transitions = (states * num_actions * num_states +  # index of the state group
                   actions * num_states +  # index of the action group
                   next_states)  # index of the next state
    transitions, _, count = tf.unique_with_counts(transitions)

    transitions = tf.stack([transitions // num_states, transitions % num_states], axis=-1)
    sparse_transitions = tf.sparse.SparseTensor(
        indices=transitions,
        values=count,
        dense_shape=(num_states * num_actions, num_states)
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
    vae_mdp = variational_action_discretizer.load(
        # "/home/florentdelgrange/workspace/hpc_hydra/policy/Bipedal-walker/vae_LS15_MC16_CER10.0-decay=0.0015_KLA0.0"
        # "-growth=5e-06_TD1.00-0.95_1e-06-2e-06_step400000_eval_elbo53.687/step3000000/eval_elbo0.227"
        "saves/BipedalWalker-v2/models/vae_LS13_MC3_CER10.0-decay=0.0015_KLA0.0-growth=5e-06_TD1.00-0.90_1e-06-2e"
        "-06_params=relaxed_state_encoding_step320000_eval_elbo55.821/step320000/eval_elbo55.821/policy"
        "/action_discretizer/LA5_MC1_CER1.0-decay=0.001_KLA0.0-growth=5e-06_TD0.25-0.17_1e-06-2e-06_params"
        "=one_output_per_action-relaxed_state_encoding/step200000/eval_elbo-0.866"
    )

    from tf_agents.environments import suite_gym
    from reinforcement_learning import labeling_functions

    environment_name = 'BipedalWalker-v2'

    start_time = time.time()
    sparse_tensor_mdp = learn_empirical_mdp(
        environment_suite=suite_gym,
        labeling_function=labeling_functions[environment_name],
        environment_name=environment_name,
        vae_mdp=vae_mdp,
        policy=vae_mdp.get_simplified_policy(),
        num_parallel_calls=16,
        episode_per_iteration=8
    )

    tf.print(sparse_tensor_mdp, summarize=-1)
