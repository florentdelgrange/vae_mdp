import base64
import os

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver


if __name__ == '__main__':
    # Set up a virtual display for rendering OpenAI gym environments.
    # display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    env_name = "BipedalWalker-v3"  # @param {type:"string"}
    num_iterations = 250  # @param {type:"integer"}
    collect_episodes_per_iteration = 2  # @param {type:"integer"}
    replay_buffer_capacity = 2048  # @param {type:"integer"}

    fc_layer_params = (256, 128)

    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 5  # @param {type:"integer"}
    num_eval_episodes = 5  # @param {type:"integer"}
    eval_interval = 10  # @param {type:"integer"}
    discount_factor = .99

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=global_step,
        gamma=discount_factor)
    tf_agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    checkpoint_dir = os.path.join('saves/', 'checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    policy_dir = os.path.join('saves/', 'policy')
    tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)

    num_episodes = tf_metrics.NumberOfEpisodes()
    env_steps = tf_metrics.EnvironmentSteps()
    avg_return = tf_metrics.AverageReturnMetric()
    observers = [num_episodes, env_steps, avg_return, replay_buffer.add_batch]
    # A driver executes the agent's exploration loop and allow the observers to collect exploration information
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env, tf_agent.collect_policy, observers=observers, num_episodes=collect_episodes_per_iteration)

    # Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)
    print("Start training...")

    for _ in range(num_iterations):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        driver.run()
        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        # replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            train_checkpointer.save(global_step)
            tf_policy_saver.save(policy_dir)

        if step % eval_interval == 0:
            avg_return = tf_metrics.AverageReturnMetric()
            dynamic_episode_driver.DynamicEpisodeDriver(eval_env, tf_agent.policy, [avg_return,
                                                                                    lambda _: eval_env.render()],
                                                        num_episodes=num_eval_episodes).run()
            print('step = {0}: Average Return = {1}'.format(step, avg_return.result()))
