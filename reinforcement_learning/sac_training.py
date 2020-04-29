import os

import PIL
import imageio
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver, greedy_policy, random_tf_policy

env_name = "HumanoidBulletEnv-v0"  # @param {type:"string"}

# use "num_iterations = 1e6" for better results,
# 1e5 is just so this doesn't take too long.
num_iterations = int(3e6)  # @param {type:"integer"}

initial_collect_steps = 10000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 1000000  # @param {type:"integer"}

batch_size = 256  # @param {type:"integer"}

critic_learning_rate = 3e-4  # @param {type:"number"}
actor_learning_rate = 3e-4  # @param {type:"number"}
alpha_learning_rate = 3e-4  # @param {type:"number"}
target_update_tau = 0.005  # @param {type:"number"}
target_update_period = 1  # @param {type:"number"}
gamma = 0.99  # @param {type:"number"}
reward_scale_factor = 1.0  # @param {type:"number"}
gradient_clipping = None  # @param

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 2500  # @param {type:"integer"}

num_eval_episodes = 30  # @param {type:"integer"}
eval_interval = 10000  # @param {type:"integer"}

py_env = suite_pybullet.load(env_name)
# py_env.render(mode='human')
py_env.reset()
img = PIL.Image.fromarray(py_env.render())
img.show()

tf_env = tf_py_environment.TFPyEnvironment(py_env)

observation_spec = tf_env.observation_spec()
action_spec = tf_env.action_spec()
critic_net = critic_network.CriticNetwork(
    (observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params)


# The output of the actor network is a normal distribution
def normal_projection_net(action_spec, init_means_output_factor=0.1):
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layer_params,
    continuous_projection_net=normal_projection_net)

global_step = tf.compat.v1.train.get_or_create_global_step()
tf_agent = sac_agent.SacAgent(
    tf_env.time_step_spec(),
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)
tf_agent.initialize()

# define the policy from the learning agent
eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_capacity)

# Dataset generates trajectories with shape [Bx2x...]
# Because SAC needs the current and the next state to perform the critic network updates
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

checkpoint_dir = os.path.join('saves/', 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=tf_agent,
    policy=collect_policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)
stochastic_policy_dir = os.path.join('saves/', 'stochastic_policy')
greedy_policy_dir = os.path.join('saves/', 'greedy_policy')
stochastic_policy_saver = policy_saver.PolicySaver(tf_agent.policy)
eval_policy_saver = policy_saver.PolicySaver(eval_policy)

num_episodes = tf_metrics.NumberOfEpisodes()
env_steps = tf_metrics.EnvironmentSteps()
avg_return = tf_metrics.AverageReturnMetric()
observers = [num_episodes, env_steps, avg_return, replay_buffer.add_batch]
# A driver executes the agent's exploration loop and allow the observers to collect exploration information
driver = dynamic_step_driver.DynamicStepDriver(
    tf_env, tf_agent.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env, collect_policy, observers=[replay_buffer.add_batch], num_steps=initial_collect_steps)


def train_and_eval():
    # Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)
    driver.run = common.function(driver.run)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    global global_step

    # Load the policy if it was trained beforehand
    if os.path.exists(checkpoint_dir):
        train_checkpointer.initialize_or_restore()
        # global_step = tf.compat.v1.train.get_global_step()
        print("Checkpoint loaded! global_step={}".format(global_step.value().numpy()))
    if not os.path.exists(stochastic_policy_dir):
        os.makedirs(stochastic_policy_dir)
    if not os.path.exists(greedy_policy_dir):
        os.makedirs(greedy_policy_dir)

    print("Initialize replay buffer...")
    initial_collect_driver.run()

    returns = []
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    print("Start training...")

    for _ in range(num_iterations):

        for _ in range(collect_steps_per_iteration):
            # Collect a few episodes using collect_policy and save to the replay buffer.
            driver.run()

        # Use data from the buffer and update the agent's network.
        # experience = replay_buffer.gather_all()
        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience)
        # replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            train_checkpointer.save(global_step)
            stochastic_policy_saver.save(stochastic_policy_dir)
            eval_policy_saver.save(greedy_policy_dir)
            with train_summary_writer.as_default():
                tf.summary.scalar('Loss', train_loss.loss, step=step)
                tf.summary.scalar('Training average returns', avg_return.result(), step=step)

        if step % eval_interval == 0:
            avg_eval_return = tf_metrics.AverageReturnMetric()
            if not os.path.exists('saves/eval'):
                os.makedirs('saves/eval')
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            video_filename = 'saves/eval/' + env_name + current_time + '.mp4'
            with imageio.get_writer(video_filename, fps=60) as video:
                tf_env.reset()
                dynamic_episode_driver.DynamicEpisodeDriver(tf_env, tf_agent.policy,
                                                            [avg_eval_return,
                                                             lambda _: video.append_data(py_env.render())],
                                                            num_episodes=num_eval_episodes).run()
            print('step = {0}: Average Return = {1}'.format(step, avg_eval_return.result()))
            with train_summary_writer.as_default():
                tf.summary.scalar('Expected returns', avg_eval_return.result(), step=step)
            returns.append(avg_eval_return.result())

    # Plots
    steps = range(0, num_iterations, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim()
    plt.savefig("saves/average_returns.png", dpi=300)


def render_policy():
    rendering_avg_return = tf_metrics.AverageReturnMetric()
    # saved_policy = tf.compat.v2.saved_model.load(stochastic_policy_dir)
    saved_policy = random_tf_policy.RandomTFPolicy(tf_agent.time_step_spec, action_spec)
    if not os.path.exists('saves/eval_videos'):
        os.makedirs('saves/eval_videos')
    video_filename = 'saves/eval_videos/' + env_name + '.mp4'
    tf_env.reset()
    with imageio.get_writer(video_filename, fps=60) as video:
        dynamic_episode_driver.DynamicEpisodeDriver(tf_env, saved_policy,
                                                    [rendering_avg_return,
                                                     lambda _: video.append_data(py_env.render())],
                                                    num_episodes=1).run()
    print('Average Return = {}'.format(rendering_avg_return.result()))


def env_random_policy_video_generation(env_name, episodes=1):
    py_env = suite_pybullet.load(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    if not os.path.exists('saves/eval_videos'):
        os.makedirs('saves/eval_videos')
    video_filename = 'saves/eval_videos/' + env_name + '_random_policy.mp4'
    with imageio.get_writer(video_filename, fps=60) as video:
        dynamic_episode_driver.DynamicEpisodeDriver(tf_env, random_policy,
                                                    [lambda _: video.append_data(py_env.render())],
                                                    num_episodes=episodes).run()
