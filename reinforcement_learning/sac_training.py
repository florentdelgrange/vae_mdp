import os
from typing import Tuple, Callable
import threading
import datetime

import PIL
import imageio

import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import tf_agents.trajectories.time_step as ts

from util.io import dataset_generator

labeling_functions = {
    'HumanoidBulletEnv-v0':
        lambda states: states.take(0, axis=-1) + 0.8 <= 0.78,  # falling down
    # np.count_nonzero(np.abs(observation[:, :, 8: 42][0::2]) > 0.99) > 0  # has stuck joints
}


class NumberOfSafetyViolations:
    def __init__(self, labeling_function):
        self._n = 0
        self._num_episodes = 0
        self.labeling_function = labeling_function

    def __call__(self, *args, **kwargs):
        trajectory = args[0]
        if self.labeling_function(trajectory.observation.numpy()).any():
            self._n += 1
        if trajectory.step_type == ts.StepType.LAST:
            self._num_episodes += 1

    def result(self):
        return self._n

    def average(self):
        return self._n / self._num_episodes


class SACLearner:
    def __init__(self,
                 env_name: str = 'HumanoidBulletEnv-v0',
                 num_iterations: int = int(3e6),
                 initial_collect_steps: int = int(1e4),
                 collect_steps_per_iteration: int = 1,
                 replay_buffer_capacity: int = int(1e6),
                 critic_learning_rate: float = 3e-4,
                 actor_learning_rate: float = 3e-4,
                 alpha_learning_rate: float = 3e-4,
                 target_update_tau: float = 5e-3,
                 target_update_period: int = 1,
                 gamma: float = 0.99,
                 reward_scale_factor: float = 20.0,
                 gradient_clipping=None,
                 actor_fc_layer_params: Tuple[int, ...] = (256, 256),
                 critic_joint_fc_layer_params: Tuple[int, ...] = (256, 256),
                 log_interval: int = 2500,
                 num_eval_episodes: int = 30,
                 eval_interval: int = int(1e4),
                 parallelization: bool = True,
                 num_parallel_environments: int = 4,
                 batch_size: int = 256,
                 labeling_function: Callable = labeling_functions['HumanoidBulletEnv-v0'],
                 eval_video: bool = False,
                 debug: bool = False,
                 save_directory_location: str = '.'):
        self.env_name = env_name

        self.num_iterations = num_iterations

        self.initial_collect_steps = initial_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity

        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.target_update_tau = target_update_tau
        self.target_update_period = target_update_period
        self.gamma = gamma
        self.reward_scale_factor = reward_scale_factor
        self.gradient_clipping = gradient_clipping

        self.actor_fc_layer_params = actor_fc_layer_params
        self.critic_joint_fc_layer_params = critic_joint_fc_layer_params

        self.log_interval = log_interval

        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval

        self.parallelization = parallelization
        self.num_parallel_environments = num_parallel_environments

        self.batch_size = batch_size

        self.eval_video = eval_video

        if parallelization:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: suite_pybullet.load(env_name)] * num_parallel_environments))
            self.tf_env.reset()
            self.py_env = suite_pybullet.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = suite_pybullet.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
            self.eval_env = self.tf_env

        self.observation_spec = self.tf_env.observation_spec()
        self.action_spec = self.tf_env.action_spec()
        self.critic_net = critic_network.CriticNetwork(
            (self.observation_spec, self.action_spec),
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

        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.observation_spec,
            self.action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=normal_projection_net)

        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.tf_agent = sac_agent.SacAgent(
            self.tf_env.time_step_spec(),
            self.action_spec,
            actor_network=self.actor_net,
            critic_network=self.critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            train_step_counter=self.global_step)
        self.tf_agent.initialize()

        # define the policy from the learning agent
        self.collect_policy = self.tf_agent.collect_policy

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.tf_agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=replay_buffer_capacity)

        # Dataset generates trajectories with shape [Bx2x...]
        # Because SAC needs the current and the next state to perform the critic network updates
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

        self.checkpoint_dir = os.path.join(save_directory_location, 'saves', 'sac_training_checkpoint')
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.tf_agent,
            policy=self.collect_policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
        )
        self.stochastic_policy_dir = os.path.join(save_directory_location, 'saves', 'stochastic_policy')
        self.stochastic_policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)

        self.num_episodes = tf_metrics.NumberOfEpisodes()
        self.env_steps = tf_metrics.EnvironmentSteps()
        self.avg_return = tf_metrics.AverageReturnMetric()

        observers = [self.num_episodes, self.env_steps, self.avg_return] if not parallelization else []
        observers += [self.replay_buffer.add_batch]
        # A driver executes the agent's exploration loop and allows the observers to collect exploration information
        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
        self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.tf_env, self.collect_policy, observers=[self.replay_buffer.add_batch], num_steps=initial_collect_steps)

        self.labeling_function = labeling_function

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = save_directory_location

    def train_and_eval(self):
        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        self.driver.run = common.function(self.driver.run)

        # load the checkpoint
        if os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            self.global_step = tf.compat.v1.train.get_global_step()
            print("Checkpoint loaded! global_step={}".format(self.global_step.numpy()))
        if not os.path.exists(self.stochastic_policy_dir):
            os.makedirs(self.stochastic_policy_dir)
        print("Initialize replay buffer...")
        self.initial_collect_driver.run()

        print("Start training...")

        for _ in range(self.num_iterations):

            for _ in range(self.collect_steps_per_iteration):
                # Collect a few steps using collect_policy and save to the replay buffer.
                self.driver.run()

            # Use data from the buffer and update the agent's network.
            # experience = replay_buffer.gather_all()
            experience, unused_info = next(self.iterator)
            train_loss = self.tf_agent.train(experience)

            step = self.tf_agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))
                self.train_checkpointer.save(self.global_step)
                self.stochastic_policy_saver.save(self.stochastic_policy_dir)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Loss', train_loss.loss, step=step)
                    if not self.parallelization:
                        tf.summary.scalar('Training average returns', self.avg_return.result(), step=step)

            if step % self.eval_interval == 0:
                eval_thread = threading.Thread(target=self.eval, args=(step,), daemon=True, name='eval')
                eval_thread.start()

            if step % (self.replay_buffer_capacity // 4) == 0:
                dataset_generation_thread = threading.Thread(target=self.save_observations,
                                                             args=(self.replay_buffer_capacity // 4,),
                                                             daemon=True,
                                                             name='dataset_gen')
                dataset_generation_thread.start()

    def eval(self, step: int = 0):
        avg_eval_return = tf_metrics.AverageReturnMetric()
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric()
        saved_policy = tf.compat.v2.saved_model.load(self.stochastic_policy_dir)
        num_safety_violations = NumberOfSafetyViolations(labeling_function=self.labeling_function)
        if self.eval_video:
            self.evaluate_policy_video(saved_policy,
                                       observers=[avg_eval_return, avg_eval_episode_length, num_safety_violations],
                                       step=str(step))
        else:
            self.eval_env.reset()
            dynamic_episode_driver.DynamicEpisodeDriver(self.eval_env, saved_policy,
                                                        [avg_eval_return, avg_eval_episode_length,
                                                         num_safety_violations],
                                                        num_episodes=self.num_eval_episodes).run()
        print('step = {0}: Average Return = {1}'.format(step, avg_eval_return.result()))
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Average returns', avg_eval_return.result(), step=step)
            tf.summary.scalar('Average episode length', avg_eval_episode_length.result(), step=step)
            tf.summary.scalar('Number of safety violations for {} episodes'.format(self.num_eval_episodes),
                              num_safety_violations.result(),
                              step=step)

    def save_observations(self, batch_size: int = 128000):
        observations_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=3)
        dataset_generator.gather_rl_observations(iter(observations_dataset),
                                                 self.labeling_function,
                                                 dataset_path=os.path.join(self.save_directory_location,
                                                                           'dataset/reinforcement_learning'))
        print("observation dataset saved")

    def evaluate_policy_video(self, policy, observers=None, step=''):
        if observers is None:
            observers = []
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        video_filename = 'saves/eval/' + self.env_name + current_time + '_step' + step + '.mp4'
        with imageio.get_writer(video_filename, fps=60) as video:
            self.eval_env.reset()
            dynamic_episode_driver.DynamicEpisodeDriver(self.eval_env, policy,
                                                        observers + [lambda _: video.append_data(self.py_env.render())],
                                                        num_episodes=self.num_eval_episodes).run()