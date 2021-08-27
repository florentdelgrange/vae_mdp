import functools
import json
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

from collections import namedtuple
from typing import Tuple, Callable, Optional
import threading
import timeit
import datetime
import numpy as np
import random

import reverb
import tf_agents
from absl import app
from absl import flags

import PIL
import imageio

import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import Progbar

from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.agents.ddpg import critic_network

from tf_agents.drivers import dynamic_step_driver, py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.metrics import tf_metrics, tf_metric, py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.trajectories.trajectory import experience_to_transitions
from tf_agents.utils import common
from tf_agents.policies import policy_saver, py_tf_eager_policy
import tf_agents.trajectories.time_step as ts

from reinforcement_learning.environments import EnvironmentLoader
from reinforcement_learning.environments.PerturbedEnvironment import PerturbedEnvironment

from util.io import dataset_generator

flags.DEFINE_string(
    'env_name', help='Name of the environment', default='HumanoidBulletEnv-v0'
)
flags.DEFINE_string(
    'env_suite', help='Environment suite', default='suite_pybullet'
)
flags.DEFINE_integer(
    'steps', help='Number of iterations', default=int(1.2e7)
)
flags.DEFINE_integer(
    'num_parallel_env', help='Number of parallel environments', default=1
)
flags.DEFINE_boolean(
    'permissive_policy_saver',
    help="Set this flag to save a permissive variance policy of the current agent's policy",
    default=False
)
flags.DEFINE_multi_float(
    'variance',
    help='variance multiplier for the permissive variance policy',
    default=[1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
)
flags.DEFINE_string(
    'save_dir', help='Save directory location', default='.'
)
flags.DEFINE_integer(
    'seed', help='set seed', default=None
)
flags.DEFINE_integer(
    'collect_steps_per_iteration',
    help='Collect steps per iteration',
    default=1
)
flags.DEFINE_integer(
    'batch_size',
    help='Batch size',
    default=64
)
flags.DEFINE_float(
    'reward_scale_factor',
    help='scale factor for rewards',
    default=1.
)
flags.DEFINE_bool(
    'prioritized_experience_replay',
    help="use priority-based replay buffer (with Deepmind's reverb)",
    default=False
)
flags.DEFINE_float(
    'priority_exponent',
    help='priority exponent for computing the probabilities of the samples from the prioritized replay buffer',
    default=0.6
)
flags.DEFINE_float(
    'gamma',
    help='discount factor',
    default=0.99
)
flags.DEFINE_float(
    'learning_rate',
    help='optimizer learning rate',
    default=3e-4
)
flags.DEFINE_float(
    'target_update_tau',
    help='target update tau',
    default=5e-3
)
flags.DEFINE_integer(
    'replay_buffer_size',
    help='replay buffer size',
    default=int(1e6)
)
flags.DEFINE_float(
    'state_perturbation',
    help='add perturbations to the state space to train policies robust to state noise',
    default=0.
)
flags.DEFINE_float(
    'action_perturbation',
    help='add perturbations to the action space to train policies robust to action noise',
    default=0.
)


FLAGS = flags.FLAGS


class NumberOfSafetyViolations(tf_metric.TFStepMetric):
    """
    Experimental
    """

    def __init__(self, labeling_function, name='NumberOfSafetyViolations'):
        super(NumberOfSafetyViolations, self).__init__(name=name, prefix='Metrics')
        self._n = common.create_variable(name='num_violations', initial_value=0., trainable=False, dtype=tf.float32)
        self._num_episodes = common.create_variable(
            name='num_episodes', initial_value=1., trainable=False, dtype=tf.float32)
        self.labeling_function = labeling_function

    @common.function(autograph=True)
    def call(self, trajectory):
        self._n.assign_add(tf.reduce_sum(tf.cast(self.labeling_function(trajectory.observation), tf.float32)))
        self._num_episodes.assign_add(tf.reduce_sum(
            tf.cast(trajectory.step_type[..., 0] == ts.StepType.LAST, tf.float32)))

        return trajectory

    def result(self):
        return self._n

    @common.function
    def average(self):
        return self._n / self._num_episodes

    @common.function
    def reset(self):
        self._n.assign(0.)
        self._num_episodes.assign(1.)


class SACLearner:
    def __init__(
            self,
            env_name: str,
            env_suite,
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
            batch_size: int = 64,
            eval_video: bool = False,
            debug: bool = False,
            save_directory_location: str = '.',
            save_exploration_dataset: bool = False,
            prioritized_experience_replay: bool = False,
            priority_exponent: float = 0.6,
            state_perturbation: float = 0.,
            action_perturbation: float = 0.,
            seed: Optional[int] = None,
    ):

        self.parallelization = parallelization and not prioritized_experience_replay

        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = batch_size // 8
        if parallelization:
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_environments
            collect_steps_per_iteration = max(1, collect_steps_per_iteration // num_parallel_environments)

        self.env_name = env_name
        # self.labeling_function = labeling_function
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

        self.num_parallel_environments = num_parallel_environments

        self.batch_size = batch_size

        self.eval_video = eval_video

        self.prioritized_experience_replay = prioritized_experience_replay

        env_loader = EnvironmentLoader(env_suite, seed=seed)
        if state_perturbation > 0. or action_perturbation > 0.:
            _load = env_loader.load
            env_loader.load = lambda env_name: PerturbedEnvironment(
                env=_load(env_name),
                state_noise=state_perturbation,
                action_noise=action_perturbation)

        if parallelization:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_loader.load(env_name)] * num_parallel_environments))
            self.tf_env.reset()
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = env_loader.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
            self.eval_env = tf_py_environment.TFPyEnvironment(env_loader.load(env_name))

        self.observation_spec = self.tf_env.observation_spec()
        self.action_spec = self.tf_env.action_spec()
        self.critic_net = critic_network.CriticNetwork(
            (self.observation_spec, self.action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform'
        )

        # The output of the actor network is a normal distribution
        def normal_projection_net(action_spec, init_means_output_factor=0.1, std_transform_multiplier=1.):
            return normal_projection_network.NormalProjectionNetwork(
                action_spec,
                mean_transform=None,
                state_dependent_std=True,
                init_means_output_factor=init_means_output_factor,
                std_transform=lambda std: std_transform_multiplier * sac_agent.std_clip_transform(std),
                scale_distribution=True)

        self.normal_projection_net = normal_projection_net

        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.observation_spec,
            self.action_spec,
            fc_layer_params=actor_fc_layer_params,
            # continuous_projection_net=normal_projection_net)
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.td_errors_loss_fn = tf.math.squared_difference
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
            td_errors_loss_fn=self.td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            train_step_counter=self.global_step)
        self.tf_agent.initialize()

        # define the policy from the learning agent
        self.collect_policy = self.tf_agent.collect_policy

        self.max_priority = tf.Variable(0., trainable=False, name='max_priority', dtype=tf.float64)
        if self.prioritized_experience_replay:
            checkpoint_path = os.path.join(save_directory_location, 'saves', env_name, 'reverb')
            reverb_checkpointer = reverb.checkpointers.DefaultCheckpointer(checkpoint_path)

            table_name = 'prioritized_replay_buffer'
            table = reverb.Table(
                table_name,
                max_size=replay_buffer_capacity,
                sampler=reverb.selectors.Prioritized(priority_exponent=priority_exponent),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1))

            reverb_server = reverb.Server([table], checkpointer=reverb_checkpointer)

            self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                data_spec=self.tf_agent.collect_data_spec,
                sequence_length=2,
                table_name=table_name,
                local_server=reverb_server)

            _add_trajectory = reverb_utils.ReverbAddTrajectoryObserver(
                py_client=self.replay_buffer.py_client,
                table_name=table_name,
                sequence_length=2,
                stride_length=1,
                priority=self.max_priority)

            self.num_episodes = py_metrics.NumberOfEpisodes()
            self.env_steps = py_metrics.EnvironmentSteps()
            self.avg_return = py_metrics.AverageReturnMetric()
            observers = [self.num_episodes, self.env_steps, self.avg_return, _add_trajectory]

            self.driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=observers,
                max_steps=collect_steps_per_iteration)
            self.initial_collect_driver = py_driver.PyDriver(
                env=self.py_env,
                policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, use_tf_function=True),
                observers=[_add_trajectory],
                max_steps=initial_collect_steps)

        else:
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.tf_agent.collect_data_spec,
                batch_size=self.tf_env.batch_size,
                max_length=replay_buffer_capacity)

            self.num_episodes = tf_metrics.NumberOfEpisodes()
            self.env_steps = tf_metrics.EnvironmentSteps()
            self.avg_return = tf_metrics.AverageReturnMetric(batch_size=self.tf_env.batch_size)
            #  self.safety_violations = NumberOfSafetyViolations(self.labeling_function)

            observers = [self.num_episodes, self.env_steps] if not parallelization else []
            observers += [self.avg_return, self.replay_buffer.add_batch]
            # A driver executes the agent's exploration loop and allows the observers to collect exploration information
            self.driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
            self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                self.tf_env,
                self.collect_policy,
                observers=[self.replay_buffer.add_batch],
                num_steps=initial_collect_steps)

        # Dataset generates trajectories with shape [Bx2x...]
        # Because SAC needs the current and the next state to perform the critic network updates
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

        self.policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'sac_policy')
        if state_perturbation > 0 or action_perturbation > 0:
            self.policy_dir = os.path.join(
                self.policy_dir,
                'perturbation_robustness_state={:.2g}_action={:.2g}'.format(state_perturbation, action_perturbation))
        self.policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)
        self._policy_dir = os.path.join(self.policy_dir, 'tmp')
        self._policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)
        self.score = tf.Variable(-1. * np.inf, trainable=False)

        self.checkpoint_dir = os.path.join(save_directory_location, 'saves', env_name, 'sac_training_checkpoint')
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.tf_agent,
            policy=self.collect_policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
            observers=observers,
            score=self.score
        )

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(
            save_directory_location, 'logs', 'gradient_tape', env_name, 'sac_agent_training', current_time)
        print("logs are written to", train_log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = os.path.join(save_directory_location, 'saves', env_name)
        self.save_exploration_dataset = save_exploration_dataset

    def save_permissive_variance_policy(self, variance_multiplier: float = 1.5):

        def map_proj(spec):
            return self.normal_projection_net(spec, std_transform_multiplier=tf.math.sqrt(variance_multiplier))

        projection_networks = tf.nest.map_structure(map_proj, self.action_spec)

        actual_projection_networks = self.actor_net._projection_networks
        self.actor_net._projection_networks = projection_networks

        checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.tf_agent,
            policy=self.collect_policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
        )
        _policy_dir = os.path.join(
            self.save_directory_location,
            'policy',
            'permissive_variance_policy-multiplier={}'.format(variance_multiplier))

        checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()
        print("Checkpoint loaded! global_step={}".format(self.global_step.numpy()))
        self._policy_saver.save(_policy_dir)

        self.actor_net._projection_networks = actual_projection_networks

    def _compute_priorities(self, experience):
        assert self.prioritized_experience_replay

        transitions = experience_to_transitions(experience, squeeze_time_dim=True)
        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action

        next_actions, next_log_pis = self.tf_agent._actions_and_log_probs(next_time_steps)
        target_input = (next_time_steps.observation, next_actions)
        target_q_values1, unused_network_state1 = self.tf_agent._target_critic_network_1(
            target_input, next_time_steps.step_type, training=False)
        target_q_values2, unused_network_state2 = self.tf_agent._target_critic_network_2(
            target_input, next_time_steps.step_type, training=False)
        target_q_values = (
                tf.minimum(target_q_values1, target_q_values2) -
                tf.exp(self.tf_agent._log_alpha) * next_log_pis)

        td_targets = tf.stop_gradient(
            self.reward_scale_factor * next_time_steps.reward +
            self.gamma * next_time_steps.discount * target_q_values)

        pred_input = (time_steps.observation, actions)
        pred_td_targets1, _ = self.tf_agent._critic_network_1(
            pred_input, time_steps.step_type, training=False)
        pred_td_targets2, _ = self.tf_agent._critic_network_2(
            pred_input, time_steps.step_type, training=False)
        critic_loss1 = self.td_errors_loss_fn(td_targets, pred_td_targets1)
        critic_loss2 = self.td_errors_loss_fn(td_targets, pred_td_targets2)
        loss = critic_loss1 + critic_loss2
        return tf.cast(loss, dtype=tf.float64)

    def train_and_eval(self, display_progressbar: bool = True, display_interval: float = 0.1):
        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        self._compute_priorities = common.function(self._compute_priorities)
        if not self.prioritized_experience_replay:
            self.driver.run = common.function(self.driver.run)

        metrics = [
            # 'avg_safety_violations_per_episode',
            'eval_avg_returns',
            'avg_eval_episode_length',
            'eval_safety_violations',
            'replay_buffer_frames',
            'training_avg_returns',
            'loss'
        ]
        if not self.parallelization:
            metrics += ['num_episodes', 'env_steps', 'loss']

        train_loss = namedtuple('loss', ['loss'])(0.)

        # load the checkpoint
        if os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            self.global_step = tf.compat.v1.train.get_global_step()
            print("Checkpoint loaded! global_step={}".format(self.global_step.numpy()))
        for policy_dir in [self.policy_dir, self._policy_dir]:
            if not os.path.exists(policy_dir):
                os.makedirs(policy_dir)

        def update_progress_bar(num_steps=1):
            if display_progressbar:
                log_values = [
                    ('loss', train_loss.loss),
                    ('replay_buffer_frames', self.replay_buffer.num_frames()),
                    # ('avg_safety_violations_per_episode', self.safety_violations.average()),
                    ('training_avg_returns', self.avg_return.result()),
                ]
                if not self.parallelization:
                    log_values += [
                        ('num_episodes', self.num_episodes.result()),
                        ('env_steps', self.env_steps.result())
                    ]
                progressbar.add(num_steps, log_values)

        if display_progressbar:
            progressbar = Progbar(target=self.num_iterations, interval=display_interval, stateful_metrics=metrics)
        else:
            progressbar = None

        env = self.tf_env if not self.prioritized_experience_replay else self.py_env

        if tf.math.less(self.replay_buffer.num_frames(), self.initial_collect_steps):
            print("Initialize replay buffer...")
            self.initial_collect_driver.run(env.current_time_step())

        print("Start training...")

        update_progress_bar(self.global_step.numpy())

        print("global_step:", self.global_step.numpy(), "\nnum_iterations:", self.num_iterations)

        for step in tf.range(self.global_step, self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.driver.run(env.current_time_step())

            # Use data from the buffer and update the agent's network.
            experience, info = next(self.iterator)
            if self.prioritized_experience_replay:
                priorities = self._compute_priorities(experience)
                self.replay_buffer.update_priorities(keys=info.key[:, 0, ...], priorities=priorities)
                is_weights = tf.cast(
                    tf.stop_gradient(tf.reduce_min(info.probability[:, 0, ...])) / info.probability[:, 0, ...],
                    dtype=tf.float32)

                if tf.reduce_max(priorities) > self.max_priority:
                    self.max_priority.assign(tf.reduce_max(priorities))
            else:
                is_weights = None

            train_loss = self.tf_agent.train(experience, weights=is_weights)

            # step = self.tf_agent.train_step_counter.numpy()

            update_progress_bar()

            if step % self.log_interval == 0:
                self.train_checkpointer.save(self.global_step)
                if self.prioritized_experience_replay:
                    self.replay_buffer.py_client.checkpoint()
                self._policy_saver.save(self._policy_dir)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.loss, step=step)
                    tf.summary.scalar('training average returns', self.avg_return.result(), step=step)

            if step % self.eval_interval == 0:
                eval_thread = threading.Thread(target=self.eval, args=(step, progressbar), daemon=True, name='eval')
                eval_thread.start()

            if self.save_exploration_dataset and step % (self.replay_buffer_capacity // 4) == 0:
                dataset_generation_thread = threading.Thread(target=self.save_observations,
                                                             args=(self.replay_buffer_capacity // 4,),
                                                             daemon=True,
                                                             name='dataset_gen')
                dataset_generation_thread.start()

    def eval(self, step: int = 0, progressbar: Optional = None):
        avg_eval_return = tf_metrics.AverageReturnMetric()
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric()
        saved_policy = tf.compat.v2.saved_model.load(self._policy_dir)
        self.eval_env.reset()
        dynamic_episode_driver.DynamicEpisodeDriver(
            self.eval_env,
            saved_policy,
            [avg_eval_return, avg_eval_episode_length],
            num_episodes=self.num_eval_episodes
        ).run()

        log_values = [
            ('eval_avg_returns', avg_eval_return.result()),
            ('avg_eval_episode_length', avg_eval_episode_length.result()),
        ]
        if progressbar is not None:
            progressbar.add(0, log_values)
        else:
            print('Evaluation')
            for key, value in log_values:
                print(key, '=', value.numpy())
        with self.train_summary_writer.as_default():
            tf.summary.scalar('Average returns', avg_eval_return.result(), step=step)
            tf.summary.scalar('Average episode length', avg_eval_episode_length.result(), step=step)
            #  tf.summary.scalar('Number of safety violations for {} episodes'.format(self.num_eval_episodes),
            #                    num_safety_violations.result(),
            #                    step=step)
        if avg_eval_return.result() >= self.score:
            self.policy_saver.save(self.policy_dir)
            self.score.assign(avg_eval_return.result())

    def save_observations(self, batch_size: int = 128000):
        observations_dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=3)
        dataset_generator.gather_rl_observations(
            iter(observations_dataset),
            #  self.labeling_function,
            dataset_path=os.path.join(self.save_directory_location,
                                      'dataset/reinforcement_learning')
        )
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


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()
    # set seed
    seed = params['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import importlib
        env_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
    except BaseException as err:
        serr = str(err)
        print("Error to load module '" + params['env_suite'] + "': " + serr)
        return -1
    learner = SACLearner(
        env_name=params['env_name'],
        env_suite=env_suite,
        num_iterations=params['steps'],
        num_parallel_environments=params['num_parallel_env'],
        parallelization=params['num_parallel_env'] > 1,
        save_directory_location=params['save_dir'],
        collect_steps_per_iteration=params['collect_steps_per_iteration'],
        batch_size=params['batch_size'],
        reward_scale_factor=params['reward_scale_factor'],
        prioritized_experience_replay=params['prioritized_experience_replay'],
        priority_exponent=params['priority_exponent'],
        gamma=params['gamma'],
        critic_learning_rate=params['learning_rate'],
        actor_learning_rate=params['learning_rate'],
        alpha_learning_rate=params['learning_rate'],
        target_update_tau=params['target_update_tau'],
        replay_buffer_capacity=params['replay_buffer_size'],
        state_perturbation=params['state_perturbation'],
        action_perturbation=params['action_perturbation'],
        seed=params['seed'],
    )

    if not os.path.exists(learner.policy_dir):
        os.makedirs(learner.policy_dir)
    with open(os.path.join(learner.policy_dir, 'params.json'), 'w+') as json_file:
        json.dump(params, json_file)

    if params['permissive_policy_saver']:
        for variance_multiplier in params['variance']:
            learner.save_permissive_variance_policy(variance_multiplier)
        return 0
    learner.train_and_eval()
    return 0


if __name__ == '__main__':
    # app.run(main)
    tf_agents.system.multiprocessing.handle_main(functools.partial(app.run, main))
