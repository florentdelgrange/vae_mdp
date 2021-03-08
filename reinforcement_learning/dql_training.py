import os
import sys
from typing import Tuple, Callable, Optional
import threading
import datetime
from absl import app
from absl import flags

import PIL
import imageio

import tensorflow as tf
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.utils.generic_utils import Progbar
from tf_agents.agents.dqn import dqn_agent

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network

from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.metrics import tf_metrics, tf_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import tf_agents.trajectories.time_step as ts


path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')
from reinforcement_learning import labeling_functions
from util.io import dataset_generator

flags.DEFINE_string(
    'env_name', help='Name of the environment', default='CartPole-v0'
)
flags.DEFINE_string(
    'env_suite', help='Environment suite', default='suite_gym'
)
flags.DEFINE_integer(
    'steps', help='Number of iterations', default=int(2e5)
)
flags.DEFINE_integer(
    'num_parallel_env', help='Number of parallel environments', default=1
)
FLAGS = flags.FLAGS


class DQNLearner:
    def __init__(self,
                 env_name: str,
                 env_suite,
                 num_iterations: int = int(2e5),
                 initial_collect_steps: int = int(1e4),
                 collect_steps_per_iteration: int = 1,
                 replay_buffer_capacity: int = int(1e6),
                 network_fc_layer_params: Tuple[int, ...] = (100, 50),
                 gamma: float = 0.99,
                 learning_rate: float = 1e-3,
                 log_interval: int = 2500,
                 num_eval_episodes: int = 30,
                 eval_interval: int = int(1e4),
                 parallelization: bool = True,
                 num_parallel_environments: int = 4,
                 batch_size: int = 64,
                 debug: bool = False,
                 save_directory_location: str = '.'):

        if collect_steps_per_iteration is None:
            collect_steps_per_iteration = batch_size
        if parallelization:
            replay_buffer_capacity = replay_buffer_capacity // num_parallel_environments
            collect_steps_per_iteration = max(1, collect_steps_per_iteration // num_parallel_environments)

        self.env_name = env_name
        self.num_iterations = num_iterations

        self.initial_collect_steps = initial_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.log_interval = log_interval

        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval

        self.parallelization = parallelization
        self.num_parallel_environments = num_parallel_environments

        self.batch_size = batch_size

        if parallelization:
            self.tf_env = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(
                [lambda: env_suite.load(env_name)] * num_parallel_environments))
            self.tf_env.reset()
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.eval_env = tf_py_environment.TFPyEnvironment(self.py_env)
        else:
            self.py_env = env_suite.load(env_name)
            self.py_env.reset()
            if debug:
                img = PIL.Image.fromarray(self.py_env.render())
                img.show()
            self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
            self.eval_env = self.tf_env

        self.observation_spec = self.tf_env.observation_spec()
        self.action_spec = self.tf_env.action_spec()

        num_actions = self.action_spec.maximum - self.action_spec.minimum + 1

        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal'))

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # it's output.
        dense_layers = [dense_layer(num_units) for num_units in network_fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.tf_agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
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
            num_parallel_calls=num_parallel_environments,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

        self.checkpoint_dir = os.path.join(save_directory_location, 'saves', env_name, 'sac_training_checkpoint')
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.tf_agent,
            policy=self.collect_policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
        )
        self.stochastic_policy_dir = os.path.join(save_directory_location, 'saves', env_name, 'policy')
        self.stochastic_policy_saver = policy_saver.PolicySaver(self.tf_agent.policy)

        self.num_episodes = tf_metrics.NumberOfEpisodes()
        self.env_steps = tf_metrics.EnvironmentSteps()
        self.avg_return = tf_metrics.AverageReturnMetric(batch_size=self.tf_env.batch_size)

        observers = [self.num_episodes, self.env_steps] if not parallelization else []
        observers += [self.avg_return, self.replay_buffer.add_batch]
        # A driver executes the agent's exploration loop and allows the observers to collect exploration information
        self.driver = dynamic_step_driver.DynamicStepDriver(
            self.tf_env, self.collect_policy, observers=observers, num_steps=collect_steps_per_iteration)
        self.initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.tf_env, self.collect_policy, observers=[self.replay_buffer.add_batch], num_steps=initial_collect_steps)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(
            save_directory_location, 'logs', 'gradient_tape', env_name, 'sac_agent_training', current_time)
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_directory_location = os.path.join(save_directory_location, 'saves', env_name)

    def train_and_eval(self, display_progressbar: bool = True, display_interval: float = 0.1):

        # Optimize by wrapping some of the code in a graph using TF function.
        self.tf_agent.train = common.function(self.tf_agent.train)
        self.driver.run = common.function(self.driver.run)

        metrics = [
            'eval_avg_returns',
            'avg_eval_episode_length',
            'replay_buffer_frames',
            'training_avg_returns'
        ]
        if not self.parallelization:
            metrics += ['num_episodes', 'env_steps']

        class AbstractLoss:
            def __init__(self):
                self.loss = 0.

        train_loss = AbstractLoss()

        # load the checkpoint
        if os.path.exists(self.checkpoint_dir):
            self.train_checkpointer.initialize_or_restore()
            self.global_step = tf.compat.v1.train.get_global_step()
            print("Checkpoint loaded! global_step={}".format(self.global_step.numpy()))
        if not os.path.exists(self.stochastic_policy_dir):
            os.makedirs(self.stochastic_policy_dir)

        def update_progress_bar(num_steps=1):
            if display_progressbar:
                log_values = [
                    ('loss', train_loss.loss),
                    ('replay_buffer_frames', self.replay_buffer.num_frames()),
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

        print("Initialize replay buffer...")
        self.initial_collect_driver.run()

        print("Start training...")

        update_progress_bar(self.global_step.numpy())

        for _ in range(self.global_step.numpy(), self.num_iterations):

            # Collect a few steps using collect_policy and save to the replay buffer.
            self.driver.run()

            # Use data from the buffer and update the agent's network.
            # experience = replay_buffer.gather_all()
            experience, _ = next(self.iterator)
            train_loss = self.tf_agent.train(experience)

            step = self.tf_agent.train_step_counter.numpy()

            update_progress_bar()

            if step % self.log_interval == 0:
                self.train_checkpointer.save(self.global_step)
                self.stochastic_policy_saver.save(self.stochastic_policy_dir)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.loss, step=step)
                    tf.summary.scalar('training average returns', self.avg_return.result(), step=step)

            if step % self.eval_interval == 0:
                eval_thread = threading.Thread(target=self.eval, args=(step, progressbar), daemon=True, name='eval')
                eval_thread.start()

    def eval(self, step: int = 0, progressbar: Optional = None):
        avg_eval_return = tf_metrics.AverageReturnMetric()
        avg_eval_episode_length = tf_metrics.AverageEpisodeLengthMetric()
        saved_policy = tf.compat.v2.saved_model.load(self.stochastic_policy_dir)
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


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()
    tf.random.set_seed(params['seed'])
    try:
        import importlib
        env_suite = importlib.import_module('tf_agents.environments.' + params['env_suite'])
    except BaseException as err:
        serr = str(err)
        print("Error to load module '" + params['env_suite'] + "': " + serr)
        return -1
    learner = DQNLearner(
        env_name=params['env_name'],
        env_suite=env_suite,
        num_iterations=params['steps'],
        num_parallel_environments=params['num_parallel_env'],
        save_directory_location=params['save_dir'],
    )
    learner.train_and_eval()
    return 0


if __name__ == '__main__':
    app.run(main)
