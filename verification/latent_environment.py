from typing import Callable, Optional
import tensorflow as tf
from tf_agents import specs
from tf_agents.environments.tf_environment import TFEnvironment


class DiscreteActionTFEnvironmentWrapper(TFEnvironment):
    def __init__(self,
                 tf_env: TFEnvironment,
                 action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 number_of_discrete_actions: int,
                 reward_scaling: Optional[float] = 1.):
        super(DiscreteActionTFEnvironmentWrapper, self).__init__(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=specs.BoundedTensorSpec(
                shape=(),
                dtype=tf.int64,
                minimum=0,
                maximum=number_of_discrete_actions - 1,
                name='latent_action'),
            batch_size=tf_env.batch_size)
        self.wrapped_env: TFEnvironment = tf_env
        self.action_embedding_function = action_embedding_function
        self.reward_scaling = reward_scaling

    def _current_time_step(self):
        ts = self.wrapped_env.current_time_step()
        return ts._replace(reward=self.reward_scaling * ts.reward)

    def _reset(self):
        ts = self.wrapped_env.reset()
        return ts._replace(reward=self.reward_scaling * ts.reward)

    def _step(self, latent_action):
        real_action = self.action_embedding_function(self.current_time_step().observation, latent_action)
        ts = self.wrapped_env.step(real_action)
        return ts._replace(reward=self.reward_scaling * ts.reward)

    def render(self):
        return self.wrapped_env.render()
