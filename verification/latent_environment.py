from typing import Callable
import tensorflow as tf
from tf_agents import specs
from tf_agents.environments.tf_environment import TFEnvironment


class DiscreteActionTFEnvironmentWrapper(TFEnvironment):
    def __init__(self,
                 tf_env: TFEnvironment,
                 action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 number_of_discrete_actions: int):
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

    def _current_time_step(self):
        return self.wrapped_env.current_time_step()

    def _reset(self):
        return self.wrapped_env.reset()

    def _step(self, latent_action):
        real_action = self.action_embedding_function(self.current_time_step().observation, latent_action)
        return self.wrapped_env.step(real_action)

    def render(self):
        return self.wrapped_env.render()
