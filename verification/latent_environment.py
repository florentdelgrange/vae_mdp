from typing import Callable
import tensorflow as tf
from tf_agents import trajectories, specs
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies import tf_policy


class LatentPolicyOverRealStateSpace(tf_policy.TFPolicy):

    def __init__(self,
                 time_step_spec,
                 labeling_function: Callable[[tf.Tensor], tf.Tensor],
                 latent_policy: tf_policy.TFPolicy,
                 state_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]):
        super().__init__(time_step_spec, latent_policy.action_spec)
        self.labeling_function = labeling_function
        self.wrapped_policy = latent_policy
        self.state_embedding_function = state_embedding_function

    def _distribution(self, time_step, policy_state):
        latent_state = self.state_embedding_function(
            time_step.observation, self.labeling_function(time_step.observation))
        _time_step = trajectories.time_step.TimeStep(
            time_step.step_type, time_step.reward, time_step.discount, latent_state)
        return self.wrapped_policy._distribution(_time_step, policy_state)


class DiscreteActionTFEnvironmentWrapper(TFEnvironment):
    def __init__(self,
                 tf_env: TFEnvironment,
                 action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 number_of_discrete_actions: int):
        super(DiscreteActionTFEnvironmentWrapper, self).__init__(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=specs.BoundedTensorSpec(
                shape=(),
                dtype=tf.int32,
                minimum=0,
                maximum=number_of_discrete_actions - 1,
                name='latent_action'),
            batch_size=tf_env.batch_size)
        self.wrapped_env: TFEnvironment = tf_env
        self.action_embedding_function = action_embedding_function

    def _current_time_step(self):
        self.wrapped_env.current_time_step()

    def _reset(self):
        self.wrapped_env.reset()

    def _step(self, latent_action):
        real_action = self.action_embedding_function(self.current_time_step().observation, latent_action)
        self.wrapped_env.step(real_action)

    def render(self):
        self.wrapped_env.render()
