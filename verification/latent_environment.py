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
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=latent_policy.action_spec,
            info_spec=latent_policy.info_spec,
            policy_state_spec=latent_policy.policy_state_spec)
        self._labeling_function = labeling_function
        self.wrapped_policy = latent_policy
        self.state_embedding_function = state_embedding_function

    def labeling_function(self, state: tf.Tensor):
        label = tf.cast(self._labeling_function(state), dtype=tf.float32)
        # take the reset label into account
        label = tf.cond(
            tf.rank(label) == 1,
            lambda: tf.expand_dims(label, axis=-1),
            lambda: label)
        return tf.concat(
            [label, tf.zeros(shape=tf.concat([tf.shape(label)[:-1], tf.constant([1], dtype=tf.int32)], axis=-1),
                             dtype=tf.float32)],
            axis=-1)

    def _distribution(self, time_step, policy_state):
        latent_state = self.state_embedding_function(
            time_step.observation, self.labeling_function(time_step.observation))
        _time_step = trajectories.time_step.TimeStep(
            step_type=time_step.step_type,
            reward=time_step.reward,
            discount=time_step.discount,
            observation=latent_state)
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
