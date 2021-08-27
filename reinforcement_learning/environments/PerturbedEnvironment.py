from typing import Any, Optional

from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.typing import types
from tf_agents.trajectories import time_step as ts
import tensorflow_probability as tfp
tfd = tfp.distributions


class PerturbedEnvironment(PyEnvironmentBaseWrapper):

    def __init__(self, env: Any, state_noise: Optional[float] = 0., action_noise: Optional[float] = 0):
        super().__init__(env)
        self.state_noise = state_noise
        self.action_noise = action_noise

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self.action_noise > 0:
            _action = tfd.MultivariateNormalDiag(loc=action, scale_identity_multiplier=self.action_noise).sample()
        else:
            _action = action

        time_step = self.wrapped_env().step(_action)

        if self.state_noise > 0:
            _observation = tfd.MultivariateNormalDiag(
                loc=time_step.observation, scale_identity_multiplier=self.state_noise).sample()
        else:
            _observation = time_step.observation

        return time_step._replace(observation=_observation)
