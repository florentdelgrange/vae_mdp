from typing import Any, Optional

from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.typing import types
from tf_agents.trajectories import time_step as ts
import numpy as np


class PerturbedEnvironment(PyEnvironmentBaseWrapper):

    def __init__(self, env: Any, state_noise: Optional[float] = 0., action_noise: Optional[float] = 0):
        super().__init__(env)
        self.state_noise = state_noise
        self.action_noise = action_noise

    def step(self, action: types.NestedArray) -> ts.TimeStep:
        if self.action_noise > 0:
            _action = np.random.multivariate_normal(
                mean=action,
                cov=np.diag(self.action_noise ** 2 * np.ones(shape=np.shape(action))))
        else:
            _action = action

        time_step = self.wrapped_env().step(_action)

        if self.state_noise > 0:
            _observation = np.random.multivariate_normal(
                mean=time_step.observation,
                cov=np.diag(self.state_noise ** 2 * np.ones(shape=np.shape(time_step.observation))))
        else:
            _observation = time_step.observation

        return time_step._replace(observation=_observation)
