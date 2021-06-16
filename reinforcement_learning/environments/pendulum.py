import numpy as np
from gym.envs.classic_control.pendulum import PendulumEnv


class PendulumRightInit(PendulumEnv):
    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=0)
        self.last_u = None
        return self._get_obs()
