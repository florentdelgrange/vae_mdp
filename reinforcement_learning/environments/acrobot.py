import numpy as np
from gym.envs.classic_control.acrobot import AcrobotEnv


class AcrobotEnvRandomInit(AcrobotEnv):
    def reset(self):
        high = np.array([np.pi / 3, np.pi / 3, self.MAX_VEL_1 / 12., self.MAX_VEL_2])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_ob()
