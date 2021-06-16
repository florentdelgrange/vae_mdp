import numpy as np
from gym.envs.box2d import LunarLander
from gym.vector.utils import spaces


class LunarLanderNoRewardShaping(LunarLander):

    def step(self, action):
        prev_shaping = self.prev_shaping
        state, reward, done, d = super().step(action)
        shaping = self.prev_shaping
        if prev_shaping is not None:
            reward += prev_shaping - shaping
        return state, reward, done, d


class LunarLanderRewardShapingAugmented(LunarLander):

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

    def step(self, action):
        state, reward, done, d = super().step(action)
        state = np.append(state, [self.prev_shaping], axis=-1)
        return state, reward, done, d
