import os

import imageio
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories.time_step import TimeStep


class VideoEmbeddingObserver:
    def __init__(self, py_env: PyEnvironment, file_name: str, fps: int = 30, num_episodes: int = 1):
        self.py_env = py_env
        self.file_name = file_name
        self.writer = None
        self.fps = fps
        self.best_rewards = -1. * np.inf
        self.cumulative_rewards = 0.
        self.num_episodes = num_episodes
        self.current_episode = 1

    def __call__(self, time_step: TimeStep, *args, **kwargs):
        if self.writer is None:
            self.writer = imageio.get_writer('{}.mp4'.format(self.file_name), fps=self.fps)
        if not time_step.is_last():
            self.writer.append_data(self.py_env.render(mode='rgb_array'))
            self.cumulative_rewards += time_step.reward
        elif self.current_episode < self.num_episodes:
            self.current_episode += 1
        else:
            self.writer.close()
            self.writer = None
            avg_rewards = np.sum(self.cumulative_rewards / self.num_episodes)
            if avg_rewards >= self.best_rewards:
                self.best_rewards = avg_rewards
                os.rename(self.file_name, '{}_rewards={:.2f}.mp4'.format(self.file_name, self.best_rewards))
            else:
                os.remove('{}.mp4'.format(self.file_name))
            self.cumulative_rewards = 0.
            self.current_episode = 1