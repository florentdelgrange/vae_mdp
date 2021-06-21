from gym.envs.registration import register
from tf_agents.environments.wrappers import HistoryWrapper

register(
    id='LunarLanderNoRewardShaping-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderNoRewardShaping',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderRewardShapingAugmented-v2',
    entry_point='reinforcement_learning.environments.lunar_lander:LunarLanderRewardShapingAugmented',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='PendulumRandomInit-v0',
    entry_point='reinforcement_learning.environments.pendulum:PendulumRandomInit',
    max_episode_steps=150,
)


class EnvironmentLoader:
    def __init__(self, environment_suite, seed=None, time_stacked_states=1):
        self.n = 0
        self.environment_suite = environment_suite
        self.seed = seed
        self.time_stacked_states = time_stacked_states

    def load(self, env_name: str):
        environment = self.environment_suite.load(env_name)
        if self.time_stacked_states > 1:
            environment = HistoryWrapper(env=environment, history_length=self.time_stacked_states)
        if self.seed is not None:
            try:
                environment.seed(self.seed + self.n)
                self.n += 1
            except NotImplementedError:
                print("Environment {} has no seed support.".format(env_name))
        return environment
