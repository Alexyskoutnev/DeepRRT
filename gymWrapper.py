import numpy as np
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class RelativePosition(ObservationWrapper):
    def __init__(self, env, lower_bound, upper_bound):
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=lower_bound, high=upper_bound)

    def observation(self, obs):
        return obs["target"] - obs["agent"]