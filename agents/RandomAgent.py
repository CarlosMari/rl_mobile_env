from agents.agent import Agent
from random import randint
import  numpy as np

class RandomAgent(Agent):
    def __init__(self, obs_dims, action_dims, alpha):
        super().__init__(obs_dims, action_dims, alpha)

    def __call__(self, _):
        actions = np.array([randint(0,3) for _ in range(5)])
        return actions