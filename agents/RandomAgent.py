from agents.agent import Agent
from random import randint
import  numpy as np

class RandomAgent(Agent):
    def __init__(self, obs_dims, action_dims, alpha):
        super().__init__(obs_dims, action_dims, alpha)

    def __call__(self, observations):
        actions = []
        for obs in observations:
            action = randint(0,3)
            # To not give the random baseline an advantage we disconnect, this approach only works 
            # in small models
            if action != 0 and sum(obs[0:3]) > 0:
                conn_bs = obs[0:3].tolist().index(1)
                actions.append(conn_bs+1)
            else:
                 actions.append(action)

        actions = np.array([randint(0,3) for _ in range(5)])
        return actions