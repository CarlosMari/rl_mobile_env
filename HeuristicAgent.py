from typing import Any
from agent import Agent
import numpy as np

class HeuristicAgentSNR(Agent):
    def __init__(self, obs_dims, action_dims, alpha):
        super().__init__(obs_dims, action_dims, alpha)

    def __call__(self, observations) -> list:
        # Receives a observation in the shape [NUM_AGENTS, OBS_DIMS]
        # Returns a np.array shape NUM_AGENTS
        # We need to set a threshold for which we decide to not be connected to anything
        actions = []
        for obs in observations:
            # Base Stations SNR
            action = np.argmax(obs[3:6])
            # If we are connected to the best bs we do nothing
            if obs[action]:
                actions.append(0)

            # If connected to bs we disconnect
            elif sum(obs[0:3]) > 0:
                    conn_bs = obs[0:3].tolist().index(1)
                    actions.append(conn_bs+1)
            # If we are not connected we connect to the best bs
            else:
                actions.append(action + 1)
        return actions

class HeuristicAgentUtility(Agent):
    def __init__(self, obs_dims, action_dims, alpha):
        super().__init__(obs_dims, action_dims, alpha)

    def __call__(self, observations) -> list:
        # Receives a observation in the shape [NUM_AGENTS, OBS_DIMS]
        # Returns a np.array shape NUM_AGENTS
        # We need to set a threshold for which we decide to not be connected to anything
        actions = []
        for obs in observations:
            # Base station utility
            action = np.argmax(obs[7:10])
            # If we are connected to the best bs we do nothing
            if obs[action]:
                actions.append(0)

            # If connected to bs we disconnect
            elif sum(obs[0:3]) > 0:
                    conn_bs = obs[0:3].tolist().index(1)
                    actions.append(conn_bs+1)
            # If we are not connected we connect to the best bs
            else:
                actions.append(action + 1)
        return actions