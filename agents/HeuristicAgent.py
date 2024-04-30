from typing import Any
from agents.agent import Agent
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
        print(actions)
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
            # Observation shape
            # (0,1,2) whether we are connected
            # (3,4,5) SNR to base station
            # (6) Utility
            # (7,8,9) Utility of given bs
            # (10,11,12) How many devices connected to each bs
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
    

class ProductRule(Agent):
    def __init__(self, obs_dims, action_dims, alpha):
        super().__init__(obs_dims, action_dims, alpha)

    def __call__(self, observations) -> Any:
         actions = []
         for obs in observations:
            snr = np.array(obs[3:6])
            utility = np.array(obs[7:10])
            product = snr * (-utility)
            action = np.argmax(product)

            if obs[action]:
                actions.append(0)

            elif sum(obs[0:3]) > 0:
                    conn_bs = obs[0:3].tolist().index(1)
                    actions.append(conn_bs+1)
            else:
                actions.append(action + 1)

         return actions