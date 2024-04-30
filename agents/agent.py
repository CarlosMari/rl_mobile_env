from abc import ABC, abstractmethod

class Agent():
    def __init__(self, obs_dims, action_dims, alpha):
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.alpha = alpha

    # Dummy Update for Heuristic methods
    def update(self, *kwargs):
        return 0