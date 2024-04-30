from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from collections import OrderedDict

import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers.monitoring import video_recorder

from agents.NNAgent import VApproximationWithNN, NNAgent

VIDEO_PATH = 'VIDEO/'
CHECKPOINT_PATH = 'CHECKPOINT'
DEVICE = torch.device('cpu')
def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 100000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 100000 == 0


device = torch.device("cpu")


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    There is no need to change this class.
    """
    def __init__(self,b):
        self.b = b
        
    def __call__(self, states):
        return self.forward(states)
        
    def forward(self, states) -> float:
        return self.b

    def update(self, states, G):
        pass


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi,
    V,
    LOG) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    Gs = []
    for i in tqdm(range(num_episodes)):
        # Generate an episode
        done = False
        loss = []
        state,_ = env.reset()
        states = []
        actions = []
        rewards = []
        # Get the trajectory of the episode
        while not done:
            #env.render()
            #state = np.array([np.append(array[0:6],array[10:13]) for array in state.values()])
            state = np.array([array[0:6] for array in state.values()])
            #state = np.array([array[0:10] for array in state.values()])
            action = pi(state)
            parsed_actions = action_handler(action, state)
            ordered_actions = OrderedDict([(i, parsed_actions[i]) for i in range(len(parsed_actions))])
            new_state, reward, terminated, truncated, _ = env.step(ordered_actions)
            reward = np.array([array for array in reward.values()])
            done = terminated or truncated 
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = new_state
        #env.close()

        for t in range(len(states)):
            # calculate all returns
            G = np.sum(np.array(rewards)[t:,:], axis=0)
            if t == 0:
                Gs.append(G)
            value = V(states[t])
            delta = G - value
            loss.append(pi.update(states[t],actions[t], gamma**t, delta))
            V.update(states[t],G)
        if LOG:
            wandb.log(
                {'return': np.mean(Gs[i]),
                'reward_max': np.max(rewards),
                'reward_mean': np.mean(rewards),
                'reward_min': np.min(rewards),
            })

        if capped_cubic_video_schedule(i) and False:
            torch.save(V.state_dict(), f'{CHECKPOINT_PATH}/VALUE/{i}.pth')
            torch.save(pi.state_dict(), f'{CHECKPOINT_PATH}/POLICY/{i}.pth')
    return Gs


def action_handler(actions, observation):
    new_actions = []
    for i in range(len(actions)):
        obs = observation[i]
        if actions[i] == 0:
            new_actions.append(0)

        elif actions[i] < 4:
            if obs[actions[i]-1]:
                new_actions.append(0)

            else:
                new_actions.append(actions[i])

        else:
            if obs[actions[i]-3-1]:
                new_actions.append(actions[i]-3)

            else:
                new_actions.append(0)
    return new_actions



        
