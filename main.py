import sys
import numpy as np
import gymnasium as gym
import mobile_env 
from reinforce import REINFORCE, Baseline
from agents.NNAgent import VApproximationWithNN, NNAgent 
from agents.HeuristicAgent import HeuristicAgentSNR, HeuristicAgentUtility, ProductRule
from agents.RandomAgent import RandomAgent

import wandb
import torch
import argparse

parser = argparse.ArgumentParser(prog='Mobile-ENV')
parser.add_argument('--agent', choices=['reinforce','utility','snr','random', 'custom'], default='utility')
parser.add_argument('--baseline', type=bool, default=True)
parser.add_argument('--episodes', type=int, default=10000)
parser.add_argument('--name', type=str, default='Test')
parser.add_argument('--num_iter', type=int, default=1)
parser.add_argument('--log', type=int, default=1)
parser.add_argument('--wrapper', type=int, default=1)

args = parser.parse_args()
AGENT = args.agent
BASELINE = args.baseline
EPISODES = args.episodes
NAME = args.name
NUM_ITER = args.num_iter
LOG = args.log == 1
WRAPPER = args.wrapper == 1

print(f'LOGGING IS SET TO {LOG}')
DEVICE = torch.device('cpu')

def test_reinforce():
    env = gym.make("mobile-small-ma-v0", render_mode = 'human')
    #actions_shape = (env.NUM_STATIONS+1, env.NUM_USERS)
    actions = 2* env.NUM_STATIONS + 1
    
    gamma = 1.
    lr = 3e-4
    if AGENT == 'reinforce':
        pi = NNAgent(6,actions,lr).to(DEVICE)
        
    elif AGENT == 'utility':
        pi = HeuristicAgentUtility(13,0,0)

    elif AGENT == 'snr':
        pi = HeuristicAgentSNR(13,0,0)

    elif AGENT == 'random':
        pi = RandomAgent(13,0,0)

    elif AGENT == 'custom':
        pi = ProductRule(13,0,0)

    if BASELINE:
        B = VApproximationWithNN(
            6,
            lr).to(DEVICE)
    else:
        B = Baseline(0.)

    return REINFORCE(env,gamma,EPISODES,pi,B,LOG,WRAPPER)

if __name__ == "__main__":
    if LOG:
        wandb.init(project='MOBILE-ENV',name = NAME)

    for _ in range(NUM_ITER):
        training_progress = test_reinforce()


