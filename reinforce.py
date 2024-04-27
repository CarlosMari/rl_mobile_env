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

class PIGreedy():
    def __init__(self,state_dims, action_space, _) -> None:
        self.state_dims = 13
        self.actions_shape = action_space
        #self.num_stations = action_space[0]-1
        #self.num_users = action_space[1]
        pass
    def __call__(self, obs):
        print('SNRS: ')
        obs_per_user = 2 * self.num_stations + 1
        actions = []
        for ue in range(self.num_users):
            offset = ue * obs_per_user
            snrs = np.array(obs[offset+self.num_stations:offset+2*self.num_stations])
            action = np.argmax(snrs)
            actions.append(action)
            print(f'SNRS : {snrs} \n action: {action}')



        return np.array(actions)
    
    def update(self, state, actions, gamma_t, delta):
        return 0

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, action_space, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()
        self.action_space = action_space

        # TODO: implement the rest here

        super(PiApproximationWithNN, self).__init__()
        self.action_space = action_space
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128,action_space)
        ).to(torch.double).to(DEVICE)

        self.sftmax = nn.Softmax()

        self.alpha = alpha
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=alpha, betas=[0.9,0.99])
        self.to(DEVICE)



    def forward(self, states, return_prob=False):
        # staes.shape -> (5,13)
        states = torch.tensor(states).double()
        out = self.model(states)
        out_prob = self.sftmax(out)
        if return_prob:
            return out_prob
        
        '''for dim in range(len(out)):
            if any(torch.isnan(out[dim])):
                print(f'State: {states[dim]}')
                print(f'Activation: {out[dim]}')
                print(f'Probs: {out_prob[dim]}')'''
        action = torch.distributions.Categorical(out_prob).sample()
        return action.cpu().numpy()


    def update(self, state, actions, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # delta -> (5,1)
        # states_shape -> (5,13)
        # actions -> (5, 1) -> (5,4)
        self.optimizer.zero_grad()

        actions = torch.from_numpy(actions).to(DEVICE)
        # policy shape -> (5,4)
        probabilities = self(state, return_prob=True)
        one_hot = F.one_hot(actions, 4)
        prob_taken = one_hot * probabilities
        prob_taken_reduced = torch.log(torch.sum(prob_taken, dim=1))

        loss = torch.from_numpy(-delta).float()  * prob_taken_reduced
        loss =  torch.sum(loss)/len(loss)
        loss.backward()
        self.optimizer.step()
        return loss.cpu()


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

class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        self.state_dims = state_dims
        
        # TODO: implement the rest here
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        ).to(torch.float32).to(DEVICE)

        self.to(DEVICE)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=alpha)
        self.loss = torch.nn.MSELoss()

    def forward(self, states) -> float:
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).to(torch.float32).to(DEVICE)
        return self.model(states)
    

    def update(self, states, G):

        states = torch.tensor(states).to(torch.float32).to(DEVICE)
        self.optimizer.zero_grad()
        G = torch.from_numpy(G).to(torch.float).to(DEVICE)
        loss = self.loss(self(states).squeeze(1), G)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()

def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:VApproximationWithNN) -> Iterable[float]:
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
        value_loss = []
        state,_ = env.reset()
        states = []
        actions = []
        rewards = []
        # Get the trajectory of the episode
        while not done:
            #env.render()
            state = np.array([array for array in state.values()])
            action = pi(state)
            ordered_actions = OrderedDict([(i, action[i]) for i in range(len(action))])
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
            value = V(states[t]).squeeze(1).detach().cpu().numpy()
            delta = G - value
            loss.append(pi.update(states[t],actions[t], gamma**t, delta))
            value_loss.append(V.update(states[t], G).cpu().detach())
        wandb.log(
            {'return': np.mean(Gs[i]),
             #'v_loss': np.mean(value_loss)}
        })

        if capped_cubic_video_schedule(i) and False:
            torch.save(V.state_dict(), f'{CHECKPOINT_PATH}/VALUE/{i}.pth')
            torch.save(pi.state_dict(), f'{CHECKPOINT_PATH}/POLICY/{i}.pth')
    return Gs
