from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm


import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers.monitoring import video_recorder

VIDEO_PATH = 'VIDEO/'
CHECKPOINT_PATH = 'CHECKPOINT'
def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 10000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 10000 == 0


device = torch.device("cpu")

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, action_space, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()

        # TODO: implement the rest here

        super(PiApproximationWithNN, self).__init__()
        self.action_space = action_space
        print
        self.model = nn.Sequential(
            nn.Linear(35, 70),
            nn.GELU(),
            nn.Linear(70,35),
            nn.GELU(),
            nn.Linear(35,action_space[0]*action_space[1]),
        ).to(torch.float).to(device)

        self.sftmax = nn.Softmax(dim=1)

        self.alpha = alpha
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=alpha, betas=[0.9,0.99])



    def forward(self, states, return_prob=False):
        states = torch.tensor(states).to(device)
        states = states.flatten()
        #print(type(states))
        #print(states.shape)
        #print(states)
        out = self.model(states)
        out = out.reshape(self.action_space[1], self.action_space[0])
        out = self.sftmax(out)
        if return_prob:
            return out
        action = torch.distributions.Categorical(out).sample()
        return action.cpu().numpy()


    def update(self, state, actions, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optimizer.zero_grad()
        policy = self(state, return_prob=True)
        sum_prob = 0
        for i in range(len(policy)):
            sum_prob += torch.log(policy[i][actions[i]])

        loss = -delta  * sum_prob
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
            nn.Linear(35, 70),
            nn.ReLU(),
            nn.Linear(70,35),
            nn.ReLU(),
            nn.Linear(35,1)
        ).to(torch.float32)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=alpha)

    def forward(self, states) -> float:
        # TODO: implement this method
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).to(torch.float32)
        return self.model(states)
    def update(self, states, G):
        # TODO: implement this method
        states = torch.tensor(states).to(torch.float32)
        self.optimizer.zero_grad()
        loss = (G - self(states))**2
        loss.backward()
        self.optimizer.step()

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
        adv = 0
        # Generate an episode
        done = False
        loss = 0
        state,_ = env.reset()
        states = []
        actions = []
        rewards = []
        # Get the trajectory of the episode
        while not done:
            action = pi(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = new_state

        for t in range(len(states)):
            # calculate all returns
            G = sum(rewards[t:])
            if t == 0:
                Gs.append(G)
            delta = G - V(states[t])
            adv += delta

            loss += pi.update(states[t],actions[t], gamma**t, delta)
            V.update(states[t], G)
        wandb.log(
            {'return': Gs[i],
             'Advantage': adv/len(states),
             'Loss': loss,
             'reward': sum(rewards)/len(rewards)}
        )

        if capped_cubic_video_schedule(i):
            torch.save(V.state_dict(), f'{CHECKPOINT_PATH}/VALUE/{i}.pth')
            torch.save(pi.state_dict(), f'{CHECKPOINT_PATH}/POLICY/{i}.pth')
    return Gs
