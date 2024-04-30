import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NNAgent(nn.Module):
    def __init__(self, state_dims, action_space, alpha, DEVICE='cpu'):
            """
            state_dims: the number of dimensions of state space
            action_dims: the number of possible actions
            alpha: learning rate
            """
            super(NNAgent, self).__init__()
            self.action_space = action_space
            self.DEVICE = DEVICE
            self.model = nn.Sequential(
                nn.Linear(state_dims, 32),
                nn.LeakyReLU(),
                nn.LayerNorm(32),
                nn.Linear(32, 64),
                nn.LeakyReLU(),
                nn.LayerNorm(64),
                nn.Linear(64,32),
                nn.LeakyReLU(),
                nn.LayerNorm(32),
                nn.Linear(32,action_space)
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

        actions = torch.from_numpy(actions).to(self.DEVICE)
        # policy shape -> (5,4)
        probabilities = self(state, return_prob=True)
        one_hot = F.one_hot(actions, self.action_space)
        prob_taken = one_hot * probabilities
        prob_taken_reduced = torch.log(torch.sum(prob_taken, dim=1))

        loss = torch.from_numpy(-delta).float()  * prob_taken_reduced
        loss =  torch.sum(loss)/len(loss)
        loss.backward()
        self.optimizer.step()
        return loss.cpu()


class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha, DEVICE='cpu'):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        self.state_dims = state_dims
        
        self.model = nn.Sequential(
            nn.Linear(state_dims, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        ).to(torch.float32).to(DEVICE)
        self.DEVICE = DEVICE
        self.to(DEVICE)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=alpha)
        self.loss = torch.nn.MSELoss()

    def forward(self, states, train=True) -> float:
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).to(torch.float32).to(self.DEVICE)
        if train:
            return self.model(states).squeeze(1).detach().cpu().numpy()
        
        return self.model(states)

    def update(self, states, G):
        states = torch.tensor(states).to(torch.float32).to(self.DEVICE)
        self.optimizer.zero_grad()
        G = torch.from_numpy(G).to(torch.float).to(self.DEVICE)
        loss = self.loss(self(states, train=False).squeeze(1), G)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()