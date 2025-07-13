import argparse
import random
from typing import Deque, Optional, Sequence, Tuple, NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple, deque
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.pi = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.v = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128 ,1)
        )
    
    def step(self, obs):
        '''
        returns 3 things: actions, values, log probabilities of actions
        dims:
        (batch,), (batch,) (batch,)
        actions is (batch,) because it is a discrete state space (so batch x scalar = batch,)
        '''
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            
            logits = self.pi(obs) # [batch]
            dist = Categorical(logits=logits)
            action = dist.sample() # [batch]
            logp = dist.log_prob(action) # [batch]
            value = self.v(obs).squeeze(-1) # [batch] (was [batch, 1] before .squeeze(-1))
            
            return action.cpu().numpy(), value.cpu().numpy(), logp.cpu().numpy()

    def act(self, obs):
        '''
        Same as step, just returns the actions
        '''
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            
            logits = self.pi(obs)
            dist = Categorical(logits=logits)
            
            return dist.sample().cpu().numpy()
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        return self.pi(obs), self.v(obs).squeeze(-1)

class TrajectoryStep(NamedTuple):
    obs: np.ndarray
    action: int
    logp: float
    reward: float
    value: float
    done: bool

class RolloutBuffer:
    buffer: Deque[Tuple[TrajectoryStep, float, float]] # state, action, prob, reward, vals, done, advantage, total_returns
    # state, action, prob, reward, vals, done, advantage, total_return
    
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add(self, TrajectoryStep, advantage: Optional[float], total_return: Optional[float]) -> None:
        self.buffer.append((TrajectoryStep, advantage, total_return))
    
    def sample(self, batch_size: int) -> Tuple[TrajectoryStep, float, float]:
        batch = random.sample(self.buffer, batch_size)
        TrajectoryStep, advantages, returns = map(list, zip(*batch))
        return TrajectoryStep, advantages, returns # returns a tuple

    def __len__(self) -> int:
        return len(self.buffer)
    
    def calculate_advantage(self, gamma, GAEbias):
        episode = []
        start_idx = 0
        for i in range(len(self.buffer)):
            if self.buffer[i][0].done:
                end_idx = i
                calculate_episode_advantage(self.buffer[start_idx:end_idx+1], gamma, GAEbias = GAEbias)
    
    def calculate_episode_advantage(self, episode, gamma: float, GAEbias: float = 1):
        deltas = []
        for t in range(len(episode)-1):
            deltas.append(episode[t][0].reward + gamma * episode[t+1][0].value - episode[t][0].value)
        
        advantages = []
        advantage = 0
        discountFactor = 1
        discountProduct = gamma * GAEbias
        for i in range(len(deltas)):
            advantage += discountFactor * deltas[i]
            discountFactor *= discountProduct # avoid expensive exponentiation
        advantages.append(advantage)
        # now for each sequential one, we subtract that delta and divide by the discount
        correction = gamma * GAEbias
        for i in range(1, len(deltas)):
            advantage -= deltas[i]
            advantage /= correction
            advantages.append(advantage)
        return advantages
            
            
            
'''
Let's break down what I need for PPO into steps:
The loss function is 3 separate loss functions:
L^CLIP, L^VF, L^S

S is the entropy.

VF I need:
-predicted values
-empirical values (V_targ) (FIND OUT HOW TO FIND THESE)

CLIP I need:
-Advantage estimates (USE GAE)
-The policy ratio (??)

How to find the policy ratio (r):
While the policy gets run, we store the log probabilities (dist.log_prob)
Then during optimization we can estimate the new log probabilities of those actions in those states
Exponentiate that, and that's r.

How to find the advantage (A):
Store the values at all states.
Compute the TDs, then just use the formula.

How to find the values (v_targ):
V_targ = A^ + V(s_t) (from PPO paper)
So if we know the values and compute the advantages, we're golden.

Total info I need to store:
-Rewards at each time step
-Values at each time step
-States at each time step
-Action taken at each time step
-(log) Probability of each action taken at each time step

Not going to worry about running it on GPUs/parallelizing right now.

We should maintain a replay buffer with a namedtuple:
timestep = [state, action, reward, next_state, value, log prob, done]
'''

# Hyperparameters
SEED = 42
epsilon = 0.2
GAEbias = 0.5 # decide later
c1 = 0.2
c2 = 0.2 # may use decay - this acts as exploration

    
    