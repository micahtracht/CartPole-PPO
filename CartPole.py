import argparse
import random
from typing import Deque, List, Optional, Sequence, Tuple, NamedTuple
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
            nn.Linear(128, 1)
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
            
            logits = self.pi(obs)  # [batch, action_dim]
            dist = Categorical(logits=logits)
            action = dist.sample()        # [batch]
            logp = dist.log_prob(action)  # [batch]
            value = self.v(obs).squeeze(-1)  # [batch]
            
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
    buffer: List[Tuple[TrajectoryStep, Optional[float], Optional[float]]]
    
    def __init__(self, capacity: int) -> None:
        self.buffer = []
        self.capacity = capacity
    
    def add(self, trajStep: TrajectoryStep, advantage: Optional[float], total_return: Optional[float]) -> None:
        self.buffer.append((trajStep, advantage, total_return))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def sample(self, batch_size: int) -> Tuple[List[TrajectoryStep], List[float], List[float]]:
        batch = random.sample(self.buffer, batch_size)
        trajStep, advantages, returns = map(list, zip(*batch))
        return trajStep, advantages, returns

    def __len__(self) -> int:
        return len(self.buffer)
    
    def calculate_advantage_and_values(self, gamma: float, GAEbias: float = 1) -> None:
        N = len(self.buffer)
        
        rewards = [self.buffer[i][0].reward for i in range(N)]
        values = [self.buffer[i][0].value for i in range(N)]
        dones = [self.buffer[i][0].done for i in range(N)]
        
        advantages = [0] * N
        gae = 0
        
        for t in reversed(range(N)):
            next_val = 0.0 if dones[t] else (values[t+1] if t+1 < N else 0)
            delta = rewards[t] + next_val * gamma - values[t]
            gae = delta + gamma * GAEbias * (0.0 if dones[t] else gae)
            advantages[t] = gae
        
        returns = [values[i] + advantages[i] for i in range(N)]
        
        self.buffer = [(self.buffer[i][0], advantages[i], returns[i]) for i in range(N)]
    
    def clear(self) -> None:
        self.buffer = []


def collect_rollouts(env, model: ActorCritic, buffer: RolloutBuffer, rollout_length: int):
    obs = env.reset()[0]
    for _ in range(rollout_length):
        obs_batch = np.expand_dims(obs, axis=0)
        actions, vals, logps = model.step(obs_batch)
        
        action = int(actions[0])
        value = float(vals[0])
        logp = float(logps[0])
        
        next_obs, reward, done, _ = env.step(action)
        
        step = TrajectoryStep(
            obs=obs,
            action=action,
            logp=logp,
            reward=reward,
            value=value,
            done=done
        )
        buffer.add(step, None, None)
        
        obs = env.reset()[0] if done else next_obs
    
    return obs


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    clip_param: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 10,
    batch_size: int = 64,
):
    obs_np = np.stack([step.obs for step, adv, ret in buffer.buffer])
    actions_np = np.array([step.action for step, adv, ret in buffer.buffer])
    old_logps_np = np.array([step.logp for step, adv, ret in buffer.buffer])
    adv_np = np.array([adv for step, adv, ret in buffer.buffer])
    ret_np = np.array([ret for step, adv, ret in buffer.buffer])
    
    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
    actions_tensor = torch.as_tensor(actions_np, dtype=torch.int64)
    old_logps_tensor = torch.as_tensor(old_logps_np, dtype=torch.float32)
    adv_tensor = torch.as_tensor(adv_np, dtype=torch.float32)
    ret_tensor = torch.as_tensor(ret_np, dtype=torch.float32)
    
    dataset_size = obs_tensor.shape[0]
    indices = list(range(dataset_size))

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start+batch_size]

            b_obs = obs_tensor[batch_idx]
            b_actions = actions_tensor[batch_idx]
            b_old_logp = old_logps_tensor[batch_idx]
            b_adv = adv_tensor[batch_idx]
            b_ret = ret_tensor[batch_idx]

            logits = model.pi(b_obs)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = (new_logp - b_old_logp).exp()
            unclipped = ratio * b_adv
            clipped = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * b_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_pred = model.v(b_obs).squeeze(-1)
            value_loss = F.mse_loss(value_pred, b_ret)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    buffer.clear()


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--rollout_length", type=int, default=2048)
    parser.add_argument("--total_updates", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    buffer = RolloutBuffer(args.capacity)

    obs = env.reset()[0]
    ep_reward = 0.0
    reward_queue = deque(maxlen=args.log_interval)

    for update in range(1, args.total_updates + 1):
        for _ in range(args.rollout_length):
            obs_batch = np.expand_dims(obs, axis=0)
            acts, vals, logps = model.step(obs_batch)
            action = int(acts[0])
            value = float(vals[0])
            logp = float(logps[0])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            step = TrajectoryStep(
                obs=obs,
                action=action,
                logp=logp,
                reward=reward,
                value=value,
                done=done
            )
            buffer.add(step, None, None)

            obs = env.reset()[0] if done else next_obs
            if done:
                reward_queue.append(ep_reward)
                ep_reward = 0.0

        buffer.calculate_advantage_and_values(args.gamma, args.gae_lambda)

        ppo_update(
            model,
            optimizer,
            buffer,
            clip_param=args.clip_param,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        if update % args.log_interval == 0 and len(reward_queue) > 0:
            avg_r = np.mean(reward_queue)
            print(f"[Update {update:4d}] avg reward (last {args.log_interval} eps): {avg_r:.2f}")

    env.close()
