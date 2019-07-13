import numpy as np
import random
import copy
from segment_tree import SumSegmentTree, MinSegmentTree
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
SIGMA_DECAY = 0.95
SIGMA_MIN = 0.005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = [Actor(state_size, action_size, random_seed).to(device) for i in range(num_agents)]
        self.actor_target = [Actor(state_size, action_size, random_seed).to(device) for i in range(num_agents)]
        self.actor_optimizer = [optim.Adam(self.actor_local[i].parameters(), lr=LR_ACTOR)for i in range(num_agents)]

        # Critic Network (w/ Target Network)
        self.critic_local = [Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device) for i in range(num_agents)]
        self.critic_target = [Critic(state_size * num_agents, action_size * num_agents, random_seed).to(device) for i in range(num_agents)]
        self.critic_optimizer = [optim.Adam(self.critic_local[i].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) for i in range(num_agents)]

        # Noise process
        self.noise = OUNoise(action_size, num_agents, random_seed)
        self.noise_epsilon = 1

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory = PrioritizedReplayMemory(BUFFER_SIZE)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.push(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if self.memory._next_idx > BATCH_SIZE:
            self.learn(GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = [torch.from_numpy(states[i]).float().to(device) for i in range(self.num_agents)]
        #self.actor_local.eval()
        with torch.no_grad():
            actions = [self.actor_local[i](states[i]).cpu().data.numpy() for i in range(self.num_agents)]
            actions = np.vstack(actions)
        #self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            gamma (float): discount factor
        """
        experiences, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = experiences
        states = [_.squeeze() for _ in states.split(1, 1)]
        actions = [_.squeeze() for _ in actions.split(1, 1)]
        rewards = [_.squeeze() for _ in rewards.split(1, 1)]
        next_states = [_.squeeze() for _ in next_states.split(1, 1)]
        dones = [_.squeeze() for _ in dones.split(1, 1)]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.cat([self.actor_target[i](next_states[i]) for i in range(self.num_agents)], 1)

        _next_states = torch.cat(next_states, 1).squeeze()

        _states = torch.cat(states, 1).squeeze()
        _actions = torch.cat(actions, 1).squeeze()

        for i in range(self.num_agents):
            Q_targets_next = self.critic_target[i](_next_states, actions_next).detach().squeeze()
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[i] + (gamma * Q_targets_next * (1 - dones[i]))

            # Compute critic loss
            Q_expected = self.critic_local[i](_states, _actions).squeeze()

            diff = Q_expected - Q_targets
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())

            critic_loss = F.mse_loss(Q_expected, Q_targets).squeeze() * weights
            critic_loss = critic_loss.mean()
            
            # Minimize the loss
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()

        # ---------------------------- update actor ---------------------------- #
        action_preds = torch.cat([self.actor_local[i](states[i]) for i in range(self.num_agents)], 1)

        for i in range(self.num_agents):
            # Compute actor loss
            actor_loss = -self.critic_local[i](_states, action_preds).mean()
            # Minimize the loss
            self.actor_optimizer[i].zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer[i].step()

        # ----------------------- update target networks ----------------------- #
        for i in range(self.num_agents):
            self.soft_update(self.critic_local[i], self.critic_target[i], TAU)
            self.soft_update(self.actor_local[i], self.actor_target[i], TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size, num_agents, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones((num_agents, action_size))
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        if self.sigma > SIGMA_MIN:
            self.sigma *= SIGMA_DECAY

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
        self.state = x + dx
        return self.state

class PrioritizedReplayMemory:
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame=1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        idx = self._next_idx
        exp = self.experience(state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(exp)
        else:
            self._storage[self._next_idx] = exp
        self._next_idx = (self._next_idx + 1) % self._maxsize


        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        states = torch.from_numpy(np.array([self._storage[i].state for i in idxes])).float().to(device)
        actions = torch.from_numpy(np.array([self._storage[i].action for i in idxes])).float().to(device)
        rewards = torch.from_numpy(np.array([self._storage[i].reward for i in idxes])).float().to(device)
        next_states = torch.from_numpy(np.array([self._storage[i].next_state for i in idxes])).float().to(device)
        dones = torch.from_numpy(np.array([self._storage[i].done for i in idxes]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)

        weights = []

        #find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame+=1
        
        #max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=device, dtype=torch.float) 
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority+1e-5) ** self._alpha
            self._it_min[idx] = (priority+1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority+1e-5))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)