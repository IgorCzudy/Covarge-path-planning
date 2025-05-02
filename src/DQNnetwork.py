import numpy as np
import torch
from torch import nn
from collections import deque
import torch.optim as optim
import random


BUFFER_SIZE = 1000
GAMMA = 0.99
BATCH_SIZE = 16        # Increased from 4 (better generalization, stable updates)
LR = 1e-3              # Reduced from 5e-2 (prevovershoot)
TAU = 0.01             # Slower target network update (more stable targets)


class DQNnetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.seq(x)

class DQN():
    def __init__(self, obs_dim, action_dim):
        self.action_dim = action_dim
        
        self.dqn = DQNnetwork(obs_dim, action_dim)
        self.target_dqn = DQNnetwork(obs_dim, action_dim)

        self.buffer = deque(maxlen=BUFFER_SIZE)

        self.optymizer = optim.Adam(self.dqn.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.update_count = 0


    def act(self, observation, epsilon):
        if random.random() < epsilon:
            return np.random.randint(0, self.action_dim) 

        observation = torch.tensor(observation, dtype=torch.float32)
        return self.dqn(observation).argmax().item()
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return 
        

        # start_idx = random.randint(0, len(self.buffer)-BATCH_SIZE)
        # from itertools import islice
        # batch = list(islice(self.buffer, start_idx, start_idx+BATCH_SIZE))
        batch = random.sample(self.buffer, BATCH_SIZE)

        observation, action, reward, next_observation, done = zip(*batch)
        observation_batch = torch.tensor(observation, dtype=torch.float32)
        action_batch = torch.tensor(action).unsqueeze(1)
        reword_batch = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        next_observation_batch = torch.tensor(next_observation, dtype=torch.float32)
        done_batch = torch.tensor(done, dtype=torch.float32).unsqueeze(1)


        # Obliczanie Q-values z głównej sieci
        q_values = self.dqn(observation_batch).gather(1, action_batch)
        
        # Obliczanie wartości docelowej Q (z sieci docelowej)
        next_Q_values = self.target_dqn(next_observation_batch).max(1, keepdim=True)[0]
        target_Q_values = reword_batch + (GAMMA * next_Q_values * (1 - done_batch))

        loss = self.loss_fn(q_values.flatten(), target_Q_values.flatten())

        self.optymizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optymizer.step()

        # Aktualizowanie sieci docelowej co N kroków
        self.update_count += 1
        if self.update_count % 50 == 0:  # np. co 100 kroków
            self.update_target()

    def update_target(self):
        """Aktualizacja sieci docelowej (target)"""
        with torch.no_grad():
            for target_param, local_param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


