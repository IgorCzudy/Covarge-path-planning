import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import pygame
import matplotlib.pyplot as plt
from TwoQTable import TwoQTable

# Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TAU = 0.01  # For soft target updates
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class HyperNetwork(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, state):
        return self.fc(state)

class MixingNetwork(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.hyper_w1 = HyperNetwork(state_dim, num_agents * 32)
        self.hyper_b1 = HyperNetwork(state_dim, 32)
        self.hyper_w2 = HyperNetwork(state_dim, 32)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(state)).view(-1, self.num_agents, 32)
        b1 = self.hyper_b1(state).view(-1, 1, 32)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(-1, 32, 1)
        b2 = self.hyper_b2(state).view(-1, 1, 1)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze()

class QMIX:
    def __init__(self, obs_dim, state_dim, action_dim, num_agents):
        self.num_agents = num_agents
        self.agents = [AgentQNetwork(obs_dim, action_dim) for _ in range(num_agents)]
        self.target_agents = [AgentQNetwork(obs_dim, action_dim) for _ in range(num_agents)]
        self.mixer = MixingNetwork(state_dim, num_agents)
        self.target_mixer = MixingNetwork(state_dim, num_agents)
        
        # Initialize targets to match online networks
        for target, online in zip(self.target_agents, self.agents):
            target.load_state_dict(online.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        self.optimizer = optim.Adam(
            list(self.mixer.parameters()) + 
            [p for agent in self.agents for p in agent.parameters()],
            lr=LR
        )
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.epsilon = EPS_START

    def act(self, observations):
        actions = []
        for i, obs in enumerate(observations):
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(0, 4))
            else:
                with torch.no_grad():
                    q = self.agents[i](torch.FloatTensor(obs))
                    actions.append(q.argmax().item())
        return actions

    def update_targets(self):
        # Soft update target networks
        for target, online in zip(self.target_agents, self.agents):
            for t_param, o_param in zip(target.parameters(), online.parameters()):
                t_param.data.copy_(TAU*o_param.data + (1-TAU)*t_param.data)
        
        for t_param, o_param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            t_param.data.copy_(TAU*o_param.data + (1-TAU)*t_param.data)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        # Sample batch
        batch = random.sample(self.buffer, BATCH_SIZE)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, state_batch, next_state_batch = zip(*batch)

        # Convert to tensors
        obs = [torch.FloatTensor(np.array([o[i] for o in obs_batch])) for i in range(self.num_agents)]
        next_obs = [torch.FloatTensor(np.array([no[i] for no in next_obs_batch])) for i in range(self.num_agents)]
        actions = torch.LongTensor(action_batch)
        rewards = torch.FloatTensor(reward_batch)
        dones = torch.FloatTensor(done_batch)
        states = torch.FloatTensor(state_batch)
        next_states = torch.FloatTensor(next_state_batch)

        # Calculate Q values for current states
        current_qs = []
        for i in range(self.num_agents):
            q_values = self.agents[i](obs[i])
            current_q = q_values.gather(1, actions[:, i].unsqueeze(1))
            current_qs.append(current_q)
        current_qs = torch.cat(current_qs, dim=1)

        # Calculate target Q values
        with torch.no_grad():
            target_qs = []
            for i in range(self.num_agents):
                target_q = self.target_agents[i](next_obs[i]).max(1)[0].unsqueeze(1)
                target_qs.append(target_q)
            target_qs = torch.cat(target_qs, dim=1)
            target_total = self.target_mixer(target_qs, next_states)
            targets = rewards + (1 - dones) * GAMMA * target_total

        # Compute loss
        q_total = self.mixer(current_qs, states)
        loss = F.mse_loss(q_total, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
        self.optimizer.step()

        # Update targets
        self.update_targets()

        # Decay epsilon
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        return loss.item()

# Training Loop
env = TwoQTable(display=True)
agent = QMIX(obs_dim=51, state_dim=53, action_dim=4, num_agents=2)

episode_rewards = []
losses = []

for episode in range(1000):
    obs, global_state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        actions = agent.act(obs)
        next_obs, reward, done, next_global = env.step(actions)
        # Scale reward
        scaled_reward = reward * 10  
        
        agent.buffer.append((obs, actions, scaled_reward, next_obs, done, global_state, next_global))
        print(obs, actions, scaled_reward, next_obs, done, global_state, next_global)
        env.render()     
        
        loss = agent.update()
        
        total_reward += reward
        obs = next_obs
        global_state = next_global
    
    episode_rewards.append(total_reward)
    losses.append(loss if loss else 0)
    
    print(f"Ep {episode}, Reward: {total_reward:.1f}, Eps: {agent.epsilon:.2f}, Loss: {losses[-1]:.2f}")

# Plotting results
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.subplot(122)
plt.plot(losses)
plt.title("Training Loss")
plt.show()