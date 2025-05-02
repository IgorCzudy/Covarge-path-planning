import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from SimpleGridEnvironment_qmax import SimpleGridEnvironment
from tqdm import tqdm
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F

from TwoQTable import TwoQTable
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Konfiguracja
BUFFER_SIZE = 10000#10000
BATCH_SIZE = 4
GAMMA = 0.99
LR = 5e-3


# Sieć pojedynczego agenta (DRQN uproszczone do DQN)
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

# Hiper-sieć generująca wagi dla warstwy mieszającej
class HyperNetwork(nn.Module):
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, state):
        weights = self.fc(state)
        return torch.abs(weights)  # Wagi nieujemne

# Sieć mieszająca QMIX
class MixingNetwork(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.hyper_w1 = HyperNetwork(state_dim, num_agents * 32)
        self.hyper_b1 = HyperNetwork(state_dim, 32)
        self.hyper_w2 = HyperNetwork(state_dim, 32)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, agent_qs, state):
        # Warstwa 1
        w1 = self.hyper_w1(state).view(len(agent_qs), -1, 32)
        b1 = self.hyper_b1(state).view(-1, 1, 32)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        
        # Warstwa 2
        w2 = self.hyper_w2(state).view(-1, 32, 1)
        b2 = self.hyper_b2(state).view(-1, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze()
    


# QMIX Agent
# state_dim == 1
class QMIX:
    def __init__(self, obs_dim, state_dim, action_dim, num_agents):
        self.agents = [AgentQNetwork(obs_dim, action_dim) for _ in range(num_agents)]
        self.target_agents = [AgentQNetwork(obs_dim, action_dim) for _ in range(num_agents)] #AgentQNetwork(obs_dim, action_dim) 
        self.mixer = MixingNetwork(state_dim, num_agents)
        self.target_mixer = MixingNetwork(state_dim, num_agents)
        
        self.optimizer = optim.RMSprop(
            list(self.mixer.parameters()) + 
            [p for agent in self.agents for p in agent.parameters()],
            lr=LR
        )
        
        self.buffer = deque(maxlen=BUFFER_SIZE)
    
    def act(self, observations, epsilon):
        actions = []
        for i, obs in enumerate(observations):
            if np.random.random() < epsilon:
                actions.append(np.random.randint(0, 4))
            else:
                q_values = self.agents[i](torch.tensor(obs))
                actions.append(torch.argmax(q_values).item())
        return actions
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        # Próbkowanie z bufora
        batch = random.sample(self.buffer, BATCH_SIZE)
        # batch[0] = observations, actions, reward, next_observations, done, global_state
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, state_batch, next_state_batch = zip(*batch)

        device = next(self.mixer.parameters()).device
        reward = torch.tensor(reward_batch, dtype=torch.float32, device=device).unsqueeze(1)
        done = torch.tensor(done_batch, dtype=torch.float32, device=device).unsqueeze(1)
        state = torch.tensor(state_batch, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state_batch, dtype=torch.float32, device=device)


        agent_qs = []
        target_agent_qs = []

        for i in range(len(self.agents)):
            
            obs = torch.stack([torch.tensor(o[i]) for o in obs_batch]).to(device)
            next_obs = torch.stack([torch.tensor(o[i]) for o in next_obs_batch]).to(device)
            
            actions = torch.tensor([a[i] for a in action_batch], dtype=torch.long, device=device).unsqueeze(1)

            # przewidz q dla lokalna obserwacja agenta 
            #Obliczenie Q-values dla wykonanych akcji (dla straty)
            q_values = self.agents[i](obs)
            # Wybieramy tylko Q-value dla akcji, która faktycznie została wykonana
            q_taken = q_values.gather(1, actions)
            agent_qs.append(q_taken)

            # Obliczenie target Q-values dla następnego stanu (dla TD target)
            # Obliczamy Q-values dla wszystkich akcji w następnym stanie (target network)
            target_q_values = self.target_agents[i](next_obs)
            # Wybieramy maksymalne Q-value (jak w standardowym DQN)
            max_target_q = target_q_values.max(dim=1, keepdim=True)[0]
            target_agent_qs.append(max_target_q)

        
        # Stackowanie agentowych Q
        agent_qs = torch.stack(agent_qs, dim=1).squeeze(-1)        # [B, num_agents]
        # target_agent_qs = torch.stack(target_agent_qs, dim=1)       # [B, num_agents, 1]

        target_agent_qs = torch.stack(target_agent_qs, dim=1).squeeze(-1)       # [B, num_agents]

        # Obliczenie Q_total
        # state = torch.nn.functional.one_hot(state.long(), num_classes = 100**2).float()
        q_total = self.mixer(agent_qs, state)                       # [B]
        
        # next_state = torch.nn.functional.one_hot(state.long(), num_classes = 100**2).float()
        target_q_total = self.target_mixer(target_agent_qs, next_state).detach()

        # Obliczanie targetów
        targets = reward + GAMMA * (1 - done) * target_q_total.unsqueeze(1)

        # Loss + optymalizacja
        loss = torch.nn.functional.mse_loss(q_total, targets.squeeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
        self.optimizer.step()



from shortestPath import ShortestPath


env = ShortestPath()
obs_dim = 83#10 #100
state_dim = 85#20 #obs_dim**2
action_dim = 4  # 0 lub 1
num_agents = 2

agent = QMIX(obs_dim, state_dim, action_dim, num_agents)

episode_rewards = []
epsilons = []

epsilon=0.9
for episode in tqdm(range(1200), desc="Training"):
    observations, global_state = env.reset()
    # for a in agent.agents:
    #     a.reset_hidden()
    # for a in agent.agents:
    #     a.reset_hidden()

    done = False
    total_reward = 0
    while not done:
        actions = agent.act(observations, epsilon=epsilon)
        
        next_observations, reward, done, next_global_state = env.step(actions)
        agent.buffer.append((observations, actions, reward, next_observations, done, global_state, next_global_state))
        agent.update()

        # env.render()
        # print(f"{observations=}, {actions=}, {reward=}, {next_observations=}, {done=}, {global_state=}, {next_global_state=}, {epsilon=}")
        # while True:  # waiting for button press
        #     event = pygame.event.wait()
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         exit()
        #     elif event.type == pygame.KEYDOWN:
        #         print("Key 'n' pressed! Moving to the next iteration.")
        #         break


        total_reward += reward  # Accumulate reward
        global_state = next_global_state
        observations = next_observations
    episode_rewards.append(total_reward)  # After the episode ends
    epsilon *= 0.99
    epsilons.append(epsilon)

import pandas as pd 

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
roll = 30
ax1.plot(pd.Series(episode_rewards).rolling(roll).mean(), label="Agent 2")
# ax1.plot(episode_rewards, label="Agent 2")
ax1.set_ylabel("Total Reward")
ax1.set_title("Reward per Episode")
ax1.legend()
ax1.grid()
ax2.plot(epsilons, color="orange")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Epsilon")
ax2.grid()
plt.tight_layout()
plt.show()


env = ShortestPath(display=True)

observations, global_state = env.reset()
done = False

import pygame 

while not done:
    # Epsilon=0 dla czystej eksploatacji
    actions = agent.act(observations, epsilon=0)
    env.render()
    print(f"{observations=}, {actions=}, {reward=}, {next_observations=}, {done=}, {global_state=}, {next_global_state=}, {epsilon=}")
    while True:  # waiting for button press
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            print("Key 'n' pressed! Moving to the next iteration.")
            break
    next_observations, reward, done, next_global_state = env.step(actions)

    global_state = next_global_state
    observations = next_observations

