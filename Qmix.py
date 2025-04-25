import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# Konfiguracja
BUFFER_SIZE = 10000
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
                # obs_one_hot = torch.nn.functional.one_hot(torch.tensor(obs), num_classes = 100).float()
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



import numpy as np
from typing import List

class SimpleGridEnvironment:
    def __init__(self, render=False):
        self.grid_size = 10
        self.target_pos = (9, 9)
        self.max_steps = 50
        self.current_step = 0
        self.render = render
        
        # Nowe zmienne do śledzenia poprzednich pozycji
        self.prev_agent1_pos = (0, 0)
        self.prev_agent2_pos = (0, 0)

    def reset(self):
        self.agent1_pos = (0,9)#(np.random.randint(0,5), np.random.randint(0,5))
        self.agent2_pos = (0,0)#(np.random.randint(0,5), np.random.randint(0,5))
        self.prev_agent1_pos = self.agent1_pos
        self.prev_agent2_pos = self.agent2_pos
        self.current_step = 0
        
        if self.render:
            self._render()
            
        return (
            [self._get_obs(self.agent1_pos),
            self._get_obs(self.agent2_pos)],
            self._get_global_state()
        )

    def _calculate_reward(self):
        """Ulepszona funkcja nagród"""
        total_reward = 0
        done = False
        
        # Oblicz odległości
        curr_dist1 = self._distance(self.agent1_pos, self.target_pos)
        curr_dist2 = self._distance(self.agent2_pos, self.target_pos)
        prev_dist1 = self._distance(self.prev_agent1_pos, self.target_pos)
        prev_dist2 = self._distance(self.prev_agent2_pos, self.target_pos)

        # Nagrody za ruch w dobrym kierunku
        if curr_dist1 < prev_dist1:
            total_reward += 0.1  # Agent 1 zbliżył się do celu
        elif curr_dist1 > prev_dist1:
            total_reward -= 0.1  # Agent 1 oddalił się
            
        if curr_dist2 < prev_dist2:
            total_reward += 0.1  # Agent 2 zbliżył się do celu
        elif curr_dist2 > prev_dist2:
            total_reward -= 0.1  # Agent 2 oddalił się

        # Duża nagroda za osiągnięcie celu
        if curr_dist1 == 0 and curr_dist2 == 0:
            total_reward += 10.0
            done = True
            
        # Kara za bezczynność (ten sam dystans)
        if curr_dist1 == prev_dist1:
            total_reward -= 0.05
        if curr_dist2 == prev_dist2:
            total_reward -= 0.05
            
        # Mała kara za każdy krok
        total_reward -= 0.01
        
        return total_reward, done

    def _distance(self, pos1, pos2):
        """Oblicz odległość Manhattan"""
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def step(self, actions: List[int]):
        # Zapisz poprzednie pozycje przed ruchem
        self.prev_agent1_pos = self.agent1_pos
        self.prev_agent2_pos = self.agent2_pos
        
        # Reszta logiki ruchu bez zmian
        self._move_agent(0, actions[0])
        self._move_agent(1, actions[1])
        
        reward, done = self._calculate_reward()
        self.current_step += 1
        
        if self.render:
            self._render()
            time.sleep(0.5)

        if self.current_step >= self.max_steps:
            done = True
            
        return (
            [self._get_obs(self.agent1_pos),
            self._get_obs(self.agent2_pos)],
            reward,
            done,
            self._get_global_state()
        )
    
    def _get_obs(self, pos):
        """Normalizacja pozycji do zakresu [0, 1]"""
        return [pos[0]/self.grid_size, pos[1]/self.grid_size]
    
    def _get_global_state(self):
        """Globalny stan jako połączone pozycje agentów"""
        return [
            *(x / self.grid_size for x in self.agent1_pos),
            *(x / self.grid_size for x in self.agent2_pos),
            *(x / self.grid_size for x in self.target_pos)
        ]
        # return [
        #     *self.agent1_pos,
        #     *self.agent2_pos,
        #     *self.target_pos
        # ]
    
    def _move_agent(self, agent_idx, action):
        """Mechanika ruchu (4 kierunki)"""
        x, y = self.agent1_pos if agent_idx == 0 else self.agent2_pos
        
        # Mapowanie akcji (0-3 dla 4 kierunków)
        if action == 0:   # GÓRA
            y = min(y + 1, self.grid_size-1)
        elif action == 1: # DÓŁ
            y = max(y - 1, 0)
        elif action == 2: # PRAWO
            x = min(x + 1, self.grid_size-1)
        elif action == 3: # LEWO
            x = max(x - 1, 0)
            
        if agent_idx == 0:
            self.agent1_pos = (x, y)
        else:
            self.agent2_pos = (x, y)
        
    def _render(self):
        """Wizualizacja ASCII gridu"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Zaznacz target
        tx, ty = self.target_pos
        grid[ty][tx] = 'T'
        
        # Zaznacz agentów
        a1x, a1y = self.agent1_pos
        a2x, a2y = self.agent2_pos
        grid[a1y][a1x] = 'A'
        grid[a2y][a2x] = 'B'
        
        # Wyświetl grid
        print("\n" + "-"*(self.grid_size*2))
        for row in reversed(grid):  # (0,0) na dole
            print(" ".join(row))



import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Optional
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt 
from ploting import plot_graph_two_agents, plot_matrix
from Rl_run import initialdouble_q_heatmap
from Rl_run import make_Q_table_plot, updatedouble_q_heatmap
class TwoQTable(): #gym.Env
    
    def __init__(self, grid_size: Tuple[int, int] = (7, 7), display: bool = False, map_number: int = 0, first_agent_starting_position = (0,0), secend_agent_starting_position = (6,6)):
        # super(TwoQTable, self).__init__()
        self.map_number = map_number

        self.first_agent_starting_position = first_agent_starting_position
        self.secend_agent_starting_position = secend_agent_starting_position

        self.first_agent_position: Tuple[int, int] = (0, 0)
        self.secend_agent_position: Tuple[int, int] = (0, 9)
        # self.target_pos = (9, 9)

        self.grid_width: int = grid_size[0]
        self.grid_height: int = grid_size[1]
        self.num_of_steps: int = 0

        self.grid: Optional[np.dnarray] = None


        if display:
            pygame.init()
            self.cell_size: int = 50  # Size of each grid cell in pixels
            self.window_width: int = self.grid_width * self.cell_size
            self.window_height: int = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )

            self.colors = {
                0: (255, 255, 255),  # White for empty cells
                1: (0, 0, 0),  # Green for obstacles
                2: (0, 255, 0),  # Green for visited cell by first agent
                3: (0, 100, 0),  # Less green for visited cell by secend agent
                9: (255, 0, 0),  # Red for the agent
                8: (150, 0, 0),  # lighter red for the secend agent
            }
            self.font = pygame.font.Font(None, 30)  # Define the font for numbers

    def make_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_width, self.grid_height), dtype=int)

        grid[0, 5] = 1
        grid[0, 6] = 1
        grid[1, 5] = 1
        grid[1, 6] = 1
        
        grid[3, 3] = 1
        grid[3, 5] = 1
        grid[4, 3] = 1
        grid[4, 5] = 1

        grid[3, 4] = 1
        grid[4, 4] = 1
        # grid[6, 5] = 1
        # grid[6, 6] = 1

        grid[5, 0] = 1
        grid[6, 0] = 1
        grid[6, 1] = 1
        return grid


    def _get_observation(self, agent_number: int) -> int:

        agent_position = self.first_agent_position if agent_number==1 else self.secend_agent_position 
        
        # assert 0 <= n <= 24
        return [agent_position[0]/self.grid_width, agent_position[1]/self.grid_width]
    
    def _get_global_state(self):
        """Globalny stan jako połączone pozycje agentów"""
        return [
            *(x / self.grid_width for x in self.first_agent_position),
            *(x / self.grid_width for x in self.secend_agent_position)
            # *(x / self.grid_width for x in self.target_pos)
        ]



    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[int, int, Dict]:
        # super().reset(seed=seed)

        self.grid = self.make_grid()
        self.first_agent_position = self.first_agent_starting_position
        self.secend_agent_position = self.secend_agent_starting_position
        self.grid[self.first_agent_position] = 2
        self.grid[self.secend_agent_position] = 3

        return (
            [self._get_observation(agent_number=1),  # Obserwacje agentów
            self._get_observation(agent_number=2)],
            self._get_global_state()         # Globalny stan
        )


    def step(self, actions: List[int]) -> Tuple[int, int, int, int, bool, bool, Dict]:
        # Ruch agentów
        action1 = actions[0]  # Agent 1
        action2 = actions[1]  # Agent 2

        assert 0 <= action1 <= 15, "action must be in the range from 0 to 15"
        assert 0 <= action2 <= 15, "action must be in the range from 0 to 15"
        self.num_of_steps += 1

        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        reword ={0: 0, 1: 0} #agent_number: reword
        for i, (agent_position, action) in enumerate([(self.first_agent_position, action1), (self.secend_agent_position, action2)]):
            x, y = agent_position
            dx, dy = moves[action]
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height): #move out of bandry
                reword[i] -= 0.1

            elif self.grid[new_x, new_y] == 1:  # Obstacle
                reword[i] -= 0.1
            
            # elif (i==1 and (new_x, new_y) == self.first_agent_position):  # Collision
            #     reword[i] = -0.1
            
            elif self.grid[new_x, new_y] == 2 or self.grid[new_x, new_y] == 3:  # Visited cell
                if i==0:
                    self.first_agent_position = new_x, new_y
                    self.grid[new_x, new_y] = 2
                else:
                    self.secend_agent_position = new_x, new_y
                    self.grid[new_x, new_y] = 3
                reword[i] -= 0.05 #0.5

            else:  # New valid move
                if i==0:
                    self.first_agent_position = new_x, new_y
                    self.grid[new_x, new_y] = 2
                else:
                    self.secend_agent_position = new_x, new_y
                    self.grid[new_x, new_y] = 3
                reword[i] += 0.1 #0.5
        
        done = False
        if np.all((self.grid == 2) | (self.grid == 3) | (self.grid == 1)):
            if reword[0] >= 0.1 :
                reword[0] = 1
            else:
                reword[1] = 1
            done = True
        return (
            [self._get_observation(0),  # Nowe obserwacje
            self._get_observation(1)],
            reword[0] + reword[1],                          # Wspólna nagroda
            done,                            # Czy epizod zakończony
            self._get_global_state()         # Nowy stan globalny
        )

        # return self._get_observation(0), self._get_observation(1), reword[0], reword[1], done, False, {}

        
    def render(self, mode="human") -> None:
        render_grid = self.grid.copy()
        render_grid[self.first_agent_position] = 9
        render_grid[self.secend_agent_position] = 8
        
        
        self.window.fill((0, 0, 0))  # Black background

        for row in range(self.grid_width):
            for col in range(self.grid_height):
                value = render_grid[row, col]
                color = self.colors.get(
                    value, (0, 0, 0)
                )  # Default to black for unknown values
                pygame.draw.rect(
                    self.window,
                    color,
                    (
                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

                pygame.draw.rect(
                    self.window,
                    (0, 0, 0),
                    (
                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                    1,
                )
                text = self.font.render(
                    f"{col + row * self.grid_height}", True, (169, 169, 169)
                )  # Draw the number (index as example)
                text_rect = text.get_rect(
                    center=(
                        col * self.cell_size + self.cell_size // 2,
                        row * self.cell_size + self.cell_size // 2,
                    )
                )

                self.window.blit(text, text_rect)


        pygame.display.flip()


env = SimpleGridEnvironment()
obs_dim = 2 #100
state_dim = 6 #obs_dim**2
action_dim = 4  # 0 lub 1
num_agents = 2

agent = QMIX(obs_dim, state_dim, action_dim, num_agents)

episode_rewards = []

epsilon=0.9
for episode in tqdm(range(400), desc="Training"):
    observations, global_state = env.reset()
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


plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode")
plt.grid()
plt.show()


env = SimpleGridEnvironment(render=True)

observations, global_state = env.reset()
done = False

while not done:
    # Epsilon=0 dla czystej eksploatacji
    actions = agent.act(observations, epsilon=0)
    import time; time.sleep(0.5)
    next_observations, reward, done, next_global_state = env.step(actions)

    global_state = next_global_state
    observations = next_observations

