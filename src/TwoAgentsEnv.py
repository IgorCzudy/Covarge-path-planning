
import sys
import os
sys.path.append(os.path.abspath('.'))
from create_data_structure import create_networkX_graph

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import pygame

class TwoAgentsEnv(gym.Env):

    def __init__(self, plot=False):
        super(TwoAgentsEnv, self).__init__()
        self.agent1_position: Tuple[int, int] = (0, 0)
        self.agent2_position: Tuple[int, int] = (6, 6)

        # self.graph_number = graph_number
        self.grid_size = 7
        self.grid_width = 7
        self.grid_height = 7
        
        self.num_of_steps = 0
        self.max_steps = 500
        
        if plot:
            # self.fig, self.ax = plt.subplots(figsize=(8, 8))
            # plt.ion()
            pygame.init()
            self.cell_size: int = 30  # Size of each grid cell in pixels
            self.window_width: int = self.grid_size * self.cell_size
            self.window_height: int = self.grid_size * self.cell_size
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


    def make_grid(self) -> nx.Graph:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.start_agent1_position = (0, 0)
        self.start_agent2_position = (6, 6)
        
        grid[4, 0] = 1
        grid[4, 1] = 1
        grid[5, 0] = 1
        grid[5, 1] = 1
        grid[6, 0] = 1
        grid[6, 1] = 1

        grid[3, 5] = 1
        grid[3, 6] = 1
        grid[4, 5] = 1
        grid[4, 6] = 1

        return grid
    

    def xy_pos_to_int(self, x: int, y: int) -> int:
        return x * self.grid_size + y


    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[int, int, Dict]:
        super().reset(seed=seed)
        self.num_of_steps = 0
        self.grid = self.make_grid()
        self.agent1_position = self.start_agent1_position
        self.agent2_position = self.start_agent2_position
        
        self.grid[self.agent1_position] = 2
        self.grid[self.agent2_position] = 3
        
        return self._get_observation(agent_number=0), self._get_observation(agent_number=1)


    def step(self, action1: int, action2: int) -> Tuple[int, int, int, int, bool, bool, Dict]:
        assert 0 <= action1 <= 3, "action must be in the range from 0 to 3"
        assert 0 <= action2 <= 3, "action must be in the range from 0 to 3"    
        self.num_of_steps += 1

        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        rewards = [0, 0]
        agents = [self.agent1_position, self.agent2_position] 
        actions = [action1, action2]

        for i, (agent, action) in enumerate(zip(agents, actions)):

            x, y = agent
            dx, dy = moves[action]
            new_x, new_y = x + dx, y + dy

            if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height): #move out of bandry
                rewards[i] -= 1.0

            elif self.grid[new_x, new_y] == 1:  # Obstacle
                rewards[i] -= 1.0
            
            elif self.grid[new_x, new_y] == 2 or self.grid[new_x, new_y] == 3:  # Visited cell
                if i == 0: 
                    self.agent1_position = new_x, new_y
                else:
                    self.agent2_position = new_x, new_y
                rewards[i] -= 0.5#0.05

            else:  # New valid move
                if i == 0: 
                    self.grid[new_x, new_y] = 2
                    self.agent1_position = new_x, new_y
                else:
                    self.grid[new_x, new_y] = 3
                    self.agent2_position = new_x, new_y
                rewards[i] += 2.0#0.1    

        done = False
        if self.num_of_steps >= self.max_steps:
            done = True
        
        if np.all((self.grid == 2) | (self.grid == 3) | (self.grid == 1)):
            if rewards[0] >= 2.0:
                rewards[0] = 100.0#1.0
            else:
                rewards[1] = 100.0#1.0
            done = True

        return self._get_observation(agent_number=0), self._get_observation(agent_number=1), rewards[0], rewards[1], done 


    def render(self, action1: int, reward1: float, action2: int, reward2: float, mode="human") -> None:

        grid = self.grid.copy()

        grid[self.agent1_position] = 9
        grid[self.agent2_position] = 8
        
        
        self.window.fill((0, 0, 0))  # Black background

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                value = grid[row, col]
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
                    f"{col + row * self.grid_size}", True, (169, 169, 169)
                )  # Draw the number (index as example)
                text_rect = text.get_rect(
                    center=(
                        col * self.cell_size + self.cell_size // 2,
                        row * self.cell_size + self.cell_size // 2,
                    )
                )

                self.window.blit(text, text_rect)


        pygame.display.flip()


    def _get_observation(self, agent_number) -> int:
        if agent_number == 0:
            agent = self.agent1_position
        else:
            agent = self.agent2_position

        x_pos, y_pos = agent
      
        grid_pos  = np.zeros((self.grid_size, self.grid_size))
        grid_pos[x_pos, y_pos] = 1

        return np.stack([(self.grid == 1).astype(int), 
                         ((self.grid == 3) | (self.grid == 2)).astype(int), 
                         grid_pos], axis=0) #(3, 10, 10)

import matplotlib.pyplot as plt
# from IPython import display  # Tylko dla Jupyter Notebook

# Inicjalizacja wykresów PRZED treningiem
plt.ion()  # Tryb interaktywny
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
line1, = ax1.plot([], [], label="Agent 1")
line2, = ax1.plot([], [], label="Agent 2")
line_eps, = ax2.plot([], [], color="orange")
line_loss1, = ax3.plot([], [], label="Loss1")
line_loss2, = ax3.plot([], [], label="Loss2")

ax1.set_ylabel("Reward")
ax2.set_ylabel("Epsilon")
ax3.set_ylabel("Loss")
ax1.legend()
ax1.grid()
ax2.grid()
ax3.grid()



env = TwoAgentsEnv()
# obs_dim = 10 #75
obs_dim = (3, 7, 7) #(C, H, W)
action_dim = 4  # 0 lub 1

from DQNnetwork import DQN
agent1 = DQN(obs_dim, action_dim, )
agent2 = DQN(obs_dim, action_dim, )

episode_rewards1 = []
episode_rewards2 = []
epsilons = []
losses1 = []
losses2 = []

epsilon_start = 1.0
epsilon_end = 0.00
epsilon_decay = 0.99

for episode in tqdm(range(500)):
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay**episode))
    observation1, observation2 = env.reset()
    done = False
    total_reward1 = 0
    total_reward2 = 0
    while not done:
        
        action1 = agent1.act(observation1, epsilon=epsilon)
        action2 = agent2.act(observation2, epsilon=epsilon)

        next_observation1, next_observation2, reward1, reward2, done = env.step(action1, action2)
        # env.render(action1, reward1, action2, reward2)
        # print(f"{observation1=}, {action1=}, {reward1=}, {next_observation1=}, {done=}, {epsilon=}")
        # while True:  # waiting for button press
        #     event = pygame.event.wait()
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         exit()
        #     elif event.type == pygame.KEYDOWN:
        #         print("Key 'n' pressed! Moving to the next iteration.")
        #         break

        agent1.buffer.append((observation1, action1, reward1, next_observation1, done))
        agent2.buffer.append((observation2, action2, reward2, next_observation2, done))
        loss1 = agent1.update()
        loss2 = agent2.update()
        losses1.append(loss1)
        losses2.append(loss2)

        total_reward1 += reward1  # Accumulate reward
        total_reward2 += reward2  # Accumulate reward

        observation1 = next_observation1
        observation2 = next_observation2

    episode_rewards1.append(total_reward1)  # After the episode ends
    episode_rewards2.append(total_reward2)  # After the episode ends
    epsilons.append(epsilon)
    epsilon *= 0.995

    if episode % 5 == 0:
        line1.set_ydata(pd.Series(episode_rewards1)) #.rolling(30).mean())
        line1.set_xdata(range(len(episode_rewards1)))
        line2.set_ydata(pd.Series(episode_rewards2)) #.rolling(30).mean())
        line2.set_xdata(range(len(episode_rewards2)))
        line_eps.set_ydata(epsilons)
        line_eps.set_xdata(range(len(epsilons)))
        
        line_loss1.set_ydata(pd.Series(losses1).rolling(50).mean())
        line_loss1.set_xdata(range(len(losses1)))
        
        line_loss2.set_ydata(pd.Series(losses2).rolling(50).mean())
        line_loss2.set_xdata(range(len(losses2)))
        

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        plt.draw()
        plt.pause(0.5)  # Wymagane do odświeżenia wykresu


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
roll = 30
ax1.plot(pd.Series(episode_rewards1).rolling(roll).mean(), label="Agent 1")
ax1.plot(pd.Series(episode_rewards2).rolling(roll).mean(), label="Agent 2")
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


env = TwoAgentsEnv(plot=True)

observations, global_state = env.reset()
done = False

while not done:
    # Epsilon=0 dla czystej eksploatacji
    action1 = agent1.act(observation1, epsilon=0.0)
    action2 = agent2.act(observation2, epsilon=0.0)
    env.render(action1, reward1, action2, reward2)

    next_observation1, next_observation2, reward1, reward2, done = env.step(action1, action2)
    
    agent1.buffer.append((observation1, action1, reward1, next_observation1, done))
    agent2.buffer.append((observation2, action2, reward2, next_observation2, done))
    agent1.update()
    agent2.update()

    observation1 = next_observation1
    observation2 = next_observation2
