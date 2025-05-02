
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
        self.agent2_position: Tuple[int, int] = (29, 29)

        # self.graph_number = graph_number
        self.grid_size = 30
        
        self.num_of_steps = 0
        self.max_steps = 600
        
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


    def make_graph(self) -> nx.Graph:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.start_agent1_position = (0, 0)
        self.start_agent2_position = (29, 29)

        for i in range(5, 20):
            for j in range(10, 15):
                grid[i, j] = 1
                grid[i, j+10] = 1

        for i in range(23, 28):
            for j in range(5, 15):
                grid[i, j] = 1



        graph, self.not_obstacles = create_networkX_graph(grid)
        nx.set_node_attributes(graph, False, 'visited')

        return graph
    

    def xy_pos_to_int(self, x: int, y: int) -> int:
        return x * self.grid_size + y


    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[int, int, Dict]:
        super().reset(seed=seed)
        self.num_of_steps = 0
        self.graph = self.make_graph()
        self.agent1_position = self.start_agent1_position
        self.agent2_position = self.start_agent2_position
        
        for agent in [self.agent1_position, self.agent2_position]:
            self.graph.nodes[self.xy_pos_to_int(agent[0], agent[1])]['visited'] = True
        
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
            node_number = self.xy_pos_to_int(x,y)
            new_node_number = self.xy_pos_to_int(new_x, new_y)

            if not self.graph.has_edge(node_number, new_node_number):  # Obstacle or out of boundry (no connection)
                rewards[i] -= 0.1
            
            elif self.graph.nodes[new_node_number]['visited']:  # Visited cell
                if i == 0: 
                    self.agent1_position = new_x, new_y
                else:
                    self.agent2_position = new_x, new_y
                rewards[i] -= 0.05

            else:  # New valid move
                self.graph.nodes[new_node_number]['visited'] = True
                if i == 0: 
                    self.agent1_position = new_x, new_y
                else:
                    self.agent2_position = new_x, new_y
                rewards[i] += 0.1

        #TODO collision panelty 
    

        done = False
        #TODO maybe at step limit 
        if self.num_of_steps >= self.max_steps:
            done = True
        
        if all([self.graph.nodes[node]['visited'] for node in self.not_obstacles]):
            rewards[0] = 1.0
            rewards[1] = 1.0
            done = True

        return self._get_observation(agent_number=0), self._get_observation(agent_number=1), rewards[0], rewards[1], done 

    def render(self, action1: int, reward1: float, action2: int, reward2: float, mode="human") -> None:
        grid  = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_nr = self.xy_pos_to_int(i, j)
                if self.graph.nodes[node_nr]['visited']:
                    grid[i ,j] = 3 #visited
                elif not list(self.graph.neighbors(node_nr)):
                    grid[i ,j] = 1 #obstyckle
        
                
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



    # def render(self, action1: int, reward1: float, action2: int, reward2: float, mode="human") -> None:
    #     act_to_str = {0: 'move up', 1: 'move down', 2: 'move left', 3: 'move right'}
    #     str_action1 = act_to_str[action1]
    #     str_action2 = act_to_str[action2]

    #     positions = {
    #         i: (i % self.grid_size, self.grid_size - 1 - (i // self.grid_size))
    #         for i in range(self.grid_size * self.grid_size)
    #     }

    #     agent1_pos = self.xy_pos_to_int(self.agent1_position[0], self.agent1_position[1])
    #     agent2_pos = self.xy_pos_to_int(self.agent2_position[0], self.agent2_position[1])

    #     node_colors = [
    #         'red' if node in {agent1_pos, agent2_pos} 
    #         else self.graph.nodes[node].get('color',
    #             'green' if self.graph.nodes[node].get('visited', False) 
    #             else 'lightblue'
    #         )
    #         for node in self.graph.nodes
    #     ]

    #     self.ax.clear()
    #     nx.draw(
    #         self.graph,
    #         pos=positions,
    #         ax=self.ax,
    #         with_labels=True,
    #         node_size=500,
    #         node_color=node_colors,
    #         font_size=10,
    #         font_weight='bold'
    #     )

    #     self.ax.set_title(f"{str_action1=} | {reward1=}, \n {str_action2=} | {reward2=}")

    #     self.fig.canvas.draw()
    #     self.fig.waitforbuttonpress(timeout=-1)

    #     self.fig.canvas.flush_events()
    #     plt.pause(0.1)


    def _get_observation(self, agent_number) -> int:
        if agent_number == 0:
            agent = self.agent1_position
        else:
            agent = self.agent2_position

        x_pos, y_pos = agent
        agent_node = self.xy_pos_to_int(agent[0], agent[1])

        neighbours = []
        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for move_x, move_y in moves:
            x_new_pos = x_pos + move_x
            y_new_pos = y_pos + move_y
            new_agent_node = x_new_pos * self.grid_size + y_new_pos
            
            if self.graph.has_edge(agent_node, new_agent_node):
                if not self.graph.nodes[new_agent_node]['visited']:
                    neighbours.extend([1, 1]) # Not visited node
                else:
                    neighbours.extend([1, 0]) # Visited node
            else:
                neighbours.extend([0, 0]) # Out of bounds, or obstyckle


        return (agent[0]/self.grid_size, agent[1]/self.grid_size, *neighbours)

        # grid = np.zeros((3, self.grid_size, self.grid_size))
        # for i in range(self.grid_size):
        #     for j in range(self.grid_size):
        #         node_nr = self.xy_pos_to_int(i, j)
        #         if self.graph.nodes[node_nr]['visited']:
        #             grid[0, i ,j] = 1 #visited
        #         elif not list(self.graph.neighbors(node_nr)):
        #             grid[1, i ,j] = 1 #obstyckle
        
        # grid[2, self.agent1_position[0], self.agent1_position[1]] = 1
        # grid[2, self.agent2_position[0], self.agent2_position[1]] = 1
                
        # return grid.flatten()




env = TwoAgentsEnv()
obs_dim = 10 #75
action_dim = 4  # 0 lub 1

from DQNnetwork import DQN
agent1 = DQN(obs_dim, action_dim, )
agent2 = DQN(obs_dim, action_dim, )

episode_rewards1 = []
episode_rewards2 = []
epsilons = []

epsilon_start = 1.0
epsilon_end = 0.00
epsilon_decay = 0.995

for episode in tqdm(range(1000)):
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
        
        agent1.buffer.append((observation1, action1, reward1, next_observation1, done))
        agent2.buffer.append((observation2, action2, reward2, next_observation2, done))
        agent1.update()
        agent2.update()

        total_reward1 += reward1  # Accumulate reward
        total_reward2 += reward2  # Accumulate reward

        observation1 = next_observation1
        observation2 = next_observation2

    episode_rewards1.append(total_reward1)  # After the episode ends
    episode_rewards2.append(total_reward2)  # After the episode ends
    epsilons.append(epsilon)
    epsilon *= 0.995


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
