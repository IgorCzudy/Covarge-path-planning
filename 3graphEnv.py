from create_data_structure import create_networkX_graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from typing import Tuple, List, Dict, Optional
import random
from tqdm import tqdm
from Rl_agents import TabularQLearningAgent


class TwoAgentsEnv(gym.Env):

    def __init__(self, grid_size: int, graph_number:int = 0, plot=False):
        super(TwoAgentsEnv, self).__init__()
        self.agent1_position: Tuple[int, int] = (0, 0)
        self.agent2_position: Tuple[int, int] = (9, 9)
        self.agent3_position: Tuple[int, int] = (0, 9)

        self.graph_number = graph_number
        self.grid_size = grid_size
        self.num_of_steps = 0
        if plot:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
        

    def make_graph(self) -> nx.Graph:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # if self.graph_number == 2:
        self.start_agent1_position = (0, 0)
        self.start_agent2_position = (9, 9)
        self.start_agent3_position = (0, 9)

        grid[0, 4] = 1
        grid[1, 4] = 1
        grid[2, 4] = 1
        grid[0, 5] = 1
        grid[1, 5] = 1
        grid[2, 5] = 1
        grid[0, 6] = 1
        grid[1, 6] = 1
        grid[2, 6] = 1
        
        grid[4, 0] = 1
        grid[4, 1] = 1
        grid[4, 2] = 1
        grid[5, 0] = 1
        grid[5, 1] = 1
        grid[5, 2] = 1
        grid[6, 0] = 1
        grid[6, 1] = 1
        grid[6, 2] = 1
        

        grid[6, 6] = 1
        grid[6, 7] = 1
        grid[6, 8] = 1
        grid[6, 9] = 1
        grid[7, 6] = 1
        grid[7, 7] = 1
        grid[7, 8] = 1
        grid[7, 9] = 1

        grid[6, 3] = 1
        grid[6, 4] = 1
        grid[6, 5] = 1
        grid[3, 6] = 1
        grid[4, 6] = 1
        grid[5, 6] = 1



        graph, self.not_obstacles = create_networkX_graph(grid)
        nx.set_node_attributes(graph, False, 'visited')

        return graph
    

    def xy_pos_to_int(self, x: int, y: int) -> int:
        return x * self.grid_size + y


    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[int, int, Dict]:
        super().reset(seed=seed)
        self.graph = self.make_graph()
        self.agent1_position = self.start_agent1_position
        self.agent2_position = self.start_agent2_position
        self.agent3_position = self.start_agent3_position


        for agent in [self.agent1_position, self.agent2_position, self.agent3_position]:
            self.graph.nodes[self.xy_pos_to_int(agent[0], agent[1])]['visited'] = True
        
        return self._get_observation(agent_number=0), self._get_observation(agent_number=1), self._get_observation(agent_number=2), {}


    def step(self, action1: int, action2: int, action3: int) -> Tuple[int, int, int, int, bool, bool, Dict]:
        assert 0 <= action1 <= 3, "action must be in the range from 0 to 3"
        assert 0 <= action2 <= 3, "action must be in the range from 0 to 3"    
        self.num_of_steps += 1

        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        rewards = [0, 0, 0]
        agents = [self.agent1_position, self.agent2_position, self.agent3_position] 
        actions = [action1, action2, action3]

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
                elif i == 1: 
                    self.agent2_position = new_x, new_y
                else:
                    self.agent3_position = new_x, new_y
                rewards[i] -= 0.05

            else:  # New valid move
                self.graph.nodes[new_node_number]['visited'] = True
                if i == 0: 
                    self.agent1_position = new_x, new_y
                elif i == 1: 
                    self.agent2_position = new_x, new_y
                else:
                    self.agent3_position = new_x, new_y
                rewards[i] += 0.1

        #collision panelty 
        # if self.agent1_position == self.agent2_position:
        #     rewards[0] -= 0.1
        #     rewards[1] -= 0.1

        # if self.agent1_position == self.agent3_position:
        #     rewards[0] -= 0.1
        #     rewards[2] -= 0.1

        # if self.agent2_position == self.agent3_position:
        #     rewards[1] -= 0.1
        #     rewards[2] -= 0.1


        done = False
        if all([self.graph.nodes[node]['visited'] for node in self.not_obstacles]):
            rewards[0] = 0.5
            rewards[1] = 0.5
            rewards[2] = 0.5
            done = True

        return self._get_observation(agent_number=0), self._get_observation(agent_number=1), self._get_observation(agent_number=2), rewards[0], rewards[1], rewards[2], done, _, _ 


    def render(self, action1: int, reward1: float, action2: int, reward2: float, action3: int, reward3: float, mode="human") -> None:
        act_to_str = {0: 'move up', 1: 'move down', 2: 'move left', 3: 'move right'}
        str_action1 = act_to_str[action1]
        str_action2 = act_to_str[action2]
        str_action3 = act_to_str[action3]

        positions = {
            i: (i % self.grid_size, self.grid_size - 1 - (i // self.grid_size))
            for i in range(self.grid_size * self.grid_size)
        }

        agent1_pos = self.xy_pos_to_int(self.agent1_position[0], self.agent1_position[1])
        agent2_pos = self.xy_pos_to_int(self.agent2_position[0], self.agent2_position[1])
        agent3_pos = self.xy_pos_to_int(self.agent3_position[0], self.agent3_position[1])


        node_colors = []
        for node in self.graph.nodes:
            if node in {agent1_pos, agent2_pos, agent3_pos}:
                node_colors.append('red')
            else:
                if self.graph.nodes[node].get('visited', False):
                    node_colors.append('green')
                else:
                    node_colors.append(self.graph.nodes[node].get('color', 'lightblue'))


        self.ax.clear()
        nx.draw(
            self.graph,
            pos=positions,
            ax=self.ax,
            with_labels=True,
            node_size=500,
            node_color=node_colors,
            font_size=10,
            font_weight='bold'
        )

        self.ax.set_title(f"{str_action1=} | {reward1=}, \n {str_action2=} | {reward2=},  \n {str_action3=} | {reward3=}")

        self.fig.canvas.draw()
        self.fig.waitforbuttonpress(timeout=-1)

        self.fig.canvas.flush_events()
        plt.pause(0.1)


    def _get_observation(self, agent_number) -> int:
        if agent_number == 0:
            agent = self.agent1_position
        elif agent_number == 1:
            agent = self.agent2_position
        else:
            agent = self.agent3_position
        return self.xy_pos_to_int(agent[0], agent[1])


    def _get_observation2(self, agent_number) -> int:
        if agent_number == 0:
            agent = self.agent1_position
        elif agent_number == 1:
            agent = self.agent2_position
        else:
            agent = self.agent3_position

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
                    neighbours.append(1) #1
                else:
                    neighbours.append(0) #0
            else:
                neighbours.append(0) # Out of bounds, or obstyckle

        bitmask = 0
        for i, visited in enumerate(neighbours):
            bitmask |= visited << i  # Bit 0: up, bit 1: down, etc.
        return agent_node * 16 + bitmask
    


    def _get_observation2(self, agent_number):
        # Bit 0: Up neighbor is unvisited
        # Bit 1: Down neighbor is unvisited
        # Bit 2: Left neighbor is unvisited
        # Bit 3: Right neighbor is unvisited
        # Collision Detection (1 bit)
        # Bit 4: Agents are in same node
        if agent_number == 0:
            agent_pos = self.agent1_position
            opponent_pos1 = self.agent2_position
            opponent_pos2 = self.agent3_position
        elif agent_number == 1:
            agent_pos = self.agent2_position
            opponent_pos1 = self.agent1_position
            opponent_pos2 = self.agent3_position
        else:
            agent_pos = self.agent3_position
            opponent_pos1 = self.agent1_position
            opponent_pos2 = self.agent2_position
          

        current_node = self.xy_pos_to_int(*agent_pos)
        opponent_node1 = self.xy_pos_to_int(*opponent_pos1)
        opponent_node2 = self.xy_pos_to_int(*opponent_pos2)
        
        x, y = agent_pos
        moves = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
        bitmask = 0
        
        for i, (dx, dy) in enumerate(moves):
            new_x = x + dx
            new_y = y + dy
            neighbor_node = self.xy_pos_to_int(new_x, new_y)
            
            if self.graph.has_edge(current_node, neighbor_node):
                if not self.graph.nodes[neighbor_node]['visited']:
                    bitmask |= 1 << i  # Set bit if neighbor is unvisited and accessible
        
        # 2. Add collision detection bit (1 bit)
        collision_bit = 1 if current_node == opponent_node1 or current_node == opponent_node2 else 0
        
        return (current_node * 32) + (bitmask << 1) + collision_bit


def plot_rewards(rewards1, rewards2, rewards3, eps, roll):
    fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(10, 8))

    axs[0].margins(x=0)
    axs[0].plot(rewards1, label='Agent 1 (Instant)', alpha=0.3)
    axs[0].plot(pd.Series(rewards1).rolling(roll).mean(), label='Agent 1 (Moving Avg)', color='blue')
    axs[0].plot(rewards2, label='Agent 2 (Instant)', alpha=0.3)
    axs[0].plot(pd.Series(rewards2).rolling(roll).mean(), label='Agent 2 (Moving Avg)', color='orange')
    axs[0].plot(rewards3, label='Agent 3 (Instant)', alpha=0.3)
    axs[0].plot(pd.Series(rewards3).rolling(roll).mean(), label='Agent 3 (Moving Avg)', color='black')
    axs[0].set_title('Reward History (Smoothed with {}-Episode Window)'.format(roll))
    axs[0].set_ylabel('Reward')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(eps, label='Exploration Rate (ε)', color='k')
    axs[1].set_title('Exploration-Exploitation Tradeoff')
    axs[1].set_xlabel('Training Episodes')
    axs[1].set_ylabel('ε Value')
    axs[1].legend(loc='upper right')
    axs[1].grid(False)
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    starting_point1_tuple = (0, 0)
    starting_point1_int = 0
    starting_point2_tuple = (9, 9)
    starting_point2_int = 80
    starting_point3_tuple = (0, 9)
    starting_point3_int = 8

    grid_size = 10
    num_states = grid_size**2 * 16
    num_actions = 4
    
    gaph_number = 3

    agent_params = {
        "number_of_action": num_actions,
        "number_of_states": num_states,
        "γ": 0.99,  # Slightly discount future rewards
        "α": 0.9,
        "ε": 0.9999,  # Start with full exploration
        "α_decay": 0.995,  # Slower alpha decay
        "ε_decay": 0.99999,  # Faster epsilon decay
        "α_min": 0.1,
        "ε_min": 0.01,  # Maintain minimal exploration
    }
    agent1 = TabularQLearningAgent(**agent_params)
    agent2 = TabularQLearningAgent(**agent_params)
    agent3 = TabularQLearningAgent(**agent_params)

    env = TwoAgentsEnv(grid_size, gaph_number)

    rewards1 = []
    rewards2 = []
    rewards3 = []
    eps = []
    for episode in tqdm(range(4000)):
        obs1, obs2, obs3, _ = env.reset()
        
        done = False
        total_reward1 = 0
        total_reward2 = 0
        total_reward3 = 0
        while not done:
            action1, _ = agent1.get_action(obs1)
            action1 = action1.item()
            
            action2, _ = agent2.get_action(obs2)
            action2 = action2.item()

            action3, _ = agent3.get_action(obs3)
            action3 = action3.item()
            

            next_obs1, next_obs2, next_obs3, reward1, reward2, reward3, done, _, _ = env.step(action1, action2, action3)
            agent1.process_transition(obs1, action1, reward1, next_obs1, done)
            agent2.process_transition(obs2, action2, reward2, next_obs2, done)
            agent3.process_transition(obs3, action3, reward3, next_obs3, done)
            
            total_reward1 += reward1
            total_reward2 += reward2
            total_reward3 += reward3
            eps.append(agent1.ε)
            obs1 = next_obs1
            obs2 = next_obs2
            obs3 = next_obs3


        rewards1.append(total_reward1)
        rewards2.append(total_reward2)
        rewards3.append(total_reward3)


    plot_rewards(rewards1, rewards2, rewards3, eps, 30)

    env = TwoAgentsEnv(grid_size, gaph_number, plot=True)
    obs1, obs2, obs3, _ = env.reset()
    actions1, actions2, actions3 = [], [], []
    done = False
    while not done:
        action1, _ = agent1.get_action(obs1)
        action1 = action1.item()
        action2, _ = agent2.get_action(obs2)
        action2 = action2.item()
        action3, _ = agent3.get_action(obs3)
        action3 = action3.item()

        actions1.append(action1)
        actions2.append(action2)
        actions3.append(action3)

        next_obs1, next_obs2, next_obs3, reward1, reward2, reward3, done, _, _ = env.step(action1, action2, action3)
        agent1.process_transition(obs1, action1, reward1, next_obs1, done)
        agent2.process_transition(obs2, action2, reward2, next_obs2, done)
        agent3.process_transition(obs3, action3, reward3, next_obs3, done)

        env.render(action1, reward1, action2, reward2, action3, reward3)
        obs1 = next_obs1
        obs2 = next_obs2
        obs3 = next_obs3


    from twoQtableHelper import plot_graph3
    plot_graph3(env, actions1, actions2, actions3, 
               starting_point1_tuple=starting_point1_tuple, 
               starting_point1_int=starting_point1_int, 
               starting_point2_tuple=starting_point2_tuple,
               starting_point2_int=starting_point2_int, 
               starting_point3_tuple=starting_point3_tuple,
               starting_point3_int=starting_point3_int
               )
