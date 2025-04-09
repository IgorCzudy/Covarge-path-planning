from Rl_agents import TabularQLearningAgent
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
    
    def __init__(self, grid_size: Tuple[int, int] = (5, 5), display: bool = False, map_number: int = 0, first_agent_starting_position = (0,0), secend_agent_starting_position = (4,4)):
        # super(TwoQTable, self).__init__()
        self.map_number = map_number

        self.first_agent_starting_position = first_agent_starting_position
        self.secend_agent_starting_position = secend_agent_starting_position

        self.first_agent_position: Tuple[int, int] = (0, 0)
        self.secend_agent_position: Tuple[int, int] = (4, 4)

        self.grid_width: int = grid_size[0]
        self.grid_height: int = grid_size[1]
        self.num_of_steps: int = 0

        self.grid: Optional[np.dnarray] = None

        self.action_space1: spaces.Discrete = spaces.Discrete(4)
        self.action_space2: spaces.Discrete = spaces.Discrete(4)

        n = self.grid_width * self.grid_height
        self.observation_space1: spaces.Discrete = spaces.Discrete(n)
        self.observation_space2: spaces.Discrete = spaces.Discrete(n)


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

        if self.map_number == 0:
            grid[0, 2] = 1
            grid[2, 3] = 1
            grid[2, 2] = 1
            grid[3, 3] = 1
            grid[3, 2] = 1
            return grid
        
        elif self.map_number == 1:
            grid[1,1] = 1
            grid[1,2] = 1 
            grid[2,1] = 1
            grid[2,2] = 1
            grid[3,3] = 1
            return grid
        
        elif self.map_number == 2:
            grid[1,1] = 1
            grid[1,2] = 1
            grid[1,3] = 1
            grid[2,1] = 1
            grid[2,2] = 1
            grid[2,3] = 1
            grid[3,1] = 1 
            grid[3,2] = 1
            grid[3,3] = 1
            return grid


        elif self.map_number == 3:

            grid[1,1] = 1
            grid[1,2] = 1 
            grid[2,1] = 1
            grid[2,2] = 1
            grid[3,3] = 1
            return grid
        

        elif self.map_number == 4:
            grid[1, 1] = 1
            grid[1, 2] = 1
            grid[2, 2] = 1
            grid[3, 2] = 1
            grid[3, 3] = 1

            return grid
        

        elif self.map_number == 5:
            # grid[0, 6] = 1
            # grid[0, 7] = 1
            # grid[1, 6] = 1
            # grid[1, 7] = 1
            
            # grid[3, 1] = 1
            # grid[3, 2] = 1
            # grid[4, 1] = 1
            # grid[4, 2] = 1

            # grid[5, 5] = 1
            # grid[5, 6] = 1
            # grid[6, 5] = 1
            # grid[6, 6] = 1

            # grid[6, 0] = 1
            # grid[7, 0] = 1
            # grid[7, 1] = 1
            return grid


        elif self.map_number == 6:   
            for i in range(1,6):
                for j in range(1,6):
                    grid[i, j] = 1
            return grid
        
        elif self.map_number == 7:
            for i in range(1,6):
                grid[4, i] = 1
            return grid

        elif self.map_number == 8:
            for i in range(1,6):
                grid[2, i] = 1
                grid[4, i] = 1
            return grid


        elif self.map_number == 9:
            grid[0, 5] = 1
            grid[0, 6] = 1
            grid[1, 5] = 1
            grid[1, 6] = 1
            
            grid[3, 3] = 1
            grid[3, 5] = 1
            grid[4, 3] = 1
            grid[4, 5] = 1

            # grid[5, 5] = 1
            # grid[5, 6] = 1
            # grid[6, 5] = 1
            # grid[6, 6] = 1

            grid[5, 0] = 1
            grid[6, 0] = 1
            grid[6, 1] = 1
            return grid


    def _get_observation(self, agent_number: int) -> int:

        agent_position = self.first_agent_position if agent_number==0 else self.secend_agent_position 
        n = (
            agent_position[1]
            + agent_position[0] * self.grid_height
        )
        # assert 0 <= n <= 24
        return n

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[int, int, Dict]:
        # super().reset(seed=seed)

        self.grid = self.make_grid()
        self.first_agent_position = self.first_agent_starting_position
        self.secend_agent_position = self.secend_agent_starting_position
        self.grid[self.first_agent_position] = 2
        self.grid[self.secend_agent_position] = 3

        return self._get_observation(agent_number=0), self._get_observation(agent_number=1), {}

    def step(self, action1: int, action2: int) -> Tuple[int, int, int, int, bool, bool, Dict]:
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

        return self._get_observation(0), self._get_observation(1), reword[0], reword[1], done, False, {}

        
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


from twoQtableHelper import plot_heatmap, plot_reward, plot_graph, run_learned, format_title, int_to_act, plot_double_q_table

if __name__ == "__main__":
    starting_point1_tuple = (0, 0)
    starting_point1_int = 0 
    starting_point2_tuple = (6, 6) #(6, 6) 
    starting_point2_int = 48 #48

    grid_size = (7, 7)
    episodes = 1550
    render = False
    if_plot_heatmap = False
    map_number = 5
    first_agent_starting_position = starting_point1_tuple
    secend_agent_starting_position = starting_point2_tuple

    agent_params = {
        "number_of_action": 4,
        "number_of_states": grid_size[0] * grid_size[1],
        "γ": 1,
        "α": 0.1,
        "ε": 0.999,
        "α_decay": 0.99,
        "ε_decay": 0.999,
        "α_min": 0.1,
        "ε_min": 0.0,
    }
    agent1 = TabularQLearningAgent(**agent_params)
    agent2 = TabularQLearningAgent(**agent_params)

    if if_plot_heatmap:
        q_hetmap1, q_hetmap2, text_annotations1, text_annotations2 = initialdouble_q_heatmap(agent1, agent2)

    env = TwoQTable(grid_size=grid_size, display=render, map_number=map_number, first_agent_starting_position=first_agent_starting_position, secend_agent_starting_position=secend_agent_starting_position)

    rewards1, rewards2, rewards_sum, εs = [], [], [], []
    for episode in tqdm(range(episodes)):
        obs1, obs2, _ = env.reset()
        done = False
        rew1, rew2, rew_sum =0, 0, 0 
        while not done:
            action1, _ = agent1.get_action(obs1)
            action2, _ = agent2.get_action(obs2) #agent1
            action1 = action1.item()
            action2 = action2.item()

            next_obs1, next_obs2, reward1, reward2, done, _, _ = env.step(action1, action2)
            rew1 += reward1
            rew2 += reward2
            rew_sum += (reward1+ reward2)
            
            εs.append(agent1.ε)
            
            agent1.process_transition(obs1, action1, reward1, next_obs1, done)
            agent2.process_transition(obs2, action2, reward2, next_obs2, done)
            obs1 = next_obs1
            obs2 = next_obs2
        rewords1.append(rew1)
        reowrds2.append(rew2)
        rewords_sum.append(rew_sum)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    import pandas as pd 
    ax1.plot(np.arange(len(pd.Series(rewords1).rolling(50).mean())), 
             pd.Series(rewords1).rolling(50).mean(),
             label="reward of 1 agent",)
     
    ax1.plot(np.arange(len(pd.Series(reowrds2).rolling(50).mean())), 
             pd.Series(reowrds2).rolling(50).mean(),
             label="reward of 2 agent",)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewords')
    ax1.legend()
            
    ax2.plot(np.arange(len(εs)), εs, label='Epsilon')
    ax2.legend()
    ax2.set_xlabel('Moves')
    ax2.set_ylabel('Epsilon')
    plt.subplots_adjust(hspace=0.6)
    plt.show()

    env = TwoQTable(grid_size=(5, 5), display=True, map_number=0)
    obs1, obs2, _ = env.reset()
    done = False
    actions1, actions2 =[], []
    while not done:
        action1, _ = agent1.get_action(obs1)
        action2, _ = agent1.get_action(obs2)
        action1 = action1.item()
        action2 = action2.item()
        actions1.append(action1)
        actions2.append(action2)

        env.render()

        int_to_act = {0: "Move up", 1: "Move down", 2: "Move left", 3: "Move right"}
        print(f"agent1: {int_to_act[action1]}, agent2: {int_to_act[action2]}")
        print(f"obs1: {obs1}, obs2: {obs2}")

        print("Press any key to continue")
        while True:  # waiting for button press
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                print("Key 'n' pressed! Moving to the next iteration.")
                break

        next_obs1, next_obs2, reward1, reward2, done, _, _ = env.step(action1, action2)
        print(f"{reward1=}, {reward2=}")

        agent1.process_transition(obs1, action1, reward1, next_obs1, done)
        agent2.process_transition(obs2, action2, reward2, next_obs2, done)
        obs1 = next_obs1
        obs2 = next_obs2

        
    from Rl_run import make_Q_table_plot, get_new_point_from_action
    make_Q_table_plot(agent1)
    make_Q_table_plot(agent2)
            obs1, obs2 = next_obs1, next_obs2

            if if_plot_heatmap:
                a1, a2 = int_to_act[action1], int_to_act[action2]
                plot_heatmap(
                    agent1, agent2, q_hetmap1, q_hetmap2, text_annotations1, text_annotations2,
                      format_title(agent1, reward1, a1, episode), format_title(agent2, reward2, a2, episode)
                )

        rewards1.append(rew1)
        rewards2.append(rew2)
        rewards_sum.append(rew_sum)

    plot_reward(rewards1, rewards2, εs)

    env = TwoQTable(grid_size=grid_size, display=True, map_number=map_number, first_agent_starting_position=first_agent_starting_position, secend_agent_starting_position=secend_agent_starting_position)
    actions1, actions2 = run_learned(env, agent1, agent2)

    # make_Q_table_plot(agent1)
    # make_Q_table_plot(agent2)
    plot_double_q_table(agent1, agent2)

    plot_graph(env, actions1, actions2, 
               starting_point1_tuple=starting_point1_tuple, 
               starting_point1_int=starting_point1_int, 
               starting_point2_tuple=starting_point2_tuple,
               starting_point2_int=starting_point2_int
               )
