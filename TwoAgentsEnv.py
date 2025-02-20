import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import pygame


class TwoAgentsEnv(gym.Env):

    def __init__(self,grid_size=(5, 5), display = False, mlflow=True):
        super(TwoAgentsEnv, self).__init__()

        self.mlflow = mlflow
        
        self.first_agent_position = (0,0)
        self.secend_agent_position = (0,4)

        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.num_of_steps = 0
        
        self.grid = None
        
        self.action_space = spaces.Discrete(16)
        n = self.grid_width * self.grid_height
        self.observation_space = spaces.Discrete(n * n)


        if display: 
            pygame.init()
            self.cell_size = 50  # Size of each grid cell in pixels
            window_width = self.grid_width * self.cell_size
            window_height = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode((window_width, window_height))
            
            self.colors = {
                        0: (255, 255, 255),  # White for empty cells
                        1: (0, 0, 0),      # Green for obstacles or paths
                        2: (0, 255, 0),      # Green for visited cell
                        9: (255, 0, 0),       # Red for the agent
                        8: (190, 0, 0)       # lighter red for the secend agent
                        }
            self.font = pygame.font.Font(None, 30)  # Define the font for numbers

        # if pygame:

            
    def make_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_width, self.grid_height), dtype=int)

        # obstycle 
        grid[0,2] = 1
        grid[2,3] = 1
        grid[2,2] = 1
        grid[3,3] = 1
        grid[3,2] = 1

        if self.grid_height > 6 and self.grid_with > 6:
            grid[6,5] = 1
            grid[5,6] = 1
            grid[5,5] = 1
            grid[6,6] = 1

            grid[4,2] = 1
            grid[5,2] = 1
        return grid
    
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.make_grid() # always the same grid 

        x1, y1 = self.first_agent_position
        self.grid[x1, y1] = 2
        x2, y2 = self.secend_agent_position
        self.grid[x2, y2] = 2

        return self._get_observation(), {}
    
    def step(self, action):
        assert 0 <= action <= 15, "action must be in the range from 0 to 15"
        
        self.num_of_steps+=1
        x1, y1 = self.first_agent_position
        x2, y2 = self.secend_agent_position
        
        # 16%4 -> first agent action
        # 16//4 -> secend agent action
        # action:0 up
        # action:1 down
        # action:2 left
        # action:3 right

        first_agent_action = action % 4
        secend_agent_action = action // 4

        def move(agent_action, x_ref, y_ref):
            x = int(x_ref)
            y = int(y_ref)
            if agent_action == 0 and x > 0:  # Move up
                x -= 1
            elif agent_action == 1 and x < self.grid_width - 1:  # Move down
                x += 1
            elif agent_action == 2 and y > 0:  # Move left
                y -= 1
            elif agent_action == 3 and y < self.grid_height - 1:  # Move right
                y += 1
            return x, y
        
        # Apply movements
        new_x1, new_y1 = move(first_agent_action, x1, y1)
        new_x2, new_y2 = move(secend_agent_action, x2, y2)

        # TODO Add that agents dont crouch eoch other 
        def evaluate_movement(x, y, new_x, new_y):
            if (new_x, new_y) == (x, y):  # Out of bounds (no movement)
                return -0.1, x, y
            elif self.grid[new_x, new_y] == 1:  # Obstacle
                return -0.1, x, y
            elif self.grid[new_x, new_y] == 2:  # Visited cell
                return -0.5, new_x, new_y    
            else:  # New valid move
                self.grid[new_x, new_y] = 2  # Mark as visited
                return 0.5, new_x, new_y
            
        reward1, _x1, _y1 = evaluate_movement(x1, y1, new_x1, new_y1)
        reward2, _x2, _y2 = evaluate_movement(x2, y2, new_x2, new_y2)

        # if _x1 == _x2 and _y1 == _y2:#both agents choose the same cell, they crached each other  
        #     reward1, reward2 = -0.10, -0.10
        # else: 
        self.first_agent_position = (_x1, _y1)
        self.secend_agent_position = (_x2, _y2)

        done= False
        if np.all((self.grid == 2) | (self.grid == 1)):
            reward1, reward2 = 0.5, 0.5 
            done = True
        # print(reward1, reward2)
        # total_reward = min([reward1, reward2])
        total_reward = reward1 + reward2

        return self._get_observation(), total_reward, done, False, {}



    def render(self, mode="human"):
        render_grid = self.grid.copy()
        x1, y1 = self.first_agent_position
        x2, y2 = self.secend_agent_position
        render_grid[x1, y1] = 9
        render_grid[x2, y2] = 8

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                import sys; sys.exit()


        self.window.fill((0, 0, 0))  # Black background

        for row in range(self.grid_width):
            for col in range(self.grid_height):
                value = render_grid[row, col]
                color = self.colors.get(value, (0, 0, 0))  # Default to black for unknown values
                pygame.draw.rect(
                    self.window,
                    color,
                    (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                )

                pygame.draw.rect(
                    self.window,
                    (0, 0, 0),
                    (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size),
                    1
                )
                text = self.font.render(f"{col + row * self.grid_height}", True, (169, 169, 169))  # Draw the number (index as example)
                text_rect = text.get_rect(center=(col * self.cell_size + self.cell_size // 2,
                                                  row * self.cell_size + self.cell_size // 2))
                
                self.window.blit(text, text_rect)
        pygame.display.flip()

    

    def _get_observation(self):

        n_first = self.first_agent_position[1] + self.first_agent_position[0] * self.grid_height
        n_secend = self.secend_agent_position[1] + self.secend_agent_position[0] * self.grid_height

        assert 0 <= n_first <= 24
        assert 0 <= n_secend <= 24
        out = n_first + n_secend * (self.grid_height*self.grid_width)
        assert 0 <= out <= 624
        return out



from Rl_run import learn_agent, make_Q_table_plot, get_sample_actions_Q_table, plot_mean_reward
from Rl_agents import TabularQLearningAgent
import mlflow

if __name__ == "__main__":

    env = TwoAgentsEnv(display=False, mlflow=True)

    agent = TabularQLearningAgent(number_of_action=16, 
                                number_of_states=25*25, 
                                γ=1.0,
                                α=0.1, 
                                ε=0.99,
                                α_decay=0.999, 
                                ε_decay=0.99999, 
                                α_min=0.1, 
                                ε_min=0.0005
                                )


    env, agent = learn_agent(env, agent, episodes = 1000, plot=False, display_qtable=False, display_pygame=False, mlflow=True)


    env = TwoAgentsEnv(display=True, mlflow=True)
    obs, _ = env.reset()
    
    done = False
    np.random.seed(42)
    i = 0
    while not done:
        i+=1
        action = agent.get_action(obs)[0]
        action = action.item()
        obs, reward, done, _, _ = env.step(action)
        print(f"{obs=} {reward=}")
        env.render()
        import time; time.sleep(0.5)
        if i >1000:
            break 
    
    # import matplotlib.pyplot as plt
    
    # fig, ax = plt.subplots(figsize=(4, 15))  # Set the figure size
    # ax.imshow(agent.Q, cmap='hot', aspect='auto', interpolation='nearest')
    # plt.title('Q-values Heatmap')
    # plt.show()

    # from ploting import plot_graph
    # import networkx as nx
    # graph = nx.Graph()
    # graph.add_nodes_from([i for i in range(25)])
    # plot_graph(graph, path)


    # obs, _ = env.reset()
    
    # done = False
    # np.random.seed(42)
    # while not done:
    #     action = np.random.randint(16)
    #     next_obs, reward, done, _, _ = env.step(action)
    #     env.render()
    #     import time; time.sleep(0.1)
        