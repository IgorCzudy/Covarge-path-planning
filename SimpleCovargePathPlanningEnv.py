import gymnasium as gym
from gymnasium import spaces
import numpy as np 
from create_data_structure import make_grid
import pygame


class SimpleCovargePathPlanningEnv(gym.Env):

    def __init__(self, grid_size=(10, 10), display = False):
        super(SimpleCovargePathPlanningEnv, self).__init__()

        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.num_of_steps = 0
        
        self.grid = None
        self.agent_pos = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_width * self.grid_height)


        if display: 
            pygame.init()
            self.cell_size = 50  # Size of each grid cell in pixels
            window_width = self.grid_width * self.cell_size
            window_height = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode((window_width, window_height))
            
            self.colors = {
                        0: (255, 255, 255),  # White for empty cells
                        1: (0, 255, 0),      # Green for obstacles or paths
                        9: (255, 0, 0)       # Red for the agent
                        }

    def make_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_width, self.grid_height), dtype=int)

        grid[4,5] = 1
        grid[5,5] = 1

        grid[8,8] = 1
        grid[9,9] = 1

        grid[2,8] = 1
        grid[2,7] = 1
        grid[1,8] = 1
        grid[1,7] = 1

        grid[5,5] = 1
        grid[6,6] = 1
        grid[4,4] = 1
        return grid


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.make_grid() # always the same grid 
        
        started_poin_x, started_poin_y = 0, 0
        self.agent_pos = [started_poin_x, started_poin_y]
        self.grid[0, 0] = 2
        
        return self._get_observation(), {}
    

    def step(self, action):
        self.num_of_steps+=1
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if action == 0 and x > 0:  # Move up
            new_x -= 1
        elif action == 1 and x < self.grid_width - 1:  # Move down
            new_x += 1
        elif action == 2 and y > 0:  # Move left
            new_y -= 1
        elif action == 3 and y < self.grid_height - 1:  # Move right
            new_y += 1
        

        reward = -1
        done = False

        if new_x == x and new_y == y: # move out of bandries 
            reward -= 10

        elif self.grid[new_x, new_y] == 1: # move to obstyckle
            reward -= 10

        else: # move
            self.agent_pos = [new_x, new_y] # marked new visited move 
            self.grid[new_x, new_y] = 2
            reward += 10

        if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
            reward += 100
            done = True

        return self._get_observation(), reward, done, False, {}


    def render(self, mode="human"):
        render_grid = self.grid.copy()
        x, y = self.agent_pos
        render_grid[x, y] = 9

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
        pygame.display.flip()


    def _get_observation(self):
        return self.agent_pos[1] + self.agent_pos[0] * self.grid_height




from Rl_run import learn_agent, make_Q_table_plot, get_sample_actions_Q_table, plot_mean_reward
from Rl_agents import TabularQLearningAgent
from run_and_plot_env import test_agent

if __name__ == "__main__":
    env = SimpleCovargePathPlanningEnv()

    agent = TabularQLearningAgent(number_of_action=4, 
                                  number_of_states=100, 
                                  γ=1, 
                                  α=0.3, 
                                  ε=0.7,
                                  α_decay=0.999, 
                                  ε_decay=0.999, 
                                  α_min=0, 
                                  ε_min=0)

    env, agent = learn_agent(env, agent, episodes = 100)

    make_Q_table_plot(agent)
    test_agent(env, agent, epochs=100)
    
    env = SimpleCovargePathPlanningEnv(display=True)
    actions, path = get_sample_actions_Q_table(env, agent)
