import gymnasium as gym
from gymnasium import spaces
import numpy as np 
from create_data_structure import make_grid
import pygame
from numpy import uint8


class MaskedCovargePathPlanningEnv(gym.Env):

    def __init__(self, grid_size=(10, 10), display=False):
        super(MaskedCovargePathPlanningEnv, self).__init__()

        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.num_of_steps = 0

        self.grid = None
        self.masked_grid = None
        self.agent_pos = None

        self.action_space = spaces.Discrete(4)

        #masked grid: for undiscaverd 1, empty cell 1, obstyckle 2, visited move 3, position of agent 4 
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.grid_width, self.grid_height), dtype=uint8)
        # self.observation_space = spaces.Dict({
            # "grid": spaces.Box(low=-1, high=2, shape=(grid_size, grid_size), dtype=uint8),
        #     "position": spaces.Discrete(n=grid_size*grid_size, start=0)
        # })

        if display: 
            pygame.init()
            self.cell_size = 50  # Size of each grid cell in pixels
            window_width = self.grid_width * self.cell_size
            window_height = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode((window_width, window_height))
            
            self.colors = {
                        1: (255, 255, 255),  # White for empty cells
                        2: (0, 255, 0),      # Green for obstacles or paths
                        3: (255, 0, 0),       # Red for the agent
                        0: (192, 192, 192)       # for undiscaverd 
                        }
        
        # self._action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }
        
    def map_xy_to_int(self, x, y):
        return y + x * self.grid_height


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = make_grid(size=self.grid_height)
        
        started_poin_x, started_poin_y = 0, 0
        self.agent_pos = [started_poin_x, started_poin_y]
        self.grid[started_poin_x, started_poin_y] = 1  # Mark the starting cell as visited
        
        self.masked_grid = np.zeros_like(self.grid, dtype=int)

        self.masked_grid[started_poin_x, started_poin_y] = self.grid[started_poin_x, started_poin_y] # you can see the point you are in 

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = started_poin_x + dx, started_poin_y + dy

            if 0 <= nx < self.grid_height and 0 <= ny < self.grid_width:
                self.masked_grid[nx, ny] = self.grid[nx, ny] # you can see the neighbours

        return self._get_observation(), {}
    

    def step(self, action):
        
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
        

        # reward = -3
        if new_x == x and new_y == y: # move out of bandries 
            reward = -100

        elif self.masked_grid[new_x, new_y] == 1: # move to obstyckle
            reward = -100

        elif self.masked_grid[new_x, new_y] == 2: # move to alredy visited cell 
            reward = -1
            self.agent_pos = [new_x, new_y]
        
        elif self.masked_grid[new_x, new_y] == 0: # move to new visited cell  
            reward = 10     
            self.agent_pos = [new_x, new_y] # marked new visited move 
            self.masked_grid[new_x, new_y] = 1
            self.grid[new_x, new_y] = 1

            # unmasked new visible grids 
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = new_x + dx, new_y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    self.masked_grid[nx, ny] = self.grid[nx, ny] # you can see the neighbours



        if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
            reward = 100
            done = True
        else:
            done = False
        
        # reward = (reward - (-10)) / (1000 - (-10))

        return self._get_observation(), reward, done, False, {} 


    def render(self):
        render_grid = self.masked_grid.copy()
        x, y = self.agent_pos
        # render_grid[x, y] = 9

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
        
        # agent_position = self.map_xy_to_int(self.agent_pos[0], self.agent_pos[1])
        
        # observation = {
        #     "grid": self.masked_grid,
        #     "position": agent_position
        # }
        
        # return observation
        observation_arr = self.masked_grid.copy()
        observation_arr[self.agent_pos] = 3
        return observation_arr
