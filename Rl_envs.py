import gymnasium as gym
from gymnasium import spaces
import numpy as np 
from create_data_structure import make_grid
import matplotlib.pyplot as plt


class MaskedCovargePathPlanningEnv(gym.Env):

    def __init__(self, grid_size: int = 10):
        super(MaskedCovargePathPlanningEnv, self).__init__()

        self.grid_size = grid_size
        
        self.grid = None
        self.masked_grid = None
        self.agent_pos = None
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=2, shape=(grid_size, grid_size)),
            "position": spaces.Discrete(n=grid_size*grid_size, start=0)
        })
        
        # self._action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }
        
    def map_xy_to_int(self, x, y):
        return y + x * self.grid_size


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = make_grid(size=self.grid_size)
        
        started_poin_x, started_poin_y = 0, 0
        self.agent_pos = [started_poin_x, started_poin_y]
        self.grid[started_poin_x, started_poin_y] = 2  # Mark the starting cell as visited
        
        self.masked_grid = -np.ones_like(self.grid, dtype=int)

        self.masked_grid[started_poin_x, started_poin_y] = self.grid[started_poin_x, started_poin_y] # you can see the point you are in 

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = started_poin_x + dx, started_poin_y + dy

            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                self.masked_grid[nx, ny] = self.grid[nx, ny] # you can see the neighbours

        return self._get_observation(), {}
    

    def step(self, action):
        
        x, y = self.agent_pos
        new_x, new_y = x, y
        
        if action == 0 and x > 0:  # Move up
            new_x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Move down
            new_x += 1
        elif action == 2 and y > 0:  # Move left
            new_y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Move right
            new_y += 1
        

        if new_x == x and new_y == y: # move out of bandries 
            reward = -10

        elif self.masked_grid[new_x, new_y] == 1: # move to obstyckle
            reward = -10

        elif self.masked_grid[new_x, new_y] == 2: # move to alredy visited cell 
            reward = -5
            self.agent_pos = [new_x, new_y]
        
        elif self.masked_grid[new_x, new_y] == 0: # move to new visited cell  
            reward = 10     
            self.agent_pos = [new_x, new_y] # marked new visited move 
            self.masked_grid[new_x, new_y] = 2
            self.grid[new_x, new_y] = 2

            # unmasked new visible grids 
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = new_x + dx, new_y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    self.masked_grid[nx, ny] = self.grid[nx, ny] # you can see the neighbours



        if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
            reward = 1000
            done = True
        else:
            done = False
        
        reward = (reward - (-10)) / (1000 - (-10))

        return self._get_observation(), reward, done, False, {} 


    def render(self):
        render_grid = self.masked_grid.copy()
        x, y = self.agent_pos
        render_grid[x, y] = 9
        print(render_grid)
        

    def _get_observation(self):
        
        agent_position = self.map_xy_to_int(self.agent_pos[0], self.agent_pos[1])
        
        observation = {
            "grid": self.masked_grid,
            "position": agent_position
        }
        
        return observation


class SimpleCovargePathPlanningEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=10):
        super(SimpleCovargePathPlanningEnv, self).__init__()

        self.grid_size = grid_size
        self.num_of_steps = 0
        
        self.grid = None
        self.agent_pos = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.visited_states={}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = make_grid() # always the same grid 
        
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
        elif action == 1 and x < self.grid_size - 1:  # Move down
            new_x += 1
        elif action == 2 and y > 0:  # Move left
            new_y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Move right
            new_y += 1
        

        reward = -1
        done = False

        # self.visited_states[(new_x, new_y)] = self.visited_states.get((new_x, new_y), 0) + 1

        if new_x == x and new_y == y: # move out of bandries 
            reward -= 10

        elif self.grid[new_x, new_y] == 1: # move to obstyckle
            reward -= 10

        else: # move
            self.agent_pos = [new_x, new_y] # marked new visited move 
            self.grid[new_x, new_y] = 2

        # if (new_x, new_y) == (9, 9):
        #     reward += 100
        #     done = True
        
        if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
            reward += 1
            done = True

        # # Mechanizm pamięci odwiedzonych stanów
        # elif self.grid[new_x, new_y] == 2: # move to alredy visited cell 
        #     reward -= 1 * self.visited_states[(new_x, new_y)]
        #     self.agent_pos = [new_x, new_y]
        
        # elif self.grid[new_x, new_y] == 0: # move to new visited cell  
        #     reward += 100
        #     self.agent_pos = [new_x, new_y] # marked new visited move 
        #     self.grid[new_x, new_y] = 2


        # if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
        #     reward += 1000
        #     done = True
        # else:
        #     done = False

        return self._get_observation(), reward, done, False, {}


    def render(self, mode="human"):
        
        render_grid = self.grid.copy()
        x, y = self.agent_pos
        render_grid[x, y] = 9
        print(render_grid)
        

    def _get_observation(self):
        return self.agent_pos[1] + self.agent_pos[0] * self.grid_size
