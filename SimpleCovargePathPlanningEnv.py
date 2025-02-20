import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import pygame


class SimpleCovargePathPlanningEnv(gym.Env):

    def __init__(self, grid_size=(5, 5), display = False, starting_point = (0, 0)):
        super(SimpleCovargePathPlanningEnv, self).__init__()

        self.starting_point = starting_point
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
                        1: (0, 0, 0),      # Green for obstacles or paths
                        2: (0, 255, 0),      # Green for visited cell
                        9: (255, 0, 0)       # Red for the agent
                        }
            self.font = pygame.font.Font(None, 30)  # Define the font for numbers

    def draw_grid(self):
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_value = 0 
                pygame.draw.rect(self.window, self.colors[cell_value],
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

                text = self.font.render(f"{y*self.grid_width + x}", True, (0, 0, 0))  # Draw the number (index as example)
                text_rect = text.get_rect(center=(x * self.cell_size + self.cell_size // 2,
                                                  y * self.cell_size + self.cell_size // 2))
                self.window.blit(text, text_rect)


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


    def change_starting_point(self):
        import random
        random.seed(42)
        self.starting_point = tuple(random.choice(np.argwhere(self.grid != 1)))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.make_grid() # always the same grid 
        
        started_poin_x, started_poin_y = self.starting_point[0], self.starting_point[1]
        self.agent_pos = [started_poin_x, started_poin_y]
        self.grid[started_poin_x, started_poin_y] = 2
        
        return self._get_observation(), {}
    

    def step(self, action):
        assert 0 <= action <= 3, "action must be in the range from 0 to 3"

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
        

        done = False
        if new_x == x and new_y == y: # move out of bandries 
            reward = -0.10

        elif self.grid[new_x, new_y] == 1: # move to obstyckle
            reward = -0.10
        
        elif self.grid[new_x, new_y] == 2: # move to alredy visited cell 
            reward = -0.01
            self.agent_pos = [new_x, new_y]
         
        else: # move
            self.agent_pos = [new_x, new_y] # marked new visited move 
            self.grid[new_x, new_y] = 2
            reward = 0.1

        if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
            reward = 1
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

                text = self.font.render(f"{col + row * self.grid_height}", True, (169, 169, 169))  # Draw the number (index as example)
                text_rect = text.get_rect(center=(col * self.cell_size + self.cell_size // 2,
                                                  row * self.cell_size + self.cell_size // 2))
                
                self.window.blit(text, text_rect)


        pygame.display.flip()


    def _get_observation(self):
        return self.agent_pos[1] + self.agent_pos[0] * self.grid_height




from Rl_run import learn_agent, make_Q_table_plot, get_sample_actions_Q_table, plot_mean_reward
from Rl_agents import TabularQLearningAgent


if __name__ == "__main__":


    env = SimpleCovargePathPlanningEnv(grid_size=(5,5),starting_point=(0, 0), display=False)

    agent = TabularQLearningAgent(number_of_action=4, 
                                    number_of_states=25, 
                                    γ=1, 
                                    α=0.1, 
                                    ε=0.99,
                                    α_decay=0.999, 
                                    ε_decay=0.9999, 
                                    α_min=0.0001, 
                                    ε_min=0.01
                                )

    env, agent = learn_agent(env, agent, episodes = 800, plot=True, display_qtable=False, display_pygame=False)


    make_Q_table_plot(agent)
    # test_agent(env, agent, epochs=100)
    
    env = SimpleCovargePathPlanningEnv(grid_size=(5,5), starting_point=(0, 0), display=True)
    actions, path = get_sample_actions_Q_table(env, agent, starting_point=(0, 0))

    from ploting import plot_graph
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(25)])
    plot_graph(graph, path)




    env = SimpleCovargePathPlanningEnv(grid_size=(5,5), starting_point=(4, 4), display=True)
    actions, path = get_sample_actions_Q_table(env, agent, starting_point=(4, 4))

    from ploting import plot_graph
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(25)])
    plot_graph(graph, path)


    env = SimpleCovargePathPlanningEnv(grid_size=(5,5), starting_point=(1, 1), display=True)
    actions, path = get_sample_actions_Q_table(env, agent, starting_point=(1, 1))

    from ploting import plot_graph
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(25)])
    plot_graph(graph, path)

    env = SimpleCovargePathPlanningEnv(grid_size=(5,5), starting_point=(4, 0), display=True)
    actions, path = get_sample_actions_Q_table(env, agent, starting_point=(4, 0))

    from ploting import plot_graph
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(25)])
    plot_graph(graph, path)



    env = SimpleCovargePathPlanningEnv(grid_size=(5,5), starting_point=(0, 3), display=True)
    actions, path = get_sample_actions_Q_table(env, agent, starting_point=(0, 3))

    from ploting import plot_graph
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(25)])
    plot_graph(graph, path)