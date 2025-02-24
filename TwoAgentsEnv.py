import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import pygame
from torch.utils.tensorboard import SummaryWriter
import torch 
import click

class TwoAgentsEnv(gym.Env):

    def __init__(self,grid_size=(5, 5), display = False, tensorb=True):
        super(TwoAgentsEnv, self).__init__()

        self.tensorb = tensorb
        
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
            self.window_width = self.grid_width * self.cell_size
            self.window_height = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            
            self.colors = {
                        0: (255, 255, 255),  # White for empty cells
                        1: (0, 0, 0),      # Green for obstacles or paths
                        2: (0, 255, 0),      # Green for visited cell
                        9: (255, 0, 0),       # Red for the agent
                        8: (190, 0, 0)       # lighter red for the secend agent
                        }
            self.font = pygame.font.Font(None, 30)  # Define the font for numbers

        if display and tensorb:
            self.frames = []

            
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
        
        if self.tensorb:
            print("Writing frame to video")
            frame = pygame.surfarray.array3d(self.window)
            frame = np.flip(frame, axis=1)
            frame = np.rot90(frame)            
            self.frames.append(frame)

        pygame.display.flip()


    def _get_observation(self):

        n_first = self.first_agent_position[1] + self.first_agent_position[0] * self.grid_height
        n_secend = self.secend_agent_position[1] + self.secend_agent_position[0] * self.grid_height

        assert 0 <= n_first <= 24
        assert 0 <= n_secend <= 24
        out = n_first + n_secend * (self.grid_height*self.grid_width)
        assert 0 <= out <= 624
        return out
    
    def get_video_tensor(self):
        video = np.array(self.frames)
        video = np.expand_dims(video, axis=0)
        video = video.astype(np.uint8)
        video = torch.from_numpy(video)
        video_transposed = video.permute(0, 1, 4, 2, 3)
        return video_transposed



from Rl_run import learn_agent, make_Q_table_plot, get_sample_actions_Q_table, plot_mean_reward
from Rl_agents import TabularQLearningAgent

@click.command()
@click.option("--gamma", type=float, default=1.0, help="Discount factor for future rewards")
@click.option("--alpha", type=float, default=0.1, help="Learning rate")
@click.option("--epsilon", type=float, default=0.99, help="Exploration rate")
@click.option("--alpha_decay", type=float, default=0.999, help="Decay rate for alpha")
@click.option("--epsilon_decay", type=float, default=0.99999, help="Decay rate for epsilon")
@click.option("--alpha_min", type=float, default=0.1, help="Minimum value for alpha")
@click.option("--epsilon_min", type=float, default=0.0005, help="Minimum value for epsilon")
@click.option("--episodes", type=int, default=500, help="Number of episodes for training")
def main(gamma, alpha, epsilon, alpha_decay, epsilon_decay, alpha_min, epsilon_min, episodes):
    writer = SummaryWriter(f"runs/two_agent_episodes={episodes}_gamma={gamma}_alpha={alpha}_epsilon={epsilon}_alpha_decay={alpha_decay}_epsilon_decay={epsilon_decay}_alpha_min={alpha_min}_epsilon_min={epsilon_min}")
    
    number_of_action=16
    number_of_states=25*25
    writer.add_hparams({"number_of_action": number_of_action,
                        "number_of_states": number_of_states,
                        "γ": gamma,        # Use "gamma" instead of "γ"
                        "α": alpha,        # Use "alpha" instead of "α"
                        "ε": epsilon,     # Use "epsilon" instead of "ε"
                        "α_decay": alpha_decay,
                        "ε_decay": epsilon_decay,
                        "α_min": alpha_min,
                        "ε_min": epsilon_min,
                        "episodes": episodes},
                        {})


    env = TwoAgentsEnv(display=True, tensorb=True)
    agent = TabularQLearningAgent(number_of_action=number_of_action, 
                                number_of_states=number_of_states, 
                                γ=gamma,
                                α=alpha, 
                                ε=epsilon,
                                α_decay=alpha_decay, 
                                ε_decay=epsilon_decay, 
                                α_min=alpha_min, 
                                ε_min=epsilon_min
                                )

    env, agent = learn_agent(env, agent, writer, episodes = episodes, plot=False, display_qtable=False, display_pygame=True)


    env = TwoAgentsEnv(display=True, tensorb=True)
    obs, _ = env.reset()    
    done = False
    np.random.seed(42)
    step = 0
    while not done:
        step += 1
        action = agent.get_action(obs)[0]
        action = action.item()
        obs, reward, done, _, _ = env.step(action)
        print(f"{obs=} {reward=}")
        env.render()
        video_tensor = env.get_video_tensor()
        last_frame = video_tensor[-1, -1]
        writer.add_image("TwoAgents Simulation", last_frame, global_step=step)

        if step >80:
            break 

    video_tensor = env.get_video_tensor()
    writer.add_video("Full TwoAgents Simulation", video_tensor, fps=1)
    writer.close()

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
    

if __name__ == "__main__":
    main()
