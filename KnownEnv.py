import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from numpy import uint8
import time
from stable_baselines3 import A2C
from Rl_agents import TabularQLearningAgent 
from DeepQlearningAgent import DeepQlearningAgent
from Rl_run import learn_agent, plot_rewards
from tqdm import tqdm


class KnownEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=(10,10), display=False):
        super(KnownEnv, self).__init__()


        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.num_of_steps = 0

        self.grid = None
        self.agent_pos = None

        self.action_space = spaces.Discrete(4)

        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_width*self.grid_height + 1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=2, 
                shape=[self.grid_width, self.grid_height],
                dtype = uint8) ,
            "position": spaces.MultiDiscrete([self.grid_width, self.grid_height])
        })

        if display:
            pygame.init()
            self.cell_size = 50
            window_width = self.grid_width * self.cell_size
            window_height = self.grid_height * self.cell_size
            self.window = pygame.display.set_mode((window_width, window_height))
            
            self.colors = {
                        0: (255, 255, 255),  # White for empty cells
                        1: (0, 0, 0), # Black for obstycle
                        2: (0, 255, 0),      # Green for visited cell
                        #(255, 0, 0),       # Red for the agent
                        }
            
            
    def map_xy_to_int(self, x, y):
        return y + x * self.grid_height
    
    def make_grid(self):
        grid = np.zeros((self.grid_width, self.grid_height), dtype=uint8)

        # obstycle 
        grid[0,2] = 1
        grid[2,3] = 1
        grid[2,2] = 1
        grid[3,3] = 1
        # grid[5,5] = 1

        # grid[8,8] = 1
        # grid[9,9] = 1

        # grid[2,8] = 1
        # grid[2,7] = 1
        # grid[1,8] = 1
        # grid[1,7] = 1

        # grid[5,5] = 1
        # grid[6,6] = 1
        # grid[4,4] = 1

        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.make_grid()

        started_poin_x, started_poin_y = 0, 0
        self.agent_pos = [started_poin_x, started_poin_y]
        self.grid[started_poin_x, started_poin_y] = 2  # Mark the starting cell as visited

        return self._get_observation(), {}

    def _get_observation(self):
        # position = self.map_xy_to_int(self.agent_pos[0], self.agent_pos[1])
        
        return {
            "grid": self.grid,
            "position": np.array([self.agent_pos[0], self.agent_pos[1]])
        }

        # return np.append(self.grid.flatten()/3, position/99) 
    
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


        if new_x == x and new_y == y: # move out of bandries 
            reward = -10
            

        elif self.grid[new_x, new_y] == 1: # move to obstyckle
            reward = -10
            

        elif self.grid[new_x, new_y] == 2: # move to alredy visited cell 
            reward = -5
            self.agent_pos = [new_x, new_y]
            

        elif self.grid[new_x, new_y] == 0: # move to new visited cell  
            reward = 5
            self.agent_pos = [new_x, new_y]
            self.grid[new_x, new_y] = 2
            
        
        if np.all((self.grid == 2) | (self.grid == 1)): # if all cells that are not obstyckle ware visited
            reward = 100
            done = True
        else:
            done = False
        
        reward = (reward - (-10)) / (100 - (-10))
        return self._get_observation(), reward, done, False, {}
    

    def render(self, mode="human"):
        render_grid = self.grid.copy()
        x, y = self.agent_pos

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                import sys; sys.exit()

        self.window.fill((0, 0, 0))  # Black background

        for row in range(self.grid_width):
            for col in range(self.grid_height):
                value = render_grid[row, col]
                color = self.colors.get(value, (255, 222, 33))  # Default to yellow for unknown values
                
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

                # red for an agent
                if row == x and col ==y:
                    pygame.draw.rect(
                    self.window,
                    (255, 0, 0),
                    (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                    )

        pygame.display.flip()


if __name__ == "__main__":
    episodes = 10_000

    
    knownEnv = KnownEnv(grid_size=(5,5), display=True)
    number_of_states = 5*5 + 2 # position as x, y 
    number_of_action = 4
    
    # agent = DeepQlearningAgent( number_of_action = number_of_action, 
    #                             number_of_states = number_of_states, 
    #                             γ=0.99, 
    #                             α=1e-2, α_decay=0.9999, l2=1e-5,
    #                             ε=0.7, ε_decay=0.999999, ε_min=0.1,
    #                             hidden=[256, 64],
    #                             double_dqn=False,
    #                             is_replayMemory=False, replayMemory_size=100,
    #                             mini_batch_size=16)

    # rewards = []
    # eps = []
    # for _ in tqdm(range(episodes)):
    #     obs, _ = knownEnv.reset()
    #     obs = np.append(obs['grid'].flatten() / 3, obs['position'] / 10)
        
    #     total_reward = 0
    #     done = False
    #     while not done:
    #         action, _ = agent.get_action(obs)
    #         next_obs, reward, done, _, _ = knownEnv.step(action)
    #         next_obs = np.append(next_obs['grid'].flatten() / 3, next_obs['position'] / 10)
            
    #         total_reward += reward

    #         agent.process_transition(obs, action, reward, next_obs, done)
    #         obs = next_obs

    #     rewards.append(total_reward)
    #     eps.append(agent.ε)

    # plot_rewards(rewards, eps, 50)
    # agent.save_agent(path="models/untill_full_covarge")


    # knownEnv = KnownEnv(display=True)
    # obs, _ = knownEnv.reset()
    # obs = np.append(obs['grid'].flatten() / 3, obs['position'] / 10)

    # done = False
    # while not done:
    #     action, _ = agent.get_action(obs, learning=False)
    #     knownEnv.render()
    #     next_obs, reward, done, _, _ = knownEnv.step(action)
    #     next_obs = np.append(next_obs['grid'].flatten(), next_obs['position'])
    #     obs = next_obs


    # model = DeepQlearningAgent.agent_load(path="models/untill_full_covarge")
    # knownEnv = KnownEnv(display=True)
    # obs, _ = knownEnv.reset()
    # obs = np.append(obs['grid'].flatten() / 3, obs['position'] / 10)
    # done = False
    # while not done:
    #     action, _ = DeepQlearningAgent.get_action_from_loaded_model(model, obs)
    #     knownEnv.render()
    #     next_obs, reward, done, _, _ = knownEnv.step(action)
    #     next_obs = np.append(next_obs['grid'].flatten(), next_obs['position'])
    #     obs = next_obs








    # knownEnv = KnownEnv()
    # number_of_states = 10*10 + 1
    # number_of_action = 4

    # model = A2C(
    #     "MlpPolicy", 
    #     knownEnv, 
    #     n_steps=100, 
    #     verbose=2,
    #     # gamma=0.999, # Determines the weight of future rewards relative to immediate rewards
    #     # ent_coef=0.9,  # Higher values encourage exploration
    #     # vf_coef=0.5, 
    #     # max_grad_norm=0.5, 
    #     # tensorboard_log="logs", 
    #     # policy_kwargs=policy_kwargs
    # )

    
    # model.learn(total_timesteps=1_000_000, tb_log_name="ppo_logs", log_interval=50)



    # knownEnv = KnownEnv(display=True)
    observation, _ = knownEnv.reset()

    # n = knownEnv.observation_space["grid"].shape[0] * knownEnv.observation_space["grid"].shape[1]
    # number_of_possible_observation = 3**n + n
    # print(f"{number_of_possible_observation=}")

    while True:
        action = knownEnv.action_space.sample()
        knownEnv.render()
        # time.sleep(0.1)
        observation, reward, done, _, _ = knownEnv.step(action)
        if done:
            break


    # knownEnv = KnownEnv()
    # model = A2C("MultiInputPolicy", knownEnv, verbose=1, tensorboard_log="./a2c_tensorboard_logs/")
    # model.learn(total_timesteps=500_000)


    # for i in range(20):
    #     vec_env = KnownEnv(display=True)
    #     obs, _ = vec_env.reset()
    #     while True:
    #         vec_env.render()
    #         time.sleep(0.1)
    #         action, _state = model.predict(obs, deterministic=False)
    #         action = action.item()
    #         obs, reward, done, _, _ = vec_env.step(action)
    #         if done:
    #             break


    # knownEnv = KnownEnv()

    # n = knownEnv.observation_space["grid"].shape[0] * knownEnv.observation_space["grid"].shape[1]
    # number_of_possible_observation = 3**n + n
    # print(f"{number_of_possible_observation=}")

    # agent = TabularQLearningAgent(number_of_action=4, 
    #                               number_of_states=number_of_possible_observation, 
    #                               γ=1, 
    #                               α=0.3, 
    #                               ε=0.7,
    #                               α_decay=0.999, 
    #                               ε_decay=0.999, 
    #                               α_min=0, 
    #                               ε_min=0)

    # env, agent = learn_agent(knownEnv, agent, episodes = 10)



