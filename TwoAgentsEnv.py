import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from torch.utils.tensorboard import SummaryWriter
import torch
import click
from Rl_run import learn_agent
from Rl_agents import TabularQLearningAgent


class TwoAgentsEnv(gym.Env):

    def __init__(self, grid_size=(5, 5), display=False, tensorb=True):
        super(TwoAgentsEnv, self).__init__()

        self.tensorb = tensorb

        self.first_agent_position = (0, 0)
        self.secend_agent_position = (4, 4)

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
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )

            self.colors = {
                0: (255, 255, 255),  # White for empty cells
                1: (0, 0, 0),  # Green for obstacles or paths
                2: (0, 255, 0),  # Green for visited cell
                9: (255, 0, 0),  # Red for the agent
                8: (190, 0, 0),  # lighter red for the secend agent
            }
            self.font = pygame.font.Font(None, 30)  # Define the font for numbers

        if display and tensorb:
            self.frames = []

    def make_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_width, self.grid_height), dtype=int)

        # obstycle
        grid[0, 2] = 1
        grid[2, 3] = 1
        grid[2, 2] = 1
        grid[3, 3] = 1
        grid[3, 2] = 1

        if self.grid_height > 6 and self.grid_with > 6:
            grid[6, 5] = 1
            grid[5, 6] = 1
            grid[5, 5] = 1
            grid[6, 6] = 1

            grid[4, 2] = 1
            grid[5, 2] = 1
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.make_grid()  # always the same grid

        self.grid[self.first_agent_position] = 2
        self.grid[self.secend_agent_position] = 2

        return self._get_observation(), {}

    def step(self, action):
        assert 0 <= action <= 15, "action must be in the range from 0 to 15"

        self.num_of_steps += 1

        # 16%4 -> first agent action
        # 16//4 -> secend agent action
        # action:0 up
        # action:1 down
        # action:2 left
        # action:3 right

        first_agent_action = action % 4
        secend_agent_action = action // 4

        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        
        reward = 0 
        for i, (agent_position, action) in enumerate([(self.first_agent_position, first_agent_action), (self.secend_agent_position, secend_agent_action)]):
            x, y = agent_position
            dx, dy = moves[action]
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height): #move out of bandry
                reward -= 0.1

            elif self.grid[new_x, new_y] == 1:  # Obstacle
                reward -= 0.1
            
            elif self.grid[new_x, new_y] == 2:  # Visited cell
                if i==0:
                    self.first_agent_position = new_x, new_y
                else:
                    self.secend_agent_position = new_x, new_y
                reward -= 0.5

            else:  # New valid move
                if i==0:
                    self.first_agent_position = new_x, new_y
                else:
                    self.secend_agent_position = new_x, new_y
                self.grid[new_x, new_y] = 2  # Mark as visited
                reward += 0.5

        done = False
        if np.all((self.grid == 2) | (self.grid == 1)):
            reward = 1
            done = True

        return self._get_observation(), reward, done, False, {}

    def render(self, mode="human"):
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

        if self.tensorb:
            frame = pygame.surfarray.array3d(self.window)
            frame = np.flip(frame, axis=1)
            frame = np.rot90(frame)
            self.frames.append(frame)

        pygame.display.flip()

    def _get_observation(self):

        n_first = (
            self.first_agent_position[1]
            + self.first_agent_position[0] * self.grid_height
        )
        n_secend = (
            self.secend_agent_position[1]
            + self.secend_agent_position[0] * self.grid_height
        )

        assert 0 <= n_first <= 24
        assert 0 <= n_secend <= 24
        out = n_first + n_secend * (self.grid_height * self.grid_width)
        assert 0 <= out <= 624
        return out

    def get_video_tensor(self):
        video = np.array(self.frames)
        video = np.expand_dims(video, axis=0)
        video = video.astype(np.uint8)
        video = torch.from_numpy(video)
        video_transposed = video.permute(0, 1, 4, 2, 3)
        return video_transposed


@click.command()
@click.option(
    "--gamma", type=float, default=1.0, help="Discount factor for future rewards"
)
@click.option("--alpha", type=float, default=0.1, help="Learning rate")
@click.option("--epsilon", type=float, default=0.99, help="Exploration rate")
@click.option("--alpha_decay", type=float, default=0.999, help="Decay rate for alpha")
@click.option(
    "--epsilon_decay", type=float, default=0.99999, help="Decay rate for epsilon"
)
@click.option("--alpha_min", type=float, default=0.1, help="Minimum value for alpha")
@click.option(
    "--epsilon_min", type=float, default=0.0005, help="Minimum value for epsilon"
)
@click.option(
    "--episodes", type=int, default=500, help="Number of episodes for training"
)
def main(
    gamma, alpha, epsilon, alpha_decay, epsilon_decay, alpha_min, epsilon_min, episodes
):
    writer = SummaryWriter(
        f"runs/two_agent_episodes={episodes}_epsilon={epsilon}_epsilon_decay={epsilon_decay}_epsilon_min={epsilon_min}"
    )

    number_of_action = 16
    number_of_states = 25 * 25
    writer.add_hparams(
        {
            "number_of_action": number_of_action,
            "number_of_states": number_of_states,
            "γ": gamma,  # Use "gamma" instead of "γ"
            "α": alpha,  # Use "alpha" instead of "α"
            "ε": epsilon,  # Use "epsilon" instead of "ε"
            "α_decay": alpha_decay,
            "ε_decay": epsilon_decay,
            "α_min": alpha_min,
            "ε_min": epsilon_min,
            "episodes": episodes,
        },
        {},
    )

    env = TwoAgentsEnv(display=False, tensorb=False)
    agent = TabularQLearningAgent(
        number_of_action=number_of_action,
        number_of_states=number_of_states,
        γ=gamma,
        α=alpha,
        ε=epsilon,
        α_decay=alpha_decay,
        ε_decay=epsilon_decay,
        α_min=alpha_min,
        ε_min=epsilon_min,
    )

    env, agent = learn_agent(
        env,
        agent,
        writer,
        episodes=episodes,
        plot=False,
        display_qtable=False,
        display_pygame=False,
        change_starting_point=False,
        debug_mode=False,
    )

    env = TwoAgentsEnv(display=True, tensorb=True)
    obs, _ = env.reset()
    done = False
    np.random.seed(42)
    step = 0

    agent.ε = 0
    while not done:
        step += 1
        action = agent.get_action(obs)[0]
        action = action.item()
        next_obs, reward, done, _, _ = env.step(action)
        agent.process_transition(obs, action, reward, next_obs, done)
        obs = next_obs
        env.render()
        video_tensor = env.get_video_tensor()
        last_frame = video_tensor[-1, -1]
        writer.add_image("TwoAgents Simulation", last_frame, global_step=step)

        if step > 80:
            break

    video_tensor = env.get_video_tensor()
    writer.add_video("Full TwoAgents Simulation", video_tensor, fps=1)
    writer.close()


if __name__ == "__main__":
    main()
