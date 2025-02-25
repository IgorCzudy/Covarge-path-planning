import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from Rl_run import learn_agent, make_Q_table_plot, get_sample_actions_Q_table
from Rl_agents import TabularQLearningAgent
from ploting import plot_graph


class SimpleCovargePathPlanningEnv(gym.Env):

    def __init__(self, grid_size=(5, 5), display=False, starting_point=(0, 0)):
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
                1: (0, 0, 0),  # Green for obstacles or paths
                2: (0, 255, 0),  # Green for visited cell
                9: (255, 0, 0),  # Red for the agent
            }
            self.font = pygame.font.Font(None, 30)  # Define the font for numbers

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

    def change_starting_point(self):
        import random

        random.seed(42)
        self.starting_point = tuple(random.choice(np.argwhere(self.grid != 1)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.make_grid()  # always the same grid

        started_poin_x, started_poin_y = self.starting_point[0], self.starting_point[1]
        self.agent_pos = [started_poin_x, started_poin_y]
        self.grid[started_poin_x, started_poin_y] = 2

        return self._get_observation(), {}

    def step(self, action):
        assert 0 <= action <= 3, "action must be in the range from 0 to 3"

        self.num_of_steps += 1
        x, y = self.agent_pos

        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        # move out of boundry, dont applay it
        if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height):
            return self._get_observation(), -0.10, False, False, {}

        move_cell_value = self.grid[new_x, new_y]
        # 1: move to obstyckle, 2: move to alredy visited cell, 0: new visited move
        reward_map = {1: -0.10, 2: -0.05, 0: 0.1}
        # reward = reward_map.get(cell_value, 0)
        reward = reward_map[move_cell_value]

        if move_cell_value != 1:  # valid move, applay it
            self.agent_pos = (new_x, new_y)
            self.grid[new_x, new_y] = 2  # marked as visited

        done = np.all((self.grid == 2) | (self.grid == 1))
        # if all cells that are not obstyckle ware visited
        if done:
            reward = 1

        return self._get_observation(), reward, done, False, {}

    def render(self, mode="human"):
        render_grid = self.grid.copy()
        x, y = self.agent_pos
        render_grid[x, y] = 9

        self.draw_grid(render_grid)

    def draw_grid(self, render_grid):
        
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

    def _get_observation(self):
        return self.agent_pos[1] + self.agent_pos[0] * self.grid_height


if __name__ == "__main__":

    env = SimpleCovargePathPlanningEnv(
        grid_size=(5, 5), starting_point=(0, 0), display=True
    )

    agent = TabularQLearningAgent(
        number_of_action=4,
        number_of_states=25,
        γ=1,
        α=0.1,
        ε=0.9,
        α_decay=0.999,
        ε_decay=0.99,
        α_min=0.1,
        ε_min=0.001,
    )

    env, agent = learn_agent(
        env,
        agent,
        episodes=15,
        plot=True,
        display_qtable=True,
        display_pygame=True,
        change_starting_point=False,
        debug_mode=True,
    )

    make_Q_table_plot(agent)

    for starting_point in [(0, 0), (4, 4), (1, 1), (4, 0), (0, 3)]:

        env = SimpleCovargePathPlanningEnv(
            grid_size=(5, 5), starting_point=starting_point, display=True
        )
        actions, path = get_sample_actions_Q_table(
            env, agent, starting_point=starting_point
        )

        plot_graph(path)
