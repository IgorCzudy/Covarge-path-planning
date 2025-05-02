
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, List, Dict, Optional
import pygame


class ShortestPath(): #gym.Env
    
    def __init__(self, grid_size: Tuple[int, int] = (9, 9), display: bool = False):
        # super(TwoQTable, self).__init__()
    
        self.first_agent_starting_position = (0, 0)
        self.secend_agent_starting_position = (0, 8)

        self.first_agent_position: Tuple[int, int] = (0, 0)
        self.secend_agent_position: Tuple[int, int] = (0, 8)

        self.grid_width: int = grid_size[0]
        self.grid_height: int = grid_size[1]
        self.grid_size = grid_size[0]
        
        self.num_of_steps: int = 0
        self.target_pos = (8, 8)

        self.grid: Optional[np.dnarray] = None

        self.max_steps = 100


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

        grid[0, 5] = 1
        grid[0, 6] = 1
        grid[1, 5] = 1
        grid[1, 6] = 1
        
        grid[3, 3] = 1
        grid[3, 5] = 1
        grid[4, 3] = 1
        grid[4, 5] = 1

        grid[3, 4] = 1
        grid[4, 4] = 1
        grid[6, 5] = 1
        grid[6, 6] = 1

        grid[5, 0] = 1
        grid[6, 0] = 1
        grid[6, 1] = 1

        grid[4, 8] = 1
        return grid

    def _distance(self, pos1, pos2):
        """Oblicz odległość Manhattan"""
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def _get_observation(self, agent_number: int) -> int:

        # agent_position = self.first_agent_position if agent_number==1 else self. 
        
        # assert 0 <= n <= 24

        if agent_number == 0:
            agent = self.first_agent_position
        else:
            agent = self.secend_agent_position

        x_pos, y_pos = agent
        agent_node = agent[0] *self.grid_width + agent[1]

        neighbours = []
        # 0: move up, 1: move down, 2: move left, 3: move right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for move_x, move_y in moves:
            x_new_pos = x_pos + move_x
            y_new_pos = y_pos + move_y

            if not (0 <= x_new_pos < self.grid_width and 0 <= y_new_pos < self.grid_height): #move out of bandry
                neighbours.extend([0, 0])

            elif self.grid[x_new_pos, y_new_pos] == 1:  # Obstacle
                neighbours.extend([0, 0])
                        
            elif self.grid[x_new_pos, y_new_pos] == 2 or self.grid[x_new_pos, y_new_pos] == 3:  # Visited cell
                neighbours.extend([1, 0])
            else:  # New valid move
                neighbours.extend([1, 1]) # Not visited node


        # return (agent[0]/self.grid_width, agent[1]/self.grid_width, *neighbours)

        # obs = self.grid.copy()
        # obs[agent] = 2
        return np.concatenate(([agent[0]/self.grid_width, agent[1]/self.grid_width], self.grid.flatten()/2)).astype(np.float32)
        # return [agent_position[0]/self.grid_width, agent_position[1]/self.grid_width]
    
    def _get_global_state(self):
        # obs = self.grid.copy()
        # obs[self.first_agent_position] = 2
        # obs[self.secend_agent_position] = 2
        return np.concatenate(([self.first_agent_position[0]/self.grid_width, self.first_agent_position[1]/self.grid_width,
                                self.secend_agent_position[0]/self.grid_width, self.secend_agent_position[1]/self.grid_width],
                                self.grid.flatten()/2)).astype(np.float32)
        # return (*self._get_observation(0), *self._get_observation(1))
        # return [
        #     *(x / self.grid_width for x in self.first_agent_position),
        #     *(x / self.grid_width for x in self.secend_agent_position)
        #     # *(x / self.grid_width for x in self.target_pos)
        # ]



    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[int, int, Dict]:
        # super().reset(seed=seed)
        self.num_of_steps = 0
        self.grid = self.make_grid()
        self.first_agent_position = self.first_agent_starting_position
        self.secend_agent_position = self.secend_agent_starting_position
        # self.grid[self.first_agent_position] = 2
        # self.grid[self.secend_agent_position] = 3

        return (
            [self._get_observation(agent_number=1),  # Obserwacje agentów
            self._get_observation(agent_number=2)],
            self._get_global_state()         # Globalny stan
        )


    def step(self, actions: List[int]) -> Tuple[int, int, int, int, bool, bool, Dict]:
        # Ruch agentów
        action1 = actions[0]  # Agent 1
        action2 = actions[1]  # Agent 2

        assert 0 <= action1 <= 15, "action must be in the range from 0 to 15"
        assert 0 <= action2 <= 15, "action must be in the range from 0 to 15"
        self.num_of_steps += 1

        # # 0: move up, 1: move down, 2: move left, 3: move right
        # moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        # reword ={0: 0, 1: 0} #agent_number: reword
        # for i, (agent_position, action) in enumerate([(self.first_agent_position, action1), (self.secend_agent_position, action2)]):
        #     x, y = agent_position
        #     dx, dy = moves[action]
        #     new_x, new_y = x + dx, y + dy
        #     if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height): #move out of bandry
        #         reword[i] -= 0.1

        #     elif self.grid[new_x, new_y] == 1:  # Obstacle
        #         reword[i] -= 0.1
                        
        #     elif self.grid[new_x, new_y] == 2 or self.grid[new_x, new_y] == 3:  # Visited cell
        #         if i==0:
        #             self.first_agent_position = new_x, new_y
        #             self.grid[new_x, new_y] = 2
        #         else:
        #             self.secend_agent_position = new_x, new_y
        #             self.grid[new_x, new_y] = 3
        #         reword[i] -= 0.05 #0.5

        #     else:  # New valid move
        #         if i==0:
        #             self.first_agent_position = new_x, new_y
        #             self.grid[new_x, new_y] = 2
        #         else:
        #             self.secend_agent_position = new_x, new_y
        #             self.grid[new_x, new_y] = 3
        #         reword[i] += 0.1 #0.5
        
        
        self.prev_agent1_pos = self.first_agent_position
        self.prev_agent2_pos = self.secend_agent_position
        
        # Reszta logiki ruchu bez zmian
        self._move_agent(0, actions[0])
        self._move_agent(1, actions[1])
        
        reward, done = self._calculate_reward()        

        if self.num_of_steps >= self.max_steps:
            done = True
            
        return (
            [self._get_observation(0),
            self._get_observation(1)],
            reward,
            done,
            self._get_global_state()
        )

        # return (
        #     [self._get_observation(0),  # Nowe obserwacje
        #     self._get_observation(1)],
        #     reword[0] + reword[1],                          # Wspólna nagroda
        #     done,                            # Czy epizod zakończony
        #     self._get_global_state()         # Nowy stan globalny
        # )


    def _move_agent(self, agent_idx, action):
        """Mechanika ruchu (4 kierunki)"""
        x, y = self.first_agent_position if agent_idx == 0 else self.secend_agent_position
        
        # Mapowanie akcji (0-3 dla 4 kierunków)
        if action == 0:   # GÓRA
            y = min(y + 1, self.grid_size-1)
        elif action == 1: # DÓŁ
            y = max(y - 1, 0)
        elif action == 2: # PRAWO
            x = min(x + 1, self.grid_size-1)
        elif action == 3: # LEWO
            x = max(x - 1, 0)
            
        if self.grid[x, y] != 1:#jeśli nie ma obstyckle
            if agent_idx == 0:
                self.first_agent_position = (x, y)
            else:
                self.secend_agent_position = (x, y)


    def _calculate_reward(self):
        """Ulepszona funkcja nagród"""
        total_reward = 0
        done = False
        
        # Oblicz odległości
        curr_dist1 = self._distance(self.first_agent_position, self.target_pos)
        curr_dist2 = self._distance(self.secend_agent_position, self.target_pos)
        prev_dist1 = self._distance(self.prev_agent1_pos, self.target_pos)
        prev_dist2 = self._distance(self.prev_agent2_pos, self.target_pos)

        # Nagrody za ruch w dobrym kierunku
        if curr_dist1 < prev_dist1:
            total_reward += 0.1  # Agent 1 zbliżył się d8o celu
        elif curr_dist1 > prev_dist1:
            total_reward -= 0.1  # Agent 1 oddalił się
        else: #Agent sie nie rudzył
            total_reward -= 0.5
            
        if curr_dist2 < prev_dist2:
            total_reward += 0.1  # Agent 2 zbliżył się do celu
        elif curr_dist2 > prev_dist2:
            total_reward -= 0.1  # Agent 2 oddalił się
        else: #Agent sie nie ruszył
            total_reward -= 0.5


        # Duża nagroda za osiągnięcie celu
        if curr_dist1 == 0 and curr_dist2 == 0:
            total_reward += 10.0
            done = True
        
        # Add exploration bonus for visiting new cells
        if self.grid[self.first_agent_position] not in [2, 3]:
            total_reward += 0.05
        if self.grid[self.secend_agent_position] not in [2, 3]:
            total_reward += 0.05

            
        # Mała kara za każdy krok
        total_reward -= 0.01
        
        return total_reward, done

        
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
                if (row, col) == self.target_pos:
                    color = (255, 0, 0)  # Red or whatever you want for the target

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

