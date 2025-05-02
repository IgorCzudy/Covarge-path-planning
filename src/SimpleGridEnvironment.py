
import numpy as np
from typing import List


class SimpleGridEnvironment:
    def __init__(self, render=False):
        self.grid_size = 10
        self.target_pos = (9, 9)
        self.max_steps = 50
        self.current_step = 0
        self.render = render
        
        # Nowe zmienne do śledzenia poprzednich pozycji
        self.prev_agent1_pos = (0, 0)
        self.prev_agent2_pos = (0, 0)

    def reset(self):
        self.agent1_pos = (0,9)#(np.random.randint(0,5), np.random.randint(0,5))
        self.agent2_pos = (0,0)#(np.random.randint(0,5), np.random.randint(0,5))
        self.prev_agent1_pos = self.agent1_pos
        self.prev_agent2_pos = self.agent2_pos
        self.current_step = 0
        
        if self.render:
            self._render()
            
        return (
            self._get_obs(self.agent1_pos),
            self._get_obs(self.agent2_pos),
        )

    def _calculate_reward(self):
        """Ulepszona funkcja nagród"""
        total_reward1 = 0
        total_reward2 = 0
        done = False
        
        # Oblicz odległości
        curr_dist1 = self._distance(self.agent1_pos, self.target_pos)
        curr_dist2 = self._distance(self.agent2_pos, self.target_pos)
        prev_dist1 = self._distance(self.prev_agent1_pos, self.target_pos)
        prev_dist2 = self._distance(self.prev_agent2_pos, self.target_pos)

        # Nagrody za ruch w dobrym kierunku
        if curr_dist1 < prev_dist1:
            total_reward1 += 0.1  # Agent 1 zbliżył się do celu
        elif curr_dist1 > prev_dist1:
            total_reward1 -= 0.1  # Agent 1 oddalił się
            
        if curr_dist2 < prev_dist2:
            total_reward2 += 0.1  # Agent 2 zbliżył się do celu
        elif curr_dist2 > prev_dist2:
            total_reward2 -= 0.1  # Agent 2 oddalił się

        # Duża nagroda za osiągnięcie celu
        if curr_dist1 == 0 and curr_dist2 == 0:
            total_reward1 += 10.0
            total_reward2 += 10.0
            done = True
            
        # Kara za bezczynność (ten sam dystans)
        if curr_dist1 == prev_dist1:
            total_reward1 -= 0.05
        if curr_dist2 == prev_dist2:
            total_reward2 -= 0.05
            
        # Mała kara za każdy krok
        total_reward1 -= 0.01
        total_reward2 -= 0.01
        
        return total_reward1, total_reward2, done

    def _distance(self, pos1, pos2):
        """Oblicz odległość Manhattan"""
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def step(self, action1, action2):
        # Zapisz poprzednie pozycje przed ruchem
        self.prev_agent1_pos = self.agent1_pos
        self.prev_agent2_pos = self.agent2_pos
        
        # Reszta logiki ruchu bez zmian
        actions = [action1, action2]
        self._move_agent(0, actions[0])
        self._move_agent(1, actions[1])
        
        reward1, reward2, done = self._calculate_reward()
        self.current_step += 1
        
        if self.render:
            self._render()
            import time
            time.sleep(0.5)

        if self.current_step >= self.max_steps:
            done = True
            
        return (
            self._get_obs(self.agent1_pos),
            self._get_obs(self.agent2_pos),
            reward1,
            reward2,
            done,
        )
    
    def _get_obs(self, pos):
        return [pos[0]/self.grid_size, pos[1]/self.grid_size]
    
    
    def _move_agent(self, agent_idx, action):
        """Mechanika ruchu (4 kierunki)"""
        x, y = self.agent1_pos if agent_idx == 0 else self.agent2_pos
        
        # Mapowanie akcji (0-3 dla 4 kierunków)
        if action == 0:   # GÓRA
            y = min(y + 1, self.grid_size-1)
        elif action == 1: # DÓŁ
            y = max(y - 1, 0)
        elif action == 2: # PRAWO
            x = min(x + 1, self.grid_size-1)
        elif action == 3: # LEWO
            x = max(x - 1, 0)
            
        if agent_idx == 0:
            self.agent1_pos = (x, y)
        else:
            self.agent2_pos = (x, y)
        
    def _render(self):
        """Wizualizacja ASCII gridu"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Zaznacz target
        tx, ty = self.target_pos
        grid[ty][tx] = 'T'
        
        # Zaznacz agentów
        a1x, a1y = self.agent1_pos
        a2x, a2y = self.agent2_pos
        grid[a1y][a1x] = 'A'
        grid[a2y][a2x] = 'B'
        
        # Wyświetl grid
        print("\n" + "-"*(self.grid_size*2))
        for row in reversed(grid):  # (0,0) na dole
            print(" ".join(row))

