import numpy as np
from typing import List, Tuple
import networkx as nx


class Zig_zag:
    def __init__(self):
        self.grid = None
        self.x_shape = None
        self.y_shape = None
        self.complete_graph = None
        self.nodes_without_obst = None

        self.path = None
        self.x = None
        self.y = None

    def _can_i_go_down(self):
        return (
            self.y + 1 < self.y_shape
            and self.grid[self.y + 1, self.x] == 0
            and (self.y + 1) * self.x_shape + self.x not in self.path
        )

    def _going_down(self):
        indx = (self.y + 1) * self.x_shape + self.x
        self.y += 1
        self.path.append(indx)

    def _can_i_go_up(self):
        return (
            self.y - 1 >= 0
            and self.grid[self.y - 1, self.x] == 0
            and (self.y - 1) * self.x_shape + self.x not in self.path
        )

    def _going_up(self):
        indx = (self.y - 1) * self.x_shape + self.x
        self.path.append(indx)
        self.y -= 1

    def _can_i__go_right(self):
        return (
            self.x + 1 < self.x_shape
            and self.grid[self.y, self.x + 1] == 0
            and self.y * self.x_shape + (self.x + 1) not in self.path
        )

    def _go_right(self):
        indx = self.y * self.x_shape + (self.x + 1)
        self.path.append(indx)
        self.x += 1

    def _can_i__go_left(self):
        return (
            self.x > 0
            and self.grid[self.y, self.x - 1] == 0
            and self.y * self.x_shape + (self.x - 1) not in self.path
        )

    def _go_left(self):
        indx = self.y * self.x_shape + (self.x - 1)
        self.x -= 1
        self.path.append(indx)

    def _find_the_closest_one(self) -> Tuple[int, int]:
        indx = self.y * self.x_shape + self.x
        filtered_connected_nodes_with_weights = [
            (neighbor, self.complete_graph[indx][neighbor].get("weight", 1))
            for neighbor in self.complete_graph.neighbors(indx)
            if neighbor not in self.path
        ]
        next_indx, _ = min(filtered_connected_nodes_with_weights, key=lambda x: x[1])

        self.y = next_indx // self.x_shape
        self.x = next_indx % self.x_shape

        return next_indx, indx

    def _are_not_all_nodes_visited(self):
        return self.nodes_without_obst - set(self.path)

    def _zig_zag_path(self) -> List[int]:

        while self._are_not_all_nodes_visited():

            if self._can_i_go_up():
                self._going_up()

            elif self.y % 2 == 0:  # right

                if self._can_i__go_right():
                    self._go_right()

                else:
                    if self._can_i__go_left():
                        self._go_left()

                    elif self._can_i_go_down():
                        self._going_down()

                    else:  # there is obstyckle down or we aredy have been there, go to the closest one
                        next_indx, indx = self._find_the_closest_one()

                        if "path" not in self.complete_graph[indx][next_indx]:
                            part_path = [next_indx]
                        else:
                            part_path = self.complete_graph[indx][next_indx]["path"][
                                ::-1
                            ][1:]

                        self.path.extend(part_path)

            else:  # left
                if self._can_i__go_left():
                    self._go_left()

                else:
                    if self._can_i__go_right():
                        self._go_right

                    if self._can_i_go_down():
                        self._going_down()

                    else:  # there is obstyckle down or we aredy have been there, go to the closest one

                        next_indx, indx = self._find_the_closest_one()

                        if "path" not in self.complete_graph[indx][next_indx]:
                            part_path = [next_indx]
                        else:
                            part_path = self.complete_graph[indx][next_indx]["path"][
                                ::-1
                            ][1:]

                        self.path.extend(part_path)

        return self.path

    def _initialize(
        self, grid: np.ndarray, complete_graph: nx.Graph, nodes_without_obst: List[int]
    ):
        self.grid = grid
        self.complete_graph = complete_graph
        self.nodes_without_obst = set(nodes_without_obst)
        self.x_shape, self.y_shape = grid.shape

        self.path = [0]  # Start from the initial node (0,0)
        self.x, self.y = 0, 0  # Starting coordinates

    def __call__(
        self, grid: np.ndarray, complete_graph: nx.Graph, nodes_without_obst: List[int]
    ) -> List[int]:

        self._initialize(grid, complete_graph, nodes_without_obst)
        path = self._zig_zag_path()
        return path
