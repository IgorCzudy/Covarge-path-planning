import random
from typing import List, Tuple, Optional, Set
import numpy as np
import networkx as nx


class Random_walk:
    def __init__(self) -> None:
        self.path: Optional[List[int]] = None
        self.nodes_without_obst: Optional[Set[int]] = None
        self.graph: Optional[nx.Graph] = None

    def _are_not_all_nodes_visited(self):
        return self.nodes_without_obst - set(self.path)

    def _graph_random_walk(self):

        while self._are_not_all_nodes_visited():

            current_node = self.path[-1]
            s = set(self.graph[current_node]) - set(self.path)

            if s:  # check if there is some not visited neighbour
                n = random.sample(s, 1)[0]
                self.path.append(n)
            else:
                to_drow = set(self.graph[current_node]) - {
                    self.path[-1]
                }  # dontt come back to node you just came from
                n = random.sample(to_drow, 1)[0]
                self.path.append(n)

        return self.path

    def _initialize(self, graph: nx.Graph, nodes_without_obst: List[int]):
        self.graph = graph
        self.nodes_without_obst = set(nodes_without_obst)
        self.path = [0]  # Start from the initial node (0,0)

    def __call__(self, graph: nx.Graph, nodes_without_obst: List[int]) -> List[int]:

        self._initialize(graph, nodes_without_obst)
        path = self._graph_random_walk()
        return path
