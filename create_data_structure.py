import numpy as np 
import networkx as nx
from typing import List, Tuple

def make_grid(size: int = 10, random: bool = False, coverage: float = 0.1) -> np.ndarray:
    grid = np.ones((size, size), dtype=int)

    # if not random and size > 5:
        # grid[0,8] = 1
    grid[4,5] = 2
    grid[5,5] = 2

    grid[8,8] = 2
    grid[9,9] = 2

    grid[2,8] = 2
    grid[2,7] = 2
    grid[1,8] = 2
    grid[1,7] = 2

    grid[5,5] = 2
    grid[6,6] = 2
    grid[4,4] = 2

    # else:
    #     total_cells = size * size
    #     num_obstacles = int(total_cells * coverage)

    #     obstacles = set()
    #     while len(obstacles) < num_obstacles:
    #         row = np.random.randint(0, size)
    #         col = np.random.randint(0, size)
    #         obstacles.add((row, col))

    #     for row, col in obstacles:
    #         grid[row, col] = 1

    return grid



def create_adjacency_matrix(grid: np.ndarray) -> Tuple[nx.Graph, List[int]]:

    height, width = grid.shape
    number_of_cells = height * width
    adjacency_matrix = np.zeros((number_of_cells, number_of_cells))

    obstacles = [i*height+j for i in range(height) for j in range(width) if grid[i][j] == 1]
    not_obstacles = [i for i in range(number_of_cells) if i not in obstacles] 
    possible_moves = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1), (1, 0), (1, 1)]


    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                continue

            for move_x, move_y in possible_moves:
                current_move_x = i + move_x
                current_move_y = j + move_y

                #TODO posiible bag here, swap width with height
                if 0 <= current_move_x < width and 0 <= current_move_y < height and grid[current_move_x][current_move_y] == 0:
                    current_index = current_move_x * height + current_move_y

                    adjacency_matrix[i*height+j, current_index] = 1

    return adjacency_matrix, not_obstacles


def create_networkX_graph(grid: np.ndarray) -> Tuple[nx.Graph, List[int]]:
    height, width = grid.shape
    number_of_cells = height * width

    obstacles = [i*height+j for i in range(height) for j in range(width) if grid[i][j] == 1]
    not_obstacles = [i for i in range(number_of_cells) if i not in obstacles] 
    possible_moves = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1), (1, 0), (1, 1)]


    G = nx.Graph()
    G.add_nodes_from([i for i in range(number_of_cells)])

    edges_to_add = []
    for node_number in G.nodes:
        if node_number in obstacles:
            continue

        moves = []
        node_x, node_y = divmod(node_number, width)

        for move_x, move_y in possible_moves:
            current_move_x = node_x + move_x
            current_move_y = node_y + move_y

            #TODO posiible bag here, swap width with height
            if 0 <= current_move_x < width and 0 <= current_move_y < height and grid[current_move_x][current_move_y] == 0:
                current_index = current_move_x * height + current_move_y
                moves.append(current_index)

        list_of_edges = list(zip([node_number] * len(moves), moves))
        edges_to_add.extend(list_of_edges)
    
    G.add_edges_from(edges_to_add)
    return G, not_obstacles


def create_fully_connected_graph(graph, nodes_without_obst):

    complete_graph = nx.Graph(graph)
    
    complete_graph = nx.Graph(graph)
    for u in nodes_without_obst:
        for v in nodes_without_obst:
            if u != v and not graph.has_edge(u, v):
                path = nx.shortest_path(graph, source=u, target=v, weight='weight')
                length = nx.shortest_path_length(graph, source=u, target=v, weight='weight')
                complete_graph.add_edge(u, v, weight=length, path=path)
            if u == v:
                complete_graph.add_edge(u, v, weight=1000, path=[])

    return complete_graph