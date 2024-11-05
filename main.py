import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx
from typing import List, Tuple
from matplotlib.patches import FancyArrowPatch


def make_grid(size: int = 10, random: bool = False, coverage: float = 0.1) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)

    if not random and size > 5:
        grid[4,4] = 1
        grid[4,5] = 1
        grid[5,5] = 1
        grid[0,0] = 1
    else:
        total_cells = size * size
        num_obstacles = int(total_cells * coverage)

        obstacles = set()
        while len(obstacles) < num_obstacles:
            row = np.random.randint(0, size)
            col = np.random.randint(0, size)
            obstacles.add((row, col))

        for row, col in obstacles:
            grid[row, col] = 1

    return grid



def plot_matrix(grid: np.ndarray):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='gray', origin='upper', extent=[0, 10, 0, 10])  # Set origin to 'upper' for top-left (0,0)
    plt.clim(-0.5, 1.5)

    plt.xticks(np.arange(0, 11, 1)) 
    plt.yticks(np.arange(0, 11, 1))
    plt.grid(which='both', color='black', linestyle='-', linewidth=1)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Grid with (0,0) in Top-Left Corner")

    plt.show()



def plot_graph(graph: nx.Graph, path: List[int]):
    grid_size = int(len(graph.nodes()) ** 0.5)
    
    pos = {i: (i % grid_size, grid_size - 1 - (i // grid_size)) for i in range(grid_size * grid_size)}
    plt.figure(figsize=(8, 8))
    
    nx.draw(graph, pos, with_labels=True, node_size=200, node_color='orange', font_size=10)

    path_nodes = set(path)
    nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_color='lightblue')

    
    for fro, to in  list(zip(path, path[1:])):
        arrow = FancyArrowPatch(pos[fro], pos[to], arrowstyle='-|>', color='red', mutation_scale=30, lw=2)
        plt.gca().add_patch(arrow)

    plt.title('NetworkX Graph of the 10x10 Grid with (0, 0) at Top-Left Corner')
    plt.axis('off')  # Hide the axis
    plt.show()


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



if __name__ == "__main__":
    grid = make_grid()

    # plot_matrix(grid)

    # graph, nodes_without_obst = create_networkX_graph(grid)
    adjacency_matrix, nodes_without_obst = create_adjacency_matrix(grid)
    graph = nx.from_numpy_array(adjacency_matrix)

    tsp = nx.approximation.traveling_salesman_problem
    path = tsp(graph, cycle=False, nodes=nodes_without_obst)
    print(path)


    plot_graph(graph, path)
