
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx
from typing import List


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
    pairs = list(zip(path, path[1:]))

    # pairs = [(1,2), (2,3), (3,4), (4,3), (3,13), (13,14), (14, 4), (4,3)]
    # pairs = [(16, 25), (25, 16)]
    not_unique_pairs = []
    unique_pairs = []
    unique_third = []
    seen = set()
    for pair in pairs:
        ordered_pair = tuple(sorted(pair))
        if ordered_pair not in seen:
            unique_pairs.append(pair)
            seen.add(ordered_pair)
        else:
            not_unique_pairs.append(pair)

            # not_unique_pairs.append((pair[1] ,pair[0]))
            if (pair[1], pair[0]) in unique_pairs:
                unique_pairs.remove((pair[1], pair[0]))
            else:
                unique_third.append((pair[0], pair[1]))

    for fro, to in unique_pairs:
        arrow = FancyArrowPatch(pos[fro], pos[to], arrowstyle='-|>', color='red', mutation_scale=30, lw=2)
        plt.gca().add_patch(arrow)
    
    for fro, to in unique_third:
        arrow = FancyArrowPatch(pos[fro], pos[to], arrowstyle='-|>', color='y', mutation_scale=30, lw=2)
        plt.gca().add_patch(arrow)

    for fro, to in not_unique_pairs:
        offset = 0.1  # Adjust this value as needed
        
        start1 = (pos[to][0] - offset, pos[to][1] - offset)
        end1 = (pos[fro][0] - offset, pos[fro][1] - offset)
        arrow1 = FancyArrowPatch(start1, end1, arrowstyle='-|>', color='red', mutation_scale=20, lw=2)
        plt.gca().add_patch(arrow1)

        start2 = (pos[fro][0] + offset, pos[fro][1] + offset)
        end2 = (pos[to][0] + offset, pos[to][1] + offset)
        arrow2 = FancyArrowPatch(start2, end2, connectionstyle="angle3" ,arrowstyle='-|>', color='orange', mutation_scale=20, lw=2)
        plt.gca().add_patch(arrow2)


    plt.title(f'{path=}')
    plt.axis('off')  # Hide the axis
    plt.show()
