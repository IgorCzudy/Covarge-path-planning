import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple


def plot_matrix(grid: np.ndarray):
    plt.figure(figsize=(6, 6))
    plt.imshow(
        grid, cmap="gray", origin="upper", extent=[0, 10, 0, 10]
    ) 
    plt.clim(-0.5, 1.5)

    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    plt.grid(which="both", color="black", linestyle="-", linewidth=1)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Grid with (0,0) in Top-Left Corner")

    plt.show()


def create_graph(number_of_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(number_of_nodes)])
    return graph


def plot_graph_two_agents(path_agent_one: List[int], path_agent_two: List[int]):
    graph = create_graph(number_of_nodes=25)
    plt.figure(figsize=(8, 8))

    plot_one_path(path_agent_one, graph, colors=None)
    plot_one_path(path_agent_two, graph, colors=1)

    plt.title(f"{path_agent_one=} || {path_agent_two}")
    plt.axis("off")
    plt.show()


def plot_graph(path: List[int]):
    graph = create_graph(number_of_nodes=25)

    plt.figure(figsize=(8, 8))
    plot_one_path(path, graph)

    plt.title(f"{path=}")
    plt.axis("off")
    plt.show()


def plot_nodes(graph: nx.Graph, pos: Dict[int, int], path_nodes: set[int]):
    nx.draw(
        graph, pos, with_labels=True, node_size=200, node_color="orange", font_size=10
    )
    nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_color="lightblue")


def generate_pairs(
    pairs: List[Tuple[int, int]]
) -> Tuple[List[int], List[int], List[int]]:
    not_unique_pairs, unique_pairs, unique_third = [], [], []
    seen = set()
    for pair in pairs:
        ordered_pair = tuple(sorted(pair))
        if ordered_pair not in seen:
            unique_pairs.append(pair)
            seen.add(ordered_pair)
        else:
            not_unique_pairs.append(pair)

            if (pair[1], pair[0]) in unique_pairs:
                unique_pairs.remove((pair[1], pair[0]))
            else:
                unique_third.append((pair[0], pair[1]))

    return not_unique_pairs, unique_pairs, unique_third


def drow_arrow(
    pos: Dict[int, int],
    fro: Tuple[int, int],
    to: Tuple[int, int],
    color_unique_pairs: str,
    mutation_scale: int,
    lw: int,
    offset: float = 0.0,
):
    arrow = FancyArrowPatch(
        pos[fro],
        pos[to],
        arrowstyle="-|>",
        color=color_unique_pairs,
        mutation_scale=mutation_scale,
        lw=lw,
    )
    plt.gca().add_patch(arrow)


def plot_one_path(path: List[int], graph: nx.Graph, colors=None):

    if colors is None:
        color_unique_pairs = "red"
        color_unique_third = "green"
    else:
        color_unique_pairs = "yellow"
        color_unique_third = "cyan"

    grid_size = int(len(graph.nodes()) ** 0.5)
    pos = {
        i: (i % grid_size, grid_size - 1 - (i // grid_size))
        for i in range(grid_size * grid_size)
    }

    plot_nodes(graph, pos=pos, path_nodes=set(path))

    pairs = list(zip(path, path[1:]))

    not_unique_pairs, unique_pairs, unique_third = generate_pairs(pairs)

    for fro, to in unique_pairs:
        drow_arrow(pos, fro, to, color_unique_pairs, mutation_scale=30, lw=2)

    for fro, to in unique_third:
        drow_arrow(pos, fro, to, color_unique_third, mutation_scale=30, lw=2)

    for fro, to in not_unique_pairs:
        offset = 0.1
        drow_arrow(
            pos, to, fro, color_unique_pairs, mutation_scale=20, lw=2, offset=-offset
        )
        drow_arrow(
            pos, fro, to, color_unique_pairs, mutation_scale=20, lw=2, offset=offset
        )
