import pytest
import sys
import os
import numpy as np 
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from main import make_grid, create_adjacency_matrix, create_networkX_graph


@pytest.mark.parametrize("size, random, coverage, expected_shape", [
    (10, False, 0.0, (10, 10)),
    (5, True, 0.3, (5, 5)),
])
def test_make_grid(size, random, coverage, expected_shape):
    grid = make_grid(size=size, random=random, coverage=coverage)
    assert grid.shape == expected_shape
    assert np.all((grid == 0) | (grid == 1))


@pytest.mark.parametrize(
    "grid_size, random, coverage, expected_nodes_count",
    [
        (10, True, 0.0, 4), 
        (10, True, 0.0, 9), 
        (10, False, 0.5, 9),
        (10, False, 1.0, 0),  
    ]
)
def test_create_adjacency_matrix(grid_size, random, coverage, expected_nodes_count):
    grid = make_grid(size=grid_size, random=random, coverage=coverage)
    adjacency_matrix, nodes_without_obstacles = create_adjacency_matrix(grid)

    assert adjacency_matrix.shape == (grid_size*grid_size, grid_size*grid_size)
    
    for idx in nodes_without_obstacles:
        assert np.sum(adjacency_matrix[idx]) >= 0


def test_create_adjacency_matrix_fully_blocked():
    fully_blocked_grid = np.ones((3,3))

    adjacency_matrix, nodes_without_obstacles = create_adjacency_matrix(fully_blocked_grid)
    assert len(nodes_without_obstacles) == 0
    assert np.all(adjacency_matrix == 0)


def test_create_networkX_graph_fully_blocked():
    fully_blocked_grid = np.ones((3,3))

    graph, _ = create_networkX_graph(fully_blocked_grid)

    assert graph.number_of_nodes() == 3*3
    assert graph.number_of_edges() == 0 


@pytest.mark.parametrize(
    "grid_size, random, coverage",
    [
        (3, False, 0.0),
        (3, True, 0.5),
        (3, True, 0.0),   
        (3, False, 1.0),
    ]
)
def test_create_networkX_graph(grid_size, random, coverage):
    grid = make_grid(size=grid_size, random=random, coverage=coverage)
    graph, _ = create_networkX_graph(grid)

    assert graph.number_of_nodes() == grid_size*grid_size

    if coverage == 0.0:
        assert nx.is_connected(graph)

    for u, v in graph.edges():
        assert grid[u // grid_size, u % grid_size] == 0 
        assert grid[v // grid_size, v % grid_size] == 0


@pytest.mark.parametrize(
    "grid_size, random, coverage",
    [
        (10, False, 0.0),
        (10, True, 0.5),
        (10, True, 0.0),
        (10, True, 1.0)   
    ]
)
def test_if_create_networkX_graph_is_equivalent_to_create_adjacency_matrix(grid_size, random, coverage):
    grid = make_grid(size=grid_size, random=random, coverage=coverage)
    
    nx_graph, nx_nodes_without_obst = create_networkX_graph(grid)

    adjacency_matrix, am_nodes_without_obst = create_adjacency_matrix(grid)
    am_graph = nx.from_numpy_array(adjacency_matrix)

    assert am_nodes_without_obst == nx_nodes_without_obst

    assert nx_graph.number_of_nodes() == am_graph.number_of_nodes()
    assert nx_graph.number_of_edges() == am_graph.number_of_edges()

    nx_edges = set(nx_graph.edges())
    am_edges = set(am_graph.edges())
    assert nx_edges == am_edges
 



