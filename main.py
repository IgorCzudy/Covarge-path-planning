import networkx as nx

from zig_zag import Zig_zag
from ploting import plot_graph, plot_matrix
from create_data_structure import make_grid, create_adjacency_matrix, create_networkX_graph, create_fully_connected_graph
from simulated_annealing_tsp import my_simulated_annealing_tsp
from random_walk import Random_walk





if __name__ == "__main__":
    # grid = make_grid(10, True, coverage = 0)
    grid = make_grid()
    plot_matrix(grid)
    graph, nodes_without_obst = create_networkX_graph(grid)
    complete_graph = create_fully_connected_graph(graph, nodes_without_obst) 

    
    zig_zag_pathfinder = Zig_zag()
    path = zig_zag_pathfinder(grid, complete_graph, nodes_without_obst)
    
    # random_walk =  Random_walk()
    # path = random_walk(graph, nodes_without_obst)
    
    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.christofides)
    # path = path[:path.index(0)-1]

    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # path = path[:path.index(0)-1]

    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # cycle = my_simulated_annealing_tsp(complete_graph, path, temp=100, alpha=0.01, max_iterations=5, N_inner=50, move = "not-fully-connected")

    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # cycle = my_simulated_annealing_tsp(complete_graph, path, temp=100, alpha=0.01, max_iterations=5, N_inner=50, move = "1-1")

    print(path)
    cost = sum(complete_graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    print("Total cost:", cost)
    plot_graph(graph, path)

    