import networkx as nx

from zig_zag import zig_zag_path
from ploting import plot_graph, plot_matrix
from create_data_structure import make_grid, create_adjacency_matrix, create_networkX_graph, create_fully_connected_graph
from simulated_annealing_tsp import my_simulated_annealing_tsp
from random_walk import graph_random_walk

if __name__ == "__main__":
    # grid = make_grid(10, True, coverage = 0)
    grid = make_grid()
    # plot_matrix(grid)
    graph, nodes_without_obst = create_networkX_graph(grid)
    complete_graph = create_fully_connected_graph(graph, nodes_without_obst) 

    algorythms = [zig_zag_path,  
                graph_random_walk, 
                nx.approximation.traveling_salesman_problem(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp), 
                my_simulated_annealing_tsp,
                nx.approximation.traveling_salesman_problem(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.christofides)]  

    path = zig_zag_path(grid, complete_graph, nodes_without_obst)
    print(path)
    cost = sum(graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    print("Total cost:", cost)
    plot_graph(graph, path)

    # tsp = nx.approximation.traveling_salesman_problem
    # path = nx.approximation.traveling_salesman_problem(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # path = nx.approximation.traveling_salesman_problem(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.christofides)
    # path = path[:path.index(0)-1]
    # print(path)
    # cost = sum(graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    # print("Total cost:", cost)
    # plot_graph(graph, path)
    # import sys; sys.exit(0)


    # AttributeError: 'NoneType' object has no attribute 'get'
    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # cycle = my_simulated_annealing_tsp(graph, path, temp=500, alpha=0.001, max_iterations=20, N_inner=500, move = "not-fully-connected")
    # print(cycle==path)
    # plot_graph(graph, cycle)
    # import sys; sys.exit(0)

    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # plot_graph(graph, path)
    
    # complete_graph = create_fully_connected_graph(graph, nodes_without_obst)

    # cycle = my_simulated_annealing_tsp(complete_graph, path, move = "not-fully-connected", temp=500, alpha=0.001, max_iterations=40, N_inner=1000,)
    # print(cycle==path)
    # print(cycle)
    # plot_graph(complete_graph, cycle)


    # cost = sum(graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    # print("Total cost:", cost)
