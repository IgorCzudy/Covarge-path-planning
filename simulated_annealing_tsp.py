import math
import random 
import networkx as nx 


def swap_two_nodes(cycle, seed=42):
    random.seed(seed)
    # nie jest w pelni polaczony wiec nie mozna po prostu zaminic wierzcholkow trzeba stawidzic czy sa polaczone 
    a, b = random.sample(range(0, len(cycle)-1), 2)
    cycle[a], cycle[b] = cycle[b], cycle[a]
    return  cycle

def move_one_node(cycle, seed=42):
    random.seed(seed)
    a, b = random.sample(range(0, len(cycle)-1), 2)
    # usuwa element na pozycji a i wtawia go na pozycje b 
    cycle.insert(b, cycle.pop(a))
    return cycle

def swap_two_nodes_for_not_fully_connected_graph(graph, cycle, seed=42):
    random.seed(seed)
    # nie jest w pelni polaczony wiec nie mozna po prostu zaminic wierzcholkow trzeba stawidzic czy sa polaczone 
    indx_to_change = random.sample(range(1, len(cycle)-1), 1)[0]
    number_of_nodes_to_change = cycle[indx_to_change]
    #TODO ograniczyc tez d otych co juz byly odwiedozne w tej czesci 
    possible_nodes = set(graph[number_of_nodes_to_change]) - {cycle[indx_to_change + 1]} - {cycle[indx_to_change - 1]}
    
    node_to_go = random.sample(possible_nodes, 1)[0]
    new_cycle = cycle[:indx_to_change] + [node_to_go]

    #dobudowanie reszty za ppomoca greedy 
    tsp = nx.approximation.traveling_salesman_problem
    node_do_dobudowania = list(set(cycle) - set(new_cycle))
    dobudowana_czesc = tsp(graph, cycle=False, nodes=node_do_dobudowania, )

    whole_ne_cycle = new_cycle + dobudowana_czesc
    return whole_ne_cycle



def my_simulated_annealing_tsp(
    graph, 
    init_cycle,
    source=None,
    temp=100,
    move="1-0",
    max_iterations=10,
    N_inner=100,
    alpha=0.01,
    seed=None,):


    if move == "1-1":
        move = swap_two_nodes
    elif move == "1-0":
        move = move_one_node
    elif move == "not-fully-connected":
        move = swap_two_nodes_for_not_fully_connected_graph
    
    cost = 0 
    for i_cycle, j_cycle in zip(init_cycle, init_cycle[1:]):
        cost += graph.get_edge_data(i_cycle, j_cycle).get('weight', 1)

    count = 0
    best_cost = cost
    cycle = list(init_cycle)
    best_path = init_cycle.copy()
    while count <= max_iterations and temp > 0:
        count+=1 
        for _ in range(N_inner):
            if move == swap_two_nodes_for_not_fully_connected_graph:
                adj_sol = move(graph, cycle)
            else:
                adj_sol = move(cycle)
            
            # adj_cost = sum([graph.get_edge_data(adj_i, adj_j).get('weight', 1) for adj_i, adj_j in zip(adj_sol, adj_sol[1:])])
            adj_cost = 0
            for adj_i, adj_j in zip(adj_sol, adj_sol[1:]):
                adj_cost+=graph.get_edge_data(adj_i, adj_j).get('weight', 1) 

            delta = adj_cost - cost
            if delta <= 0:#new is better
                # set as adjent solustion 
                cycle = adj_sol
                cost = adj_cost
                if cost < best_cost:
                    count = 0
                    best_path = cycle.copy()
                    best_cost = cost
            else:
                # accept even a worse solution with some probab 
                p = math.exp(-delta / temp)
                random.seed(42)
                if p >= random.random():
                    cycle = adj_sol
                    cost = adj_cost
        temp -= temp * alpha

    return best_path
