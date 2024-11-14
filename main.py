import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx
from typing import List, Tuple
from matplotlib.patches import FancyArrowPatch
from collections import Counter
from itertools import permutations, product
import multiprocessing
import random 
import math 


def make_grid(size: int = 10, random: bool = False, coverage: float = 0.1) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)

    if not random and size > 5:
        # grid[0,8] = 1
        grid[4,5] = 1
        grid[5,5] = 1

        grid[8,8] = 1
        grid[9,9] = 1

        grid[2,8] = 1
        grid[2,7] = 1
        grid[1,8] = 1
        grid[1,7] = 1
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


def graph_random_walk(graph, nodes_without_obst):

    nodes = list(nodes_without_obst)
    path = [nodes[0]] # starting point

    dead_point = [] # TODO this deadpoint needs more thinking, possible optymalization here  

    while set(nodes_without_obst) - set(path):# do poki wszystkie wiercholki nie zostana odwiedzone 
        current_node = path[-1]
        s = set(graph[current_node]) - set(path)
        if s: # check if it is not empty 
            n = random.sample(s, 1)[0]
            path.append(n)
            dead_point = []
        else: # jesli nie ma nieodwiedzonych wiercholkow z sotepnych z danego wierzcholka cofnik sie do jakiegokolwiek zajetego 
            dead_point.append(current_node)
            n = random.sample(set(graph[current_node])-set(dead_point), 1)[0]
            path.append(n)
        
    return path


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


def going_down(x, y, grid, x_shape, y_shape, path):
    
    indx = (y+1)*x_shape + x
    path.append(indx) # add down 
    y+= 1
    return x, y, path

    
def can_i_go_down(x, y, y_shape, x_shape, grid, path):
    if y+1 < y_shape and grid[y+1, x] == 0 and \
        (y+1)*x_shape + x not in path:    #chck if down is ok 
        return True
    
    return False

def can_i_go_up(x, y, x_shape, grid, path):
    if y-1 >=0 and grid[y-1, x] == 0 and (y-1)*x_shape + x not in path:
        return True
    return False


def going_up(x, y, grid, x_shape, y_shape, path):
    indx = (y-1)*x_shape + x
    path.append(indx) # add down 
    y-=1
    return x, y, path

def find_the_closest_one(x, y, x_shape, complete_graph, path):
    indx = y*x_shape + x
    filtered_connected_nodes_with_weights = [(neighbor, complete_graph[indx][neighbor].get('weight', 1)) for neighbor in complete_graph.neighbors(indx) if neighbor not in path]
    next_indx, _ = min(filtered_connected_nodes_with_weights, key=lambda x: x[1])
    
    y = next_indx // x_shape
    x = next_indx % x_shape
    return x, y, next_indx, indx


def zig_zag_path(grid: np.ndarray, complete_graph: nx.Graph, nodes_without_obst: List[int]) -> List[int]:
    
    x_shape, y_shape = grid.shape
    path = [0] #starting from node 0 - (0,0)

    x = 0
    y = 0 
    while set(nodes_without_obst) - set(path): # untill not all nodes were visited 

        if can_i_go_up(x, y, x_shape, grid, path):
            x, y, path = going_up(x, y, grid, x_shape, y_shape, path)

        elif y % 2 == 0:# right 
            if x+1 < x_shape and grid[y, x+1] == 0 and y*x_shape + (x+1) not in path: #no obstyckle
                indx = y*x_shape + (x+1)
                path.append(indx)
                x += 1
            else: # obstyckle
                if can_i_go_down(x, y, y_shape, x_shape, grid, path):
                    x, y, path = going_down(x, y, grid, x_shape, y_shape, path)
                else: # there is obstyckle down or we aredy have been there 
                    # go to the closest one 
                    x, y, next_indx, indx = find_the_closest_one(x, y, x_shape, complete_graph, path)
                    
                    if 'path' not in complete_graph[indx][next_indx]:
                        part_path = [next_indx]
                    else:
                        part_path = complete_graph[indx][next_indx]['path'][::-1][1:]
                    path.extend(part_path)                  

        else:# left
            if x > 0 and grid[y, x-1] == 0 and y*x_shape + (x-1) not in path: #no obstyckle
                indx = y*x_shape + (x-1)
                x -= 1
                path.append(indx)

            else: # obstacle or end of board or we alredy have been there 
                if can_i_go_down(x, y, y_shape, x_shape, grid, path):
                    x, y, path = going_down(x, y, grid, x_shape, y_shape, path)
                else: # there is obstyckle down or we aredy have been there 
                # go to the closest one
                    x, y, next_indx, indx = find_the_closest_one(x, y, x_shape, complete_graph, path)
                    
                    if 'path' not in complete_graph[indx][next_indx]:
                        part_path = [next_indx]
                    else:
                        part_path = complete_graph[indx][next_indx]['path'][::-1][1:]
                    path.extend(part_path)
    
    return path


if __name__ == "__main__":
    # grid = make_grid(10, True, coverage = 0)
    grid = make_grid()
    plot_matrix(grid)
    graph, nodes_without_obst = create_networkX_graph(grid)

    # path = graph_random_walk(graph, nodes_without_obst)
    # print(path)
    # cost = sum(graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    # print("Total cost:", cost)
    # plot_graph(graph, path)

    # complete_graph = nx.Graph(graph)
    # for u in nodes_without_obst:
    #     for v in nodes_without_obst:
    #         if u != v and not graph.has_edge(u, v):
    #             path = nx.shortest_path(graph, source=u, target=v, weight='weight')
    #             length = nx.shortest_path_length(graph, source=u, target=v, weight='weight')
    #             complete_graph.add_edge(u, v, weight=length, path=path)
    
    # path = zig_zag_path(grid, complete_graph, nodes_without_obst)
    # print(path)
    # cost = sum(graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    # print("Total cost:", cost)
    # plot_graph(graph, path)

    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    # path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.christofides)
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

    tsp = nx.approximation.traveling_salesman_problem
    path = tsp(graph, cycle=True, nodes=nodes_without_obst, method=nx.approximation.traveling_salesman.greedy_tsp)
    plot_graph(graph, path)
    complete_graph = nx.Graph(graph)
    for u in nodes_without_obst:
        for v in nodes_without_obst:
            if u != v and not graph.has_edge(u, v):
                length = nx.shortest_path_length(graph, source=u, target=v, weight='weight')
                complete_graph.add_edge(u, v, weight=length)
            if u == v:
                complete_graph.add_edge(u, v, weight=1000)
                
    cycle = my_simulated_annealing_tsp(graph, path, move = "not-fully-connected", temp=500, alpha=0.001, max_iterations=40, N_inner=1000,)
    plot_graph(complete_graph, cycle)
    print(cycle==path)
    print(cycle)


    # cost = sum(graph.get_edge_data(path[i], path[i + 1]).get('weight', 1) for i in range(len(path) - 1))
    # print("Total cost:", cost)
