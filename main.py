import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx


def create_adjacency_matrix(grid):
    moves = [(1,1), (1,0), (0,1), (1,-1), (-1,-1), (-1,0), (0,-1), (-1,1)]
    height, width = grid.shape
    number_of_cells = height * width
    adjacency_matrix = np.zeros((number_of_cells, number_of_cells))

    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                continue
            else:
                if (i*height+j) %width-1 >=0:
                    adjacency_matrix[i*height+j][(i*height+j)-1] = 1
                
                if (i*height+j)%width+1 < width:
                    adjacency_matrix[i*height+j][(i*height+j)+1] = 1
                
                if (i*height+j) - width >=0:
                    adjacency_matrix[i*height+j][(i*height+j)-width] = 1
                
                if (i*height+j)+width < number_of_cells:
                    adjacency_matrix[i*height+j][(i*height+j)+width] = 1
                
                if (i*height+j)%width-1 >=0 and (i*height+j) - width >=0: 
                    adjacency_matrix[i*height+j][(i*height+j)-width-1] = 1
                
                if (i*height+j) - width >=0 and (i*height+j)%width+1 < width:
                    adjacency_matrix[i*height+j][(i*height+j)-width+1] = 1
                
                if (i*height+j)+width < number_of_cells and (i*height+j)%width-1 >=0: 
                    adjacency_matrix[i*height+j][(i*height+j)+width-1] = 1
                
                if (i*height+j)%width+1 < width and (i*height+j)+width < number_of_cells:
                    adjacency_matrix[i*height+j][(i*height+j)+width+1] = 1

    obstacles = [i*height+j for i in range(height) for j in range(width) if grid[i][j] == 1]
    for idx in obstacles:
        adjacency_matrix[idx, :] = 0
        adjacency_matrix[:, idx] = 0


    return adjacency_matrix


def create_networkX_graph(grid):
    height, width = grid.shape
    number_of_cells = height * width

    obstacles = [i*height+j for i in range(height) for j in range(width) if grid[i][j] == 1]

    G = nx.Graph()
    G.add_nodes_from([i for i in range(number_of_cells)])

    edges_to_add = []
    for node_number in G.nodes:
        moves = []
        if node_number%width-1 >=0:
            moves.append(node_number-1) 

        if node_number%width+1 < width:
            moves.append(node_number+1)

        if node_number - width >=0:
            moves.append(node_number-width )

        if node_number+width < number_of_cells:
            moves.append(node_number+width)

        if node_number%width-1 >=0 and node_number - width >=0: 
            moves.append(node_number-width-1)

        if node_number - width >=0 and node_number%width+1 < width:
            moves.append(node_number-width+1)

        if node_number+width < number_of_cells and node_number%width-1 >=0: 
            moves.append(node_number+width-1)

        if node_number%width+1 < width and node_number+width < number_of_cells:
            moves.append(node_number+width+1)


        list_of_edges = list(zip([node_number] * len(moves), moves))
        edges_to_add.extend(list_of_edges)
    
    
    filtered_edges = [edge for edge in edges_to_add if edge[0] not in obstacles and edge[1] not in obstacles]
    G.add_edges_from(filtered_edges)
    return G, [i for i in range(number_of_cells) if i not in obstacles]


def plot_graph(graph, path):
    grid_size = 10
    pos = {i: (i % grid_size, grid_size - 1 - (i // grid_size)) for i in range(grid_size * grid_size)}

    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos, with_labels=True, node_size=100, node_color='lightblue', font_size=10)
    
    path_nodes = set(path)
    nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_color='orange', node_size=200)
    
    # Highlight the path edges
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=3)

    plt.title('NetworkX Graph of the 10x10 Grid with (0, 0) at Top-Left Corner')
    plt.show()


def plot_matrix(grid):
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


def make_grid(size=10, random=False, coverage=0.1):
    grid = np.zeros((size, size), dtype=int)

    if not random:
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



grid = make_grid()

# plot_matrix(grid)

graph, nodes_without_obst = create_networkX_graph(grid)
# adjacency_matrix = create_adjacency_matrix(grid)
# graph = nx.from_numpy_array(adjacency_matrix)

tsp = nx.approximation.traveling_salesman_problem
path = tsp(graph, cycle=False, nodes=nodes_without_obst)
print(path)


plot_graph(graph, path)
