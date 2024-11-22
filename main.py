import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import random 

def make_3d_grid(size: int = 10, random: bool = False, coverage: float = 0.1) -> np.ndarray:
    grid = np.zeros((size, size, size), dtype=int)

    if not random and size > 5:
        grid[1,1,1] = 1
        # grid[4,5] = 1
        # grid[5,5] = 1

        # grid[8,8] = 1
        # grid[9,9] = 1

        # grid[2,8] = 1
        # grid[2,7] = 1
        # grid[1,8] = 1
        # grid[1,7] = 1
    else:
        total_cells = size * size * size
        num_obstacles = int(total_cells * coverage)

        obstacles = set()
        while len(obstacles) < num_obstacles:
            row = np.random.randint(0, size)
            col = np.random.randint(0, size)
            high = np.random.randint(0, size)
            obstacles.add((row, col, high))

        for row, col, high in obstacles:
            grid[row, col, high] = 1

    return grid


def map_corinats_to_node_number(x, y, z, grid):
    x_shape, y_shape, z_shape = grid.shape 
    if not (0 <= x < x_shape and 0 <= y < y_shape and 0 <= z < z_shape):
        raise ValueError(f"Coordinates ({x}, {y}, {z}) are out of bounds for grid shape {grid.shape}.")

    return x * (y_shape * z_shape) + y * z_shape + z


def map_node_number_to_coordinates(node_number, grid):
    x_shape, y_shape, z_shape = grid.shape 

    max_node_number = x_shape * y_shape * z_shape - 1
    if not (0 <= node_number <= max_node_number):
        raise ValueError(f"Node number {node_number} is out of bounds for grid shape {grid.shape}.")

    x = node_number // (y_shape * z_shape)
    remaining = node_number % (y_shape * z_shape)
    y = remaining // z_shape
    z = remaining % z_shape

    return (x, y, z)


def plot_3d_grid(grid, path = [0,1,2,3]):
    dark_indices = np.argwhere(grid == 1)
    light_indices = np.argwhere(grid == 0)

    fig = go.Figure()

    if dark_indices.size > 0:
        fig.add_trace(go.Scatter3d(
            x=dark_indices[:, 0],
            y=dark_indices[:, 1],
            z=dark_indices[:, 2],
            mode='markers+text',
            # text=[f"({x}, {y}, {z} - {map_corinats_to_node_number(x, y, z, grid)})" for x, y, z in dark_indices],
            textposition='top center',
            marker=dict(size=5, color='black'),
            name='1', 
            visible=True  
        ))

    if light_indices.size > 0:
        fig.add_trace(go.Scatter3d(
            x=light_indices[:, 0],
            y=light_indices[:, 1],
            z=light_indices[:, 2],
            mode='markers+text',
            # text=[f"({x}, {y}, {z} - {map_corinats_to_node_number(x, y, z, grid)})" for x, y, z in light_indices],
            textposition='top center',
            marker=dict(size=5, color='lightgray'),
            name='0',
            visible=True
        ))

    arrows = []
    lines = []
    path_with_xyz_corr = [map_node_number_to_coordinates(i, grid) for i in path]

    for (x, y, z), (next_x, next_y, next_z) in zip(path_with_xyz_corr, path_with_xyz_corr[1:]):
        dx = next_x - x
        dy = next_y - y
        dz = next_z - z

        if dx != 0:  
            dx += 0.5
        elif dy != 0:
            dy += 0.5
        elif dz != 0:
            dz += 0.5

        arrow = go.Cone(
            x=[x],
            y=[y],
            z=[z],
            u=[dx],
            v=[dy],
            w=[dz],
            colorscale=[[0, 'red'], [1, 'red']],
            sizemode="absolute",
            sizeref=0.2,
            anchor="tail",
            visible=True  # Initially visible
        )
        fig.add_trace(arrow)
        arrows.append(arrow)

        line = go.Scatter3d(
            x=[x, next_x],
            y=[y, next_y],
            z=[z, next_z],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Line 1',
            visible=True
        )
        fig.add_trace(line)
        lines.append(line)

    steps = []
    num_elements = len(arrows) + len(lines)
    for i, arrow in enumerate(arrows):
        
        
        visibility = [False for i in range(num_elements)]
        for j in range(2 * (i + 1)):  # Each state includes both arrow and line visibility
            visibility[j] = True
        

        steps.append({
            'label': f'Step {i + 1}',
            'method': 'update',
            'args': [{'visible': [True, True] + visibility}]
        })
        

    fig.update_layout(
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'visible': True,
                'prefix': 'State:',
                'font': {
                    'size': 20,
                    'color': 'black'
                }
            },
            'steps': steps
        }],
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title='Interactive 3D Grid with Arrows'
    )

    fig.show()

def build_graph(grid):
    x_shape, y_shape, z_shape = grid.shape 

    G = nx.Graph()
    G.add_nodes_from([i for i in range(x_shape * y_shape * z_shape)]) # umber of nodes is x_shape * y_shape * z_shape

    moves = [   (0,0,1), # up 
        (-1,0,0),         (1,0,0), #left, center, right
                (0,0,-1), # down
            (0,1,0), (0,-1,0)] #forward, back
    

    for x in range(x_shape):
        for y in range(y_shape):
            for z in range(z_shape):
                
                for move in moves:
                    move_x, move_y, move_z = x + move[0], y + move[1], z + move[2]
                    if 0 <= move_x < x_shape and \
                        0 <= move_y < y_shape and \
                        0 <= move_z < z_shape:
                            if grid[move_x, move_y, move_z] != 1 and grid[x, y, z] != 1: #there is no obstyckle there

                                number_of_curr_node = map_corinats_to_node_number(x, y, z, grid)
                                number_of_node_to_move = map_corinats_to_node_number(move_x, move_y, move_z, grid)
                                G.add_edge(number_of_curr_node, number_of_node_to_move)
    
    return G
                


def plot_graph(graph, nodes_that_are_obtyckle):

    fig = go.Figure()

    nodes_xyz_corr_dark = np.array([map_node_number_to_coordinates(i, grid) for i in graph.nodes if i in nodes_that_are_obtyckle])
    nodes_xyz_corr_light = np.array([map_node_number_to_coordinates(i, grid) for i in graph.nodes if i not in nodes_that_are_obtyckle])    

    if nodes_xyz_corr_dark.size > 0:
        fig.add_trace(go.Scatter3d(
            x=nodes_xyz_corr_dark[:, 0],
            y=nodes_xyz_corr_dark[:, 1],
            z=nodes_xyz_corr_dark[:, 2],
            mode='markers+text',
            text=[f"({x}, {y}, {z} - {map_corinats_to_node_number(x, y, z, grid)})" for x, y, z in nodes_xyz_corr_dark],
            textposition='top center',
            marker=dict(size=5, color='black'),
            name='1', 
            visible=True  
        ))

    if nodes_xyz_corr_light.size > 0:
        fig.add_trace(go.Scatter3d(
            x=nodes_xyz_corr_light[:, 0],
            y=nodes_xyz_corr_light[:, 1],
            z=nodes_xyz_corr_light[:, 2],
            mode='markers+text',
            text=[f"({x}, {y}, {z} - {map_corinats_to_node_number(x, y, z, grid)})" for x, y, z in nodes_xyz_corr_light],
            textposition='top center',
            marker=dict(size=5, color='lightgray'),
            name='0',
            visible=True
        ))


    for i, node in enumerate(graph):
        for neighbour in graph[node]:
            x, y, z = map_node_number_to_coordinates(node, grid)
            next_x, next_y, next_z = map_node_number_to_coordinates(neighbour, grid)
            line = go.Scatter3d(
                x=[x, next_x],
                y=[y, next_y],
                z=[z, next_z],
                mode='lines',
                line=dict(color='blue', width=4),
                name='Line 1',
                visible=True
            )
            fig.add_trace(line)

    fig.show()

def random_walk(graph, nodes_that_are_not_obtyckle):
    random.seed(42)
    path = [0] # start with first node 
    visited = set([0])
    all_nodes = set(nodes_that_are_not_obtyckle)

    while visited < all_nodes:
        possible_new_moves = set(graph[path[-1]]) - visited
        
        if possible_new_moves: # if its not empty 
            next = random.choice(list(possible_new_moves))
            visited.add(next)
            path.append(next)
        
        else: # we need to come back to already visited node 
            possible_moves = graph[path[-1]]
            if not possible_moves: #list is empty
                raise ValueError(f"Node {path[-1]} does not have any neighboors. No possible path")
            next = random.choice(list(possible_moves))
            visited.add(next)
            path.append(next)

    return path 


if __name__ == "__main__":
    grid = make_3d_grid(random=True)
    print(grid)
    
    # plot_3d_grid(grid)

    graph = build_graph(grid)
    nodes_that_are_obtyckle = [ map_corinats_to_node_number(x,y,z,grid) for x, y, z in np.argwhere(grid == 1)]
    nodes_that_are_not_obtyckle = list(set(graph.nodes) - set(nodes_that_are_obtyckle))
    # plot_graph(graph, nodes_that_are_obtyckle)
    
    path = random_walk(graph, nodes_that_are_not_obtyckle)
    print(path)

    # tsp = nx.approximation.traveling_salesman_problem
    # path = tsp(graph, cycle=True, nodes=nodes_that_are_not_obtyckle, method=nx.approximation.traveling_salesman.christofides)
    # path = path[:path.index(0)-1]
    # print(path)

    plot_3d_grid(grid, path)
