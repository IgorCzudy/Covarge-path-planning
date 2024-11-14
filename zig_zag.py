import numpy as np 
from typing import List
import networkx as nx



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
                # can_i_go_left():
                # go_left()
                if x > 0 and grid[y, x-1] == 0 and y*x_shape + (x-1) not in path: #no obstyckle
                    indx = y*x_shape + (x-1)
                    x -= 1
                    path.append(indx)

                elif can_i_go_down(x, y, y_shape, x_shape, grid, path):
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
                if x+1 < x_shape and grid[y, x+1] == 0 and y*x_shape + (x+1) not in path:# can i go right
                    #go right
                    indx = y*x_shape + (x+1)
                    path.append(indx)
                    x += 1
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
