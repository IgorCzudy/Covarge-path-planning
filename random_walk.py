import random

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
