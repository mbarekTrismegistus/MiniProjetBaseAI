# astar.py
import heapq
import time

def manhattan(p, goal):

    return abs(p[0] - goal[0]) + abs(p[1] - goal[1]) # [cite: 86, 87]

def get_neighbors(node, grid):

    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    rows = len(grid)
    cols = len(grid[0])
    
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def astar(start, goal, grid, algo_type="A*", weight=1.0):
    start_time = time.time()
    
    open_set = []
    heapq.heappush(open_set, (0, 0, start, [start]))
    
    closed_set = set() 
    
    max_open_size = 0
    nodes_expanded = 0
    
    while open_set:
        if len(open_set) > max_open_size:
            max_open_size = len(open_set)
            
        f, g, current, path = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        nodes_expanded += 1
        
        if current == goal:
            execution_time = time.time() - start_time
            return path, g, nodes_expanded, execution_time, max_open_size
            
        for neighbor in get_neighbors(current, grid):
            if neighbor not in closed_set:
                tentative_g = g + 1
                
                # Configuration de f(n) selon l'algorithme choisi
                if algo_type == "UCS":
                    h = 0 # f(n) = g(n)
                    f_score = tentative_g
                elif algo_type == "Greedy":
                    h = manhattan(neighbor, goal)
                    tentative_g = 0 # On annule g(n) pour le tri
                    f_score = h
                else: # A* classique (W=1.0) ou Weighted A* (W>1.0)
                    h = manhattan(neighbor, goal)
                    f_score = tentative_g + weight * h # Intégration du poids W
                
                real_g = g + 1 
                heapq.heappush(open_set, (f_score, real_g, neighbor, path + [neighbor]))

    # Si aucun chemin n'est trouvé
    execution_time = time.time() - start_time
    return None, float('inf'), nodes_expanded, execution_time, max_open_size