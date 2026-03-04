# markov.py
import numpy as np
import random

def get_policy(path, goal):

    policy = {}
    for i in range(len(path) - 1):
        current = path[i]
        nxt = path[i+1]
        action = (nxt[0] - current[0], nxt[1] - current[1])
        policy[current] = action
    
    # Pour le but, on définit une action nulle ou on le gère séparément
    policy[goal] = (0, 0)
    return policy

def get_lateral_directions(action):
    dx, dy = action
    if dx != 0: # Déplacement horizontal
        return [(0, 1), (0, -1)]
    elif dy != 0: # Déplacement vertical
        return [(1, 0), (-1, 0)]
    return []

def build_transition_matrix(grid, policy, start, goal, epsilon):

    rows = len(grid)
    cols = len(grid[0])
    
    # Identifier tous les états accessibles (cellules libres)
    states = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                states.append((r, c))
                
    # Ajout des états absorbants [cite: 34]
    states.append("GOAL")
    states.append("FAIL")
    
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    # Initialisation de la matrice P avec des zéros
    P = np.zeros((n_states, n_states))
    
    for s in states:
        idx = state_to_idx[s]
        
        # États absorbants : P_GOAL,GOAL = 1 et P_FAIL,FAIL = 1 [cite: 35, 36]
        if s == "GOAL" or s == goal:
            P[idx, state_to_idx["GOAL"]] = 1.0
            continue
        if s == "FAIL":
            P[idx, state_to_idx["FAIL"]] = 1.0
            continue
            
        # Si l'état n'a pas de politique définie (hors chemin A*), 
        # on considère qu'il erre ou échoue. Pour simplifier, on l'envoie vers FAIL.
        if s not in policy:
            P[idx, state_to_idx["FAIL"]] = 1.0
            continue
            
        action = policy[s]
        lateral_actions = get_lateral_directions(action)
        
        # Distribution des probabilités selon le taux d'incertitude [cite: 32]
        transitions = [
            (action, 1.0 - epsilon),
            (lateral_actions[0], epsilon / 2.0),
            (lateral_actions[1], epsilon / 2.0)
        ]
        
        for act, prob in transitions:
            nx, ny = s[0] + act[0], s[1] + act[1]
            
            # (option) rester sur place si collision/obstacle [cite: 32]
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                next_state = (nx, ny)
            else:
                next_state = s # Reste sur place
                
            next_idx = state_to_idx[next_state]
            P[idx, next_idx] += prob
            
    return P, states, state_to_idx

def verify_stochastic(P):
    row_sums = np.sum(P, axis=1)
    return np.allclose(row_sums, 1.0)

def compute_pi_n(P, initial_idx, n):
    n_states = P.shape[0]
    pi_0 = np.zeros(n_states)
    pi_0[initial_idx] = 1.0
    
    P_n = np.linalg.matrix_power(P, n)
    pi_n = np.dot(pi_0, P_n)
    return pi_n

def analyze_absorption(P, states, state_to_idx):

    transient_states = [s for s in states if s not in ["GOAL", "FAIL"]]
    absorbing_states = ["GOAL", "FAIL"]
    
    t_idx = [state_to_idx[s] for s in transient_states]
    a_idx = [state_to_idx[s] for s in absorbing_states]
    
    Q = P[np.ix_(t_idx, t_idx)]
    R = P[np.ix_(t_idx, a_idx)]
    
    # Matrice fondamentale N = (I - Q)^(-1) [cite: 39]
    I = np.eye(len(t_idx))
    try:
        N = np.linalg.inv(I - Q)
        # Probabilités d'absorption B = N * R
        B = np.dot(N, R)
        # Temps moyen avant absorption t = N * 1
        t = np.dot(N, np.ones(len(t_idx)))
        return N, B, t, transient_states
    except np.linalg.LinAlgError:
        return None, None, None, None

def simulate_monte_carlo(start, goal, policy, grid, epsilon, N_simulations, max_steps=1000):

    rows = len(grid)
    cols = len(grid[0])
    
    success_count = 0
    fail_count = 0
    times_to_goal = []
    
    for _ in range(N_simulations):
        current = start
        steps = 0
        
        while steps < max_steps:
            if current == goal:
                success_count += 1
                times_to_goal.append(steps)
                break
                
            if current not in policy:
                fail_count += 1
                break
                
            action = policy[current]
            lateral_actions = get_lateral_directions(action)
            
            # Tirage au sort de l'action réelle selon epsilon
            rand_val = random.random()
            if rand_val < (1 - epsilon):
                act_taken = action
            elif rand_val < (1 - epsilon + epsilon/2):
                act_taken = lateral_actions[0]
            else:
                act_taken = lateral_actions[1]
                
            nx, ny = current[0] + act_taken[0], current[1] + act_taken[1]
            
            # Reste sur place si collision
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                current = (nx, ny)
                
            steps += 1
            
        if steps == max_steps:
            fail_count += 1 # Considéré comme échec si max_steps atteint
            
    p_goal = success_count / N_simulations
    p_fail = fail_count / N_simulations
    avg_time = sum(times_to_goal) / len(times_to_goal) if times_to_goal else float('inf')
    
    return p_goal, p_fail, avg_time