# experiments.py
import astar
import markov
import time

# --- Définition des 3 grilles pour l'expérience E.1 ---
# 0 = Libre, 1 = Obstacle

# Grille Facile (chemin direct possible)
grid_easy = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

# Grille Moyenne (quelques obstacles)
grid_medium = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
]

# Grille Difficile (labyrinthe ou cul-de-sac)
grid_hard = [
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0]
]

def run_E1():
    """
    E.1 : Comparer UCS vs Greedy vs A* sur 3 grilles (facile/moyenne/difficile).
    """
    print("=== E.1: Comparaison des algorithmes de planification ===")
    grids = {"Facile": grid_easy, "Moyenne": grid_medium, "Difficile": grid_hard}
    algos = ["UCS", "Greedy", "A*"]
    
    for name, grid in grids.items():
        print(f"\n--- Grille: {name} ---")
        start = (0, 0)
        goal = (len(grid)-1, len(grid[0])-1)
        
        for algo in algos:
            path, cost, nodes, exec_time, mem = astar.astar(start, goal, grid, algo_type=algo)
            print(f"{algo:7} | Coût: {cost:3} | Nœuds explorés: {nodes:3} | Temps: {exec_time:.5f}s | Mémoire (Max OPEN): {mem}")

def run_E2():
    """
    E.2 : Fixer A* et varier l'incertitude epsilon.
    Mesurer le coût prévu vs la probabilité réelle d'atteindre GOAL.
    """
    print("\n=== E.2: Impact de l'incertitude (Markov) sur le plan A* ===")
    grid = grid_medium
    start = (0, 0)
    goal = (len(grid)-1, len(grid[0])-1)
    
    # 1. Planification déterministe avec A*
    path, cost, _, _, _ = astar.astar(start, goal, grid, algo_type="A*")
    print(f"Chemin A* prévu : {path}")
    print(f"Coût prévu (déterministe) : {cost}")
    
    # 2. Création de la politique
    policy = markov.get_policy(path, goal)
    
    # 3. Variation de epsilon
    epsilons = [0, 0.1, 0.2, 0.3]
    N_simulations = 1000
    
    for eps in epsilons:
        p_goal, p_fail, avg_time = markov.simulate_monte_carlo(start, goal, policy, grid, eps, N_simulations)
        print(f"Epsilon: {eps} | Proba Succès: {p_goal*100:.1f}% | Proba Échec: {p_fail*100:.1f}% | Temps moyen: {avg_time:.1f} pas")

def run_E3():
    """
    E.3 : Tester deux heuristiques admissibles (h=0 vs Manhattan) et comparer expansions.
    Note : h=0 correspond exactement à UCS.
    """
    print("\n=== E.3: Comparaison des heuristiques admissibles (h=0 vs Manhattan) ===")
    grid = grid_medium
    start = (0, 0)
    goal = (len(grid)-1, len(grid[0])-1)
    
    # h = 0 (UCS)
    _, cost_ucs, nodes_ucs, _, _ = astar.astar(start, goal, grid, algo_type="UCS")
    
    # h = Manhattan (A*)
    _, cost_astar, nodes_astar, _, _ = astar.astar(start, goal, grid, algo_type="A*")
    
    print(f"Heuristique Nulle (h=0)       -> Coût: {cost_ucs}, Nœuds explorés: {nodes_ucs}")
    print(f"Heuristique Manhattan (A*)  -> Coût: {cost_astar}, Nœuds explorés: {nodes_astar}")
    print("Observation : L'heuristique de Manhattan réduit considérablement l'espace de recherche tout en garantissant l'optimalité (admissible).")

def run_E4():
    print("\n=== E.4: Weighted A* (Compromis vitesse vs optimalité) ===")
    grid = grid_hard # On utilise la grille difficile pour bien voir la différence
    start = (0, 0)
    goal = (len(grid)-1, len(grid[0])-1)
    
    weights = [1.0, 1.5, 2.0, 3.0]
    
    for w in weights:
        path, cost, nodes, exec_time, mem = astar.astar(start, goal, grid, algo_type="A*", weight=w)
        if path:
            print(f"Poids W={w:<3} | Coût: {cost:3} | Nœuds explorés: {nodes:3} | Temps: {exec_time:.5f}s")
        else:
            print(f"Poids W={w:<3} | Aucun chemin trouvé.")

if __name__ == "__main__":
    run_E1()
    run_E2()
    run_E3()
    run_E4()