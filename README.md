# Planification Robuste sur Grille : A* & Chaînes de Markov

Ce dépôt contient le code source et les expérimentations d'un mini-projet en Intelligence Artificielle combinant planification déterministe et modélisation stochastique. 

L'objectif est de faire naviguer un agent sur une grille 2D (contenant des obstacles) d'un point de départ s_0 vers un but g, tout en évaluant la robustesse du plan face à des incertitudes de déplacement (glissement, erreur d'action).

## Fonctionnalités

1. **Recherche Heuristique Déterministe :**
   - Implémentation des algorithmes **A***, **UCS** (Uniform Cost Search) et **Greedy**.
   - Utilisation de l'heuristique de Manhattan (admissible et cohérente).
   - Variante **Weighted A*** pour analyser le compromis entre vitesse d'exploration et optimalité.

2. **Modélisation Stochastique (Chaînes de Markov) :**
   - Déduction d'une politique déterministe pi à partir du chemin trouvé par A*.
   - Construction d'une matrice de transition P intégrant un taux d'erreur epsilon (probabilité de dévier latéralement).
   - Simulations de **Monte-Carlo** pour estimer empiriquement la probabilité de succès, le taux d'échec et le temps moyen pour atteindre le but.

## Architecture du Projet

Le projet est découpé en trois modules principaux :

* `astar.py` : Gère la grille, les heuristiques, et les algorithmes de recherche de chemin (A*, UCS, Greedy, Weighted A*).
* `markov.py` : Gère la création de la politique, la construction de la matrice de transition stochastique, et les simulations de Monte-Carlo.
* `experiments.py` : Script principal exécutant 4 scénarios de test pour évaluer et comparer les modèles.

## Prérequis et Installation

Ce projet nécessite **Python 3.x** et la bibliothèque **NumPy** pour les calculs matriciels.

1. Clonez ce dépôt :
   git clone https://github.com/votre-nom-utilisateur/planification-robuste-ia.git
   cd planification-robuste-ia

2. Installez les dépendances :
   pip install numpy

## Utilisation

Pour lancer toutes les expériences et afficher les résultats dans le terminal, exécutez simplement le script experiments.py :

python3 experiments.py

## Résultats des Expériences

Voici les résultats obtenus lors de l'exécution des scénarios de test détaillés dans le projet :

### E.1 : Comparaison des algorithmes (UCS, Greedy, A*)
Comparaison sur 3 grilles (Facile, Moyenne, Difficile).

| Grille | Algorithme | Coût | Nœuds explorés | Temps (s) | Mémoire (Max OPEN) |
|---|---|---|---|---|---|
| **Facile** (Vide) | UCS | 8 | 25 | ~0.00018 | 9 |
| | Greedy | 8 | 9 | ~0.00006 | 8 |
| | A* | 8 | 25 | ~0.00016 | 9 |
| **Moyenne** (Obstacles)| UCS | 8 | 18 | ~0.00006 | 3 |
| | Greedy | 8 | 9 | ~0.00006 | 4 |
| | A* | 8 | 16 | ~0.00007 | 3 |
| **Difficile** (Couloir)| UCS | 15 | 17 | ~0.00007 | 2 |
| | Greedy | 15 | 16 | ~0.00006 | 2 |
| | A* | 15 | 16 | ~0.00006 | 2 |

**Observation :** A* garantit l'optimalité tout en explorant moins de nœuds que UCS sur des grilles avec obstacles. Greedy est rapide mais sous-optimal.

### E.2 : Impact de l'incertitude (Markov) sur le plan A*
Sur la grille moyenne, le chemin optimal A* prévu a un coût de 8 pas. Voici les statistiques sur 1000 simulations de Monte-Carlo en variant le taux d'erreur epsilon.

| Epsilon | Proba Succès | Proba Échec | Temps moyen (pas) |
|---|---|---|---|
| 0.0 (Parfait) | 100.0% | 0.0% | 8.0 |
| 0.1 | 90.0% | 10.0% | 9.0 |
| 0.2 | 80.5% | 19.5% | 10.0 |
| 0.3 | 68.0% | 32.0% | 12.0 |

**Observation :** Un plan déterministe parfait devient très vulnérable dans un environnement incertain. À epsilon = 0.3, l'agent échoue dans près d'un tiers des cas.

### E.3 : Comparaison des heuristiques admissibles
Sur la grille moyenne :
* **Heuristique Nulle (h=0, UCS)** -> Coût: 8, Nœuds explorés: 18
* **Heuristique Manhattan (A*)** -> Coût: 8, Nœuds explorés: 16

**Observation :** L'heuristique de Manhattan réduit l'espace de recherche tout en garantissant l'optimalité (admissible et cohérente).

### E.4 : Weighted A* (Compromis vitesse vs optimalité)
Sur la grille difficile :
* **Poids W=1.0** : Coût 15, 16 nœuds explorés
* **Poids W=1.5** : Coût 15, 16 nœuds explorés
* **Poids W=2.0** : Coût 15, 16 nœuds explorés
* **Poids W=3.0** : Coût 15, 16 nœuds explorés

**Observation :** En raison de la topologie très contrainte de la grille difficile (couloir unique), il n'y a pas de chemins sous-optimaux alternatifs. Les performances restent donc identiques quel que soit le poids.

## Auteur
* **ETTALEBY M'barek** - *ENSETM - SDIA*
