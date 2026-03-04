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

### Détail des Expériences Exécutées :
* **E.1** : Comparaison des performances (Coût, Nœuds explorés, Temps, Mémoire) de UCS, Greedy et A* sur 3 grilles (Facile, Moyenne, Difficile).
* **E.2** : Fixation du plan A* et variation du niveau d'incertitude markovienne (epsilon) pour mesurer la chute de la probabilité de réussite.
* **E.3** : Comparaison de l'expansion de l'espace de recherche entre une heuristique nulle (h=0) et l'heuristique de Manhattan.
* **E.4** : Test de l'algorithme Weighted A* avec différents poids (W).


## Auteur
* **ETTALEBY M'barek** - *SDIA*
