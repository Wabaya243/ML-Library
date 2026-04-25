# Taille de la grille (GRID_SIZE x GRID_SIZE)
GRID_SIZE = 8

# Taille en pixels de chaque cellule de la grille
CELL_SIZE = 100

# Vitesse d'affichage pygame
FPS = 30

# Récompenses utilisées par le Q-Learning
REWARD_GOAL = 10          # l'agent atteint l'objectif
REWARD_STEP = -0.1        # chaque pas coûte un peu
REWARD_OBSTACLE = -5      # collision avec obstacle

# Paramètres RL
EPISODES = 300
MAX_STEPS = 200

# Actions possibles : (dx, dy)
ACTIONS = {
    0: (-1, 0),  # haut
    1: (1, 0),   # bas
    2: (0, -1),  # gauche
    3: (0, 1),   # droite
}
