import numpy as np
from random import randint
import random

class EnvGrid(object):
    """
        Environnement de type grille pour un agent RL.
        0 = case neutre
        1 = objectif (fin de l'épisode)
        -1 = piège / mauvaise case
    """
    def __init__(self):
        super(EnvGrid, self).__init__()

        # Définition de la grille (3x3)
        self.grid = [
            [0, 0, 1],   # Ligne 0
            [0, -1, 0],  # Ligne 1
            [0, 0, 0]    # Ligne 2
        ]

        # Position de départ (ligne 2, colonne 0)
        self.y = 2
        self.x = 0

        # Définition des actions possibles
        # Chaque action est un déplacement (dy, dx)
        self.actions = [
            [-1, 0],  # 0 = Haut
            [1, 0],   # 1 = Bas
            [0, -1],  # 2 = Gauche
            [0, 1]    # 3 = Droite
        ]

    def reset(self):
        """
            Réinitialise la position de l'agent au point de départ.
            Retourne l'état initial sous forme d'indice (1 à 9).
        """
        self.y = 2
        self.x = 0
        return (self.y*3 + self.x + 1)

    def step(self, action):
        """
            Effectue une action (0-3) et met à jour la position.
            Retourne :
                - le nouvel état
                - la récompense de la case atteinte
        """
        # Appliquer déplacement avec bornes pour ne pas sortir de la grille
        self.y = max(0, min(self.y + self.actions[action][0], 2))
        self.x = max(0, min(self.x + self.actions[action][1], 2))

        # Retourne l'état (index 1-9) et la récompense
        return (self.y*3 + self.x + 1), self.grid[self.y][self.x]

    def show(self):
        """
            Affiche la grille dans la console.
            Le symbole "X" indique la position actuelle de l'agent.
        """
        print("---------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                print("%s\t" % (pt if y != self.y or x != self.x else "X"), end="")
                x += 1
            y += 1
            print("")

    def is_finished(self):
        """
            Vérifie si l'agent est sur la case objectif (1).
        """
        return self.grid[self.y][self.x] == 1

def take_action(st, Q, eps):
    """
        Choisit une action selon la stratégie epsilon-greedy.
        eps = probabilité de choisir une action aléatoire.
    """
    if random.uniform(0, 1) < eps:
        # Exploration : action au hasard
        action = randint(0, 3)
    else:
        # Exploitation : meilleure action selon Q
        action = np.argmax(Q[st])
    return action

if __name__ == '__main__':
    env = EnvGrid()
    st = env.reset()

    # Table Q initialisée à 0 (10 états x 4 actions)
    Q = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    # Entraînement pendant 100 épisodes
    for _ in range(100):
        st = env.reset()  # Réinitialiser l'environnement

        while not env.is_finished():
            # Choisir une action avec epsilon = 0.4 (exploration modérée)
            at = take_action(st, Q, 0.4)

            # Exécuter l'action
            stp1, r = env.step(at)

            # Choisir action suivante de manière totalement greedy (eps=0)
            atp1 = take_action(stp1, Q, 0.0)

            # Mise à jour de la table Q
            # Formule : Q(s,a) += alpha * (r + gamma*Q(s',a') - Q(s,a))
            Q[st][at] = Q[st][at] + 0.1 * (r + 0.9 * Q[stp1][atp1] - Q[st][at])

            st = stp1  # Passer au prochain état

    # Afficher la Q-table apprise (états 1 à 9)
    for s in range(1, 10):
        print(s, Q[s])
