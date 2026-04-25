import numpy as np
from random import randint
import random

class EnvGrid(object):
    """
        Environnement grille 3x3 :
        0 = case neutre
        1 = objectif
        -1 = piège
    """
    def __init__(self):
        super(EnvGrid, self).__init__()

        self.grid = [
            [0, 0, 1], 
            [0, -1, 0],
            [0, 0, 0]
        ]

        self.y = 2
        self.x = 0

        self.actions = [
            [-1, 0],  # haut
            [1, 0],   # bas
            [0, -1],  # gauche
            [0, 1]    # droite
        ]

    def reset(self):
        self.y = 2
        self.x = 0
        return (self.y*3 + self.x + 1)

    def step(self, action):
        self.y = max(0, min(self.y + self.actions[action][0], 2))
        self.x = max(0, min(self.x + self.actions[action][1], 2))
        return (self.y*3 + self.x + 1), self.grid[self.y][self.x]

    def is_finished(self):
        return self.grid[self.y][self.x] == 1



# -----------------------------------------------------------
#  POLITIQUE : epsilon-greedy sur (Q1 + Q2)
# -----------------------------------------------------------
def take_action_doubleQ(st, Q1, Q2, eps):
    if random.uniform(0, 1) < eps:
        return randint(0, 3)
    return np.argmax(Q1[st] + Q2[st])



# -----------------------------------------------------------
#  BOUCLE DOUBLE Q-LEARNING
# -----------------------------------------------------------
if __name__ == '__main__':
    env = EnvGrid()

    # Q1 et Q2 avec 10 états (0 inutilisé)
    Q1 = np.zeros((10, 4))
    Q2 = np.zeros((10, 4))

    alpha = 0.1
    gamma = 0.9
    episodes = 200

    for _ in range(episodes):

        st = env.reset()

        while not env.is_finished():

            # Choix action epsilon-greedy sur Q1+Q2
            at = take_action_doubleQ(st, Q1, Q2, eps=0.4)

            stp1, r = env.step(at)

            # Mise à jour Double Q-learning
            if random.random() < 0.5:
                # --- Mise à jour Q1 ---
                best_a = np.argmax(Q1[stp1])
                target = r + gamma * Q2[stp1][best_a]
                Q1[st][at] += alpha * (target - Q1[st][at])
            else:
                # --- Mise à jour Q2 ---
                best_a = np.argmax(Q2[stp1])
                target = r + gamma * Q1[stp1][best_a]
                Q2[st][at] += alpha * (target - Q2[st][at])

            st = stp1

    # -----------------------------------------------------------
    #  AFFICHAGE FINALE
    # -----------------------------------------------------------
    print("Q1 :")
    for s in range(1, 10):
        print(s, Q1[s])

    print("\nQ2 :")
    for s in range(1, 10):
        print(s, Q2[s])

    print("\nPOLITIQUE (greedy sur Q1+Q2) :")
    for s in range(1, 10):
        print(s, np.argmax(Q1[s] + Q2[s]))
