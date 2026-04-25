import numpy as np
from settings import GRID_SIZE


class DoubleQAgent:
    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0):
        # Hyperparamètres RL
        self.lr = lr                # taux d'apprentissage
        self.gamma = gamma          # facteur de discount
        self.epsilon = epsilon      # exploration initiale
        self.min_epsilon = 0.1
        self.decay = 0.985          # diminution de l'exploration

        # Deux Q-tables indépendantes
        self.Q1 = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.Q2 = np.zeros((GRID_SIZE, GRID_SIZE, 4))

    def choose_action(self, state):
        # Politique epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)  # exploration

        x, y = state
        # Combinaison des deux Q-tables
        q_values = self.Q1[x, y] + self.Q2[x, y]
        return np.argmax(q_values)  # exploitation

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        # Valeur Q prédite
        # On choisit au hasard quelle table mettre à jour
        if np.random.rand() < 0.5:
            # Mise à jour de Q1
            a_max = np.argmax(self.Q1[nx, ny])          # action max selon Q1
            target = reward + self.gamma * self.Q2[nx, ny, a_max]  # Q2 pour la valeur
            self.Q1[x, y, action] += self.lr * (target - self.Q1[x, y, action])
        else:
            # Mise à jour de Q2
            a_max = np.argmax(self.Q2[nx, ny])          # action max selon Q2
            target = reward + self.gamma * self.Q1[nx, ny, a_max]  # Q1 pour la valeur
            self.Q2[x, y, action] += self.lr * (target - self.Q2[x, y, action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def get_policy(self):
        # Pour affichage ou exécution : argmax sur Q1+Q2
        return np.argmax(self.Q1 + self.Q2, axis=2)
