import numpy as np
from settings import GRID_SIZE


class QAgent:
    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0):
        # Hyperparamètres RL
        self.lr = lr                # taux d'apprentissage
        self.gamma = gamma          # facteur de discount
        self.epsilon = epsilon      # exploration initiale
        self.min_epsilon = 0.1
        self.decay = 0.995          # diminution de l'exploration

        # Q-Table : GRID_SIZE x GRID_SIZE x 4 actions
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

    def choose_action(self, state):
        # Politique epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)  # exploration

        x, y = state
        return np.argmax(self.q_table[x, y])  # exploitation

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        # Valeur Q prédite
        predict = self.q_table[x, y, action]

        # Valeur cible (Bellman)
        target = reward + self.gamma * np.max(self.q_table[nx, ny])

        # Mise à jour Q-learning
        self.q_table[x, y, action] = predict + self.lr * (target - predict)

    def decay_epsilon(self):
        # L'exploration baisse au fil des épisodes
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
