import numpy as np
import random

# ============================================
#            ENVIRONNEMENT GRIDWORLD
# ============================================
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        """Réinitialise et retourne l'état initial sous forme (x, y)."""
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        """Effectue une action et retourne (nouvel état, récompense, done)."""

        x, y = self.pos

        # Déplacements limités dans la grille
        if action == 0:     # UP
            x = max(x - 1, 0)
        elif action == 1:   # DOWN
            x = min(x + 1, self.size - 1)
        elif action == 2:   # LEFT
            y = max(y - 1, 0)
        elif action == 3:   # RIGHT
            y = min(y + 1, self.size - 1)

        self.pos = (x, y)

        # But atteint
        if self.pos == self.goal:
            return self.pos, 10, True

        # Sinon pénalité de mouvement
        return self.pos, -1, False

    def state_to_index(self, state):
        """Convertit (x,y) → index entier de la Q-table."""
        x, y = state
        return x * self.size + y


# ============================================
#           DOUBLE Q-LEARNING
# ============================================

env = GridWorld(size=4)

n_states = env.size * env.size
n_actions = 4

# Deux Q-tables indépendantes
Q1 = np.zeros((n_states, n_actions))
Q2 = np.zeros((n_states, n_actions))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

episodes = 1000

for episode in range(episodes):
    state = env.reset()
    state_idx = env.state_to_index(state)
    done = False

    while not done:

        # ======================================================
        # POLITIQUE EPSILON-GREEDY sur (Q1 + Q2) = comportement
        # ======================================================
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)  # exploration
        else:
            action = np.argmax(Q1[state_idx] + Q2[state_idx])  # exploitation

        new_state, reward, done = env.step(action)
        new_state_idx = env.state_to_index(new_state)

        # ======================================================
        #     DOUBLE Q-LEARNING : mise à jour alternée
        # ======================================================
        if random.random() < 0.5:
            # -------------------------------
            # Mise à jour de Q1
            # Sélection de l'action greedy depuis Q1
            # mais évaluation par Q2 → évite la surestimation
            # -------------------------------
            best_a = np.argmax(Q1[new_state_idx])
            target = reward + gamma * Q2[new_state_idx, best_a]
            Q1[state_idx, action] += alpha * (target - Q1[state_idx, action])

        else:
            # -------------------------------
            # Mise à jour de Q2
            # -------------------------------
            best_a = np.argmax(Q2[new_state_idx])
            target = reward + gamma * Q1[new_state_idx, best_a]
            Q2[state_idx, action] += alpha * (target - Q2[state_idx, action])

        # Passer à l'état suivant
        state_idx = new_state_idx

    # Décroissance progressive de epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Q combiné pour visualisation
Q = Q1 + Q2


# ============================================
#              AFFICHAGE DES RÉSULTATS
# ============================================

print("\n=== Q-TABLE FINALE (Q1 + Q2) ===")
print(np.round(Q, 2))

print("\n=== POLITIQUE OPTIMALE ===")
actions = ["↑", "↓", "←", "→"]
policy = np.array(actions)[np.argmax(Q, axis=1)]
print(policy.reshape(env.size, env.size))
