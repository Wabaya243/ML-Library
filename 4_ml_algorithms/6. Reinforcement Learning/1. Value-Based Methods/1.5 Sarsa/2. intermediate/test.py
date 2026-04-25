import numpy as np
import random

# ============================
#    ENVIRONNEMENT GRIDWORLD
# ============================
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        """Réinitialise la position et retourne l'état (x, y)."""
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        """Applique l'action et retourne (nouvel état, récompense, done)."""
        x, y = self.pos

        if action == 0:     # UP
            x = max(x - 1, 0)
        elif action == 1:   # DOWN
            x = min(x + 1, self.size - 1)
        elif action == 2:   # LEFT
            y = max(y - 1, 0)
        elif action == 3:   # RIGHT
            y = min(y + 1, self.size - 1)

        self.pos = (x, y)

        if self.pos == self.goal:
            return self.pos, 10, True

        return self.pos, -1, False

    def state_to_index(self, state):
        """Convertit un tuple (x,y) en index d'état unique."""
        x, y = state
        return x * self.size + y

# ============================
#        SARSA
# ============================

env = GridWorld(size=4)

n_states = env.size * env.size
n_actions = 4

Q = np.zeros((n_states, n_actions))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    state_idx = env.state_to_index(state)

    # Choisir action initiale selon epsilon-greedy
    if random.random() < epsilon:
        action = random.randint(0, n_actions - 1)
    else:
        action = np.argmax(Q[state_idx])

    done = False

    while not done:
        # Exécuter l'action et observer le prochain état et récompense
        new_state, reward, done = env.step(action)
        new_state_idx = env.state_to_index(new_state)

        # Choisir action suivante selon epsilon-greedy (SARSA)
        if random.random() < epsilon:
            next_action = random.randint(0, n_actions - 1)
        else:
            next_action = np.argmax(Q[new_state_idx])

        # 🔹 MISE À JOUR SARSA
        td_target = reward + gamma * Q[new_state_idx][next_action]
        td_error = td_target - Q[state_idx][action]
        Q[state_idx][action] += alpha * td_error

        # Passer à l'état et action suivants
        state_idx = new_state_idx
        action = next_action

    # Décroissance de epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# ============================
# AFFICHAGE
# ============================

print("\n=== Q-TABLE FINALE (SARSA) ===")
print(np.round(Q, 2))

print("\n=== POLITIQUE OPTIMALE (SARSA) ===")
actions = ["↑", "↓", "←", "→"]
policy = np.array(actions)[np.argmax(Q, axis=1)]
print(policy.reshape(env.size, env.size))
