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
        """Always return a tuple (x, y)."""
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
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
        """Convert tuple (x,y) to a state index."""
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError(f"State must be (x,y), got: {state}")

        x, y = state
        return x * self.size + y


# ============================
#        Q-LEARNING
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

    done = False

    while not done:
        # EPSILON-GREEDY
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state_idx])

        new_state, reward, done = env.step(action)
        new_state_idx = env.state_to_index(new_state)

        # Q UPDATE
        td_target = reward + gamma * np.max(Q[new_state_idx])
        td_error = td_target - Q[state_idx][action]
        Q[state_idx][action] += alpha * td_error

        state_idx = new_state_idx

    epsilon = max(epsilon_min, epsilon * epsilon_decay)


print("\n=== Q-TABLE FINALE ===")
print(np.round(Q, 2))

print("\n=== POLITIQUE OPTIMALE ===")
actions = ["↑", "↓", "←", "→"]
policy = np.array(actions)[np.argmax(Q, axis=1)]
print(policy.reshape(env.size, env.size))
