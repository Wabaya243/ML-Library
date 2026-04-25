import gymnasium as gym
import numpy as np
import random

# ============================
#   ENVIRONNEMENT FROZENLAKE
# ============================

# Option glissant ou non glissant :
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")  

state_space = env.observation_space.n
action_space = env.action_space.n

# ============================
#   DOUBLE Q-TABLES
# ============================
Q1 = np.zeros((state_space, action_space))
Q2 = np.zeros((state_space, action_space))

# Hyperparamètres RL
episodes = 8000
alpha = 0.1
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
decay = 0.9995

rewards_history = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    env.render()

    while not done:
        # ---------------------------------------------
        #   POLITIQUE EPSILON-GREEDY sur (Q1 + Q2)
        # ---------------------------------------------
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q1[state] + Q2[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---------------------------------------------
        #       DOUBLE Q-LEARNING UPDATE
        # ---------------------------------------------
        if random.random() < 0.5:
            # Mise à jour de Q1
            best_a = np.argmax(Q1[next_state])
            target = reward + gamma * Q2[next_state, best_a]
            Q1[state, action] += alpha * (target - Q1[state, action])
        else:
            # Mise à jour de Q2
            best_a = np.argmax(Q2[next_state])
            target = reward + gamma * Q1[next_state, best_a]
            Q2[state, action] += alpha * (target - Q2[state, action])

        state = next_state
        total_reward += reward

    # Mise à jour epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    rewards_history.append(total_reward)

    if ep % 500 == 0:
        print(f"Episode {ep}: reward={total_reward}, epsilon={epsilon:.3f}")

print("\nTraining terminé.")

# Politique finale combinée
Q_final = Q1 + Q2

print("\n=== Q-table (Double Q-learning) ===")
print(np.round(Q_final, 3))

print("\n=== Politique greedy ===")
actions = ["←", "↓", "→", "↑"]
policy = np.array(actions)[np.argmax(Q_final, axis=1)]
print(policy.reshape(4, 4))  # pour FrozenLake-v1 4x4
