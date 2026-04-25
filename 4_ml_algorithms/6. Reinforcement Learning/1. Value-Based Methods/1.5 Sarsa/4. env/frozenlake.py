import gymnasium as gym
import numpy as np
import random

# ============================
#   ENVIRONNEMENT FROZENLAKE
# ============================

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")  

state_space = env.observation_space.n
action_space = env.action_space.n

# Initialisation Q-table
Q = np.zeros((state_space, action_space))

# Hyperparamètres
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

    # 🔹 Choisir action initiale selon epsilon-greedy
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    while not done:
        # Exécuter l'action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 🔹 Choisir action suivante selon epsilon-greedy (SARSA)
        if random.random() < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[next_state])

        # 🔹 MISE À JOUR SARSA
        td_target = reward + gamma * Q[next_state, next_action]
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        # Passer à l'état et action suivants
        state = next_state
        action = next_action
        total_reward += reward

    # Décroissance de epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    rewards_history.append(total_reward)

    if ep % 500 == 0:
        print(f"Episode {ep}: reward={total_reward}, epsilon={epsilon:.3f}")

print("\nTraining terminé.")

# Politique finale SARSA
Q_final = Q

print("\n=== Q-table (SARSA) ===")
print(np.round(Q_final, 3))

print("\n=== Politique greedy ===")
actions = ["←", "↓", "→", "↑"]
policy = np.array(actions)[np.argmax(Q_final, axis=1)]
print(policy.reshape(4, 4))  # pour FrozenLake-v1 4x4
