import gymnasium as gym
import numpy as np
import random

env = gym.make("Taxi-v3", render_mode="ansi")

# Création de la Q-table
state_space = env.observation_space.n
action_space = env.action_space.n

q_table = np.zeros((state_space, action_space))

# Hyperparamètres RL
episodes = 5000
alpha = 0.1      # learning rate
gamma = 0.99     # discount factor
epsilon = 1.0    # exploration
min_epsilon = 0.01
decay = 0.9995

rewards_history = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    # 🔹 Choisir action initiale selon epsilon-greedy
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])

    while not done:
        next_state, reward, done, truncated, info = env.step(action)

        # 🔹 Choisir action suivante selon epsilon-greedy (SARSA)
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state])

        # 🔹 MISE À JOUR SARSA
        td_target = reward + gamma * q_table[next_state, next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += alpha * td_error

        state = next_state
        action = next_action
        total_reward += reward

        print(env.render())  # affichage de l'environnement

    # Décroissance epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    rewards_history.append(total_reward)

    if ep % 500 == 0:
        print(f"Episode {ep} | Reward {total_reward} | Epsilon={epsilon:.3f}")

print("Training terminé.")

# Politique finale SARSA
Q_final = q_table

print("\n=== Q-table (SARSA) ===")
print(np.round(Q_final, 3))

# Pour Taxi, les actions possibles sont :
actions = ["South", "North", "East", "West", "Pickup", "Dropoff"]

# Politique greedy finale
policy = np.array(actions)[np.argmax(Q_final, axis=1)]
print(policy)  # pas de reshape, Taxi a 500 états différents
