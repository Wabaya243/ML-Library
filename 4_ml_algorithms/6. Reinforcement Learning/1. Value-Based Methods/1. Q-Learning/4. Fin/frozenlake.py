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

Q = np.zeros((state_space, action_space))

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
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ---------------------------------------------
        #       Q-LEARNING UPDATE
        # ---------------------------------------------
        # Mise à jour Bellman
        q_old = Q[state, action]
        q_target = reward + gamma * np.max(Q[next_state])
        
        Q[state, action] = q_old + alpha * (q_target - q_old)


        state = next_state
        total_reward += reward

    # Mise à jour epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    rewards_history.append(total_reward)

    if ep % 500 == 0:
        print(f"Episode {ep}: reward={total_reward}, epsilon={epsilon:.3f}")

print("\nTraining terminé.")

# Politique finale combinée
Q_final = Q

print("\n=== Q-table (Q-learning) ===")
print(np.round(Q_final, 3))

print("\n=== Politique greedy ===")
actions = ["←", "↓", "→", "↑"]
policy = np.array(actions)[np.argmax(Q_final, axis=1)]
print(policy.reshape(4, 4))  # pour FrozenLake-v1 4x4
