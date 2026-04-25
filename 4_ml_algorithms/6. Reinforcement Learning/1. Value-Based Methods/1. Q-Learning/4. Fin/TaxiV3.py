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
gamma = 0.99     # discount
epsilon = 1.0    # exploration
min_epsilon = 0.01
decay = 0.9995

rewards_history = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Politique epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, truncated, info = env.step(action)
        
        # Mise à jour Bellman
        q_old = q_table[state, action]
        q_target = reward + gamma * np.max(q_table[next_state])
        
        q_table[state, action] = q_old + alpha * (q_target - q_old)
        print(env.render())
        
        state = next_state
        total_reward += reward
    
    # réduction epsilon
    epsilon = max(min_epsilon, epsilon * decay)
    rewards_history.append(total_reward)
    
    if ep % 500 == 0:
        print(f"Episode {ep} | Reward {total_reward} | Epsilon={epsilon:.3f}")

print("Training terminé.")

# Politique finale combinée
Q_final = Q

print("\n=== Q-table (Q-learning) ===")
print(np.round(Q_final, 3))

print("\n=== Politique greedy ===")
actions = ["←", "↓", "→", "↑"]
policy = np.array(actions)[np.argmax(Q_final, axis=1)]
print(policy.reshape(4, 4))  # pour FrozenLake-v1 4x4
