# Soft Actor-Critic



# Importing the libraries

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

# Setting the hyperparameters


actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.99
tau = 0.005
buffer_size = 1e6
batch_size = 128
alpha = 0.2  # Entropy coefficient
hidden_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def preprocess_state(state):
    # state: (96, 96, 3)
    state = np.transpose(state, (2, 0, 1))  # -> (3, 96, 96)
    state = state / 255.0
    return state

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calcul automatique de la taille de sortie
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            conv_out = self.conv(dummy)
            self.output_dim = conv_out.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)


# Implementing the Replay Buffer

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32)
    
    def __len__(self):
        return len(self.buffer)

# Building the Actor Network

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()

        self.fc1 = nn.Linear(self.encoder.output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.encoder(state)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        return action


# Building the Critic Network

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()

        self.fc1 = nn.Linear(self.encoder.output_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = self.encoder(state)
        x = torch.cat([x, action], dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return self.fc3(x)


# Building the SAC Agent

class SACAgent:

    def __init__(self):
        self.device = device
        self.actor = Actor().to(self.device)
        self.critic_1 = Critic().to(self.device)
        self.critic_2 = Critic().to(self.device)
        self.target_critic_1 = Critic().to(self.device)
        self.target_critic_2 = Critic().to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        state = preprocess_state(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor.sample(state)
    
        return action.cpu().numpy()[0]

    def update(self, batch_size, gamma=gamma, tau=tau, alpha=alpha):
        if len(self.replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = np.array([preprocess_state(s) for s in state])
        next_state = np.array([preprocess_state(s) for s in next_state])

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_state_action = self.actor.sample(next_state)
            target_q1_next = self.target_critic_1(next_state, next_state_action)
            target_q2_next = self.target_critic_2(next_state, next_state_action)
            target_q_min = torch.min(target_q1_next, target_q2_next) - alpha * torch.log(1 - next_state_action.pow(2) + 1e-6)
            target_q = reward + (1 - done) * gamma * target_q_min.to(self.device)
        # Update of the Critic 1 network
        current_q1 = self.critic_1(state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        # Update of the Critic 2 network
        current_q2 = self.critic_2(state, action)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        # Update of the Actor network
        entropy = torch.log(1 - self.actor.sample(state).pow(2) + 1e-6)
        actor_loss = (-self.critic_1(state, self.actor.sample(state)) + alpha * entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update of the Critic Target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Setting up the environment

#env = gym.make("CarRacing-v3")
env = gym.make("CarRacing-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Creating the agent

agent = SACAgent()

# Implementing the Training Loop

num_episodes = 100
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.update(batch_size)
        state = next_state
        episode_reward += reward
    print(f"Episode {episode}: Total Reward: {episode_reward}")
