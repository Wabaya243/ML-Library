
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque


class PolicyNetwork(nn.Module):

  def __init__(self, state_size, action_size, seed = 42):
    super(PolicyNetwork, self).__init__()
    # Seed pour reproductibilité
    self.seed = torch.manual_seed(seed)
    # Couche d'entrée -> 64 neurones
    self.fc1 = nn.Linear(state_size, 128)
    # Couche cachée -> 64 neurones
    self.fc2 = nn.Linear(128, 64)
    # Couche cachée -> 32 neurones
    self.fc3 = nn.Linear(64, 32)
    # Couche de sortie : une Q-value par action possible
    self.fc4 = nn.Linear(32, action_size)

  def forward(self, state):
    # Forward pass : passage dans le réseau
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.relu(x)
    return torch.softmax(self.fc4(x), dim=1)   # Retourne les probabilités des actions


# Part 2 - Training the AI
# Partie 2 – Entraînement de l'IA

# Setting up the environment
# Configuration de l’environnement LunarLander-v3

import gymnasium as gym
env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n

print('State shape: ', state_shape)   # Format de l'état (8 valeurs)
print('State size: ', state_size)     # Nombre de caractéristiques
print('Number of actions: ', number_actions) # 4 actions possibles

# Initializing the hyperparameters
# Initialisation des hyperparamètres du DQN


# Implementing Experience Replay
# Implémentation du Replay Memory (mémoire d’expériences)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class REINFORCEAgent():

  def __init__(self, state_size, action_size, seed = 42, learning_rate = 1e-3, gamma=0.99):
    # Définir le device : GPU si disponible, sinon CPU
    self.device = device
    self.gamma = gamma

    # Nombre de caractéristiques de l'état (dimension de l'observation)
    self.state_size = state_size

    # Nombre d'actions possibles dans l'environnement
    self.action_size = action_size

    self.policy = PolicyNetwork(state_size, action_size).to(self.device)

    # Optimiseur Adam pour mettre à jour les poids du réseau principal
    self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)

    self.saved_log_probs = []
    self.rewards = []

  def act(self, state):
    # Convertir l'état en tenseur PyTorch et ajouter une dimension batch
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    # Obtenir les probabilités des actions à partir de la politique
    probs = self.policy(state)
    # Créer une distribution catégorique à partir des probabilités
    dist = torch.distributions.Categorical(probs)

    # Choisir une action en fonction de la distribution catégorique
    action = dist.sample()
    # Enregistrer le log de la probabilité de l'action choisie
    self.saved_log_probs.append(dist.log_prob(action))
    
    return action.item()

  def store_reward(self, reward):
    self.rewards.append(reward)
    
  def learn(self):
    # Calcul des retours cumulés à partir des récompenses reçues
    returns = []
    G = 0

    # Calcul des retours cumulés en partant de la fin
    for r in reversed(self.rewards):
        G = r + self.gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns).float().to(self.device)

    # normalisation (reduit la variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    # le 1e-8 est pour éviter la division par zéro

    #calcul de la loss (REINFORCE)
    loss = 0
    for log_prob, G in zip(self.saved_log_probs, returns):
        loss += -log_prob * G

    # Rétropropagation
    self.optimizer.zero_grad()  # Remise à zéro des gradients
    loss.backward()              # Calcul des gradients
    self.optimizer.step()        # Mise à jour des poids du réseau

    # Reset du Buffers
    self.saved_log_probs.clear()
    self.rewards.clear()
    


# Initialisation de l’agent avec les tailles d’entrée/sortie de l’environnement

agent = REINFORCEAgent(state_size, number_actions)

# Charger les poids sauvegardés
checkpoint = torch.load("Save/checkpoint.pth")

agent.policy.load_state_dict(checkpoint)

def play_episode(agent, env_name='LunarLander-v3', render=False):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    score = 0
    frames = []

    while not done:
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        score += reward

        if render:
            frames.append(env.render())

    env.close()

    if render:
        filename = 'Video/Loaded_model_video.mp4'
        imageio.mimsave(filename, frames, fps=30)
        print(f'Vidéo enregistrée : {filename}')

    return score

score = play_episode(agent, render=True)
print('Score obtenu :', score)
