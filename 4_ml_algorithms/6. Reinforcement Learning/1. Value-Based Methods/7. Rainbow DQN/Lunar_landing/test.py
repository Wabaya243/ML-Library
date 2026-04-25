
import os
import random
from re import A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple
import imageio
import gymnasium as gym

class Network(nn.Module):

  def __init__(self, state_size, action_size, seed = 42):
    super(Network, self).__init__()
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
    return self.fc4(x)   # Retourne les Q-values

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

learning_rate = 5e-4               # Taux d’apprentissage
minibatch_size = 100               # Taille d’un batch dans le Replay Memory
discount_factor = 0.99             # Gamma (réduction future)
replay_buffer_size = int(1e5)      # Taille du replay buffer
interpolation_parameter = 1e-3     # Tau pour le soft update du réseau cible

# Implementing Experience Replay
# Implémentation du Replay Memory (mémoire d’expériences)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

  def __init__(self, capacity):
    self.device = device
    self.capacity = capacity        # Taille maximale
    self.memory = []                # Tableau qui stocke les transitions

  def push(self, event):
    # Ajoute une nouvelle transition
    self.memory.append(event)
    # Si la mémoire dépasse la capacité, on supprime la plus ancienne
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def sample(self, batch_size):
    # Sélectionne un batch aléatoire dans la mémoire
    experiences = random.sample(self.memory, k = batch_size)

    # Convertit les batchs en tenseurs PyTorch
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

    return states, next_states, actions, rewards, dones

# Implementing the DQN class
# Implémentation de l’agent DQN

class Agent():

  def __init__(self, state_size, action_size):
    # Définir le device : GPU si disponible, sinon CPU
    self.device = device

    # Nombre de caractéristiques de l'état (dimension de l'observation)
    self.state_size = state_size

    # Nombre d'actions possibles dans l'environnement
    self.action_size = action_size

    # Réseau Q principal (online network) : prédit les Q-values pour chaque action
    self.local_qnetwork = Network(state_size, action_size).to(self.device)

    # Réseau Q cible (target network) : utilisé pour le calcul des Q-cibles et mis à jour lentement
    self.target_qnetwork = Network(state_size, action_size).to(self.device)

    # Optimiseur Adam pour mettre à jour les poids du réseau principal
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)

    # Replay Memory pour stocker les expériences (state, action, reward, next_state, done)
    self.memory = ReplayMemory(replay_buffer_size)

    # Compteur pour décider quand entraîner le réseau (ici tous les 4 pas)
    self.t_step = 0

  def step(self, state, action, reward, next_state, done):
    # Ajouter la transition (state, action, reward, next_state, done) dans le Replay Memory
    self.memory.push((state, action, reward, next_state, done))

    # Incrémenter le compteur de pas modulo 4
    self.t_step = (self.t_step + 1) % 4

    # Si c'est le 4ème pas, effectuer un apprentissage
    if self.t_step == 0:
      # S'assurer qu'on a assez d'expériences pour former un batch
      if len(self.memory.memory) > minibatch_size:
        # Échantillonner un batch aléatoire de transitions
        experiences = self.memory.sample(100)

        # Apprentissage à partir de ce batch
        self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    # Convertir l'état en tenseur PyTorch et ajouter une dimension batch
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    # Mettre le réseau en mode évaluation (pas de dropout, pas de batchnorm)
    self.local_qnetwork.eval()
    with torch.no_grad():
      # Obtenir les Q-values prédites pour chaque action
      action_values = self.local_qnetwork(state)
    # Repasser en mode entraînement
    self.local_qnetwork.train()

    # Politique epsilon-greedy : exploration ou exploitation
    if random.random() > epsilon:
      # Exploitation : choisir l'action avec la Q-value maximale
      return np.argmax(action_values.cpu().data.numpy())
    else:
      # Exploration : choisir une action aléatoire
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    # Déballer le batch d'expériences
    states, next_states, actions, rewards, dones = experiences

    # Calcul des Q-values cibles en utilisant le réseau cible
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

    # Formule DQN : Q_target = r + gamma * max(Q_next) * (1 - done)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)

    # Q-values prédites par le réseau principal pour les actions choisies
    q_expected = self.local_qnetwork(states).gather(1, actions)

    # Calcul de la loss (Mean Squared Error)
    loss = F.mse_loss(q_expected, q_targets)

    # Rétropropagation
    self.optimizer.zero_grad()  # Remise à zéro des gradients
    loss.backward()              # Calcul des gradients
    self.optimizer.step()        # Mise à jour des poids du réseau

    # Mise à jour douce (soft update) du réseau cible
    self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

  def soft_update(self, local_model, target_model, interpolation_parameter):
    # Met à jour le réseau cible pour qu'il suive lentement le réseau principal
    # θ_target = τ * θ_local + (1 - τ) * θ_target
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)


# Créer l'agent comme avant
agent = Agent(state_size, number_actions)

# Charger les poids sauvegardés
checkpoint_path = 'Save/Final_checkpoint.pth'
agent.local_qnetwork.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Mettre le réseau en mode évaluation
agent.local_qnetwork.eval()

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
