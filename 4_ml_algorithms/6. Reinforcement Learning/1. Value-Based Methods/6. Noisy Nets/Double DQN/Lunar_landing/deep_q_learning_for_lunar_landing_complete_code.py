# Deep Q-Learning for Lunar Landing
# Deep Q-Learning pour l’atterrissage lunaire (Lunar Lander)

# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !apt-get install -y swig
# !pip install gymnasium[box2d]

# Importing the libraries
# Importation des librairies essentielles : PyTorch, NumPy, Gym, etc.

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

# Part 1 - Building the AI
# Partie 1 – Construction de l'IA

# Creating the architecture of the Neural Network
# Création de l’architecture du réseau neuronal : 3 couches fully-connected

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        
        # Nombre d'entrées (features de l'état) et sorties (neurones de la couche)
        self.in_features = in_features
        self.out_features = out_features

        # Poids déterministes (mu) et leur écart-type (sigma) pour le bruit
        # Ces paramètres sont appris pendant l'entraînement
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers pour stocker le bruit factorisé (epsilon)
        # Pas de gradient ici, juste pour le calcul forward
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        # Valeur initiale du sigma pour le bruit
        self.sigma_init = sigma_init
        
        # Initialisation des poids et du bruit
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialise les poids mu et sigma ainsi que les biais."""
        # La plage de mu est inversement proportionnelle à la racine du nombre d'entrées
        mu_range = 1 / np.sqrt(self.in_features)

        # Poids mu uniformément distribués
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # Poids sigma initialisés à sigma_init (constante)
        self.weight_sigma.data.fill_(self.sigma_init)

        # Biais mu uniformément distribués
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # Biais sigma initialisés à sigma_init (constante)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        """Génère un nouveau bruit factorisé pour cette couche."""
        # Vecteurs aléatoires pour l'entrée et la sortie
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        
        # Bruit factorisé pour les poids (outer product)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # Bruit direct pour les biais
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        Calcul de la sortie de la couche Noisy Linear.
        Si on est en entraînement, on ajoute le bruit aux poids et biais.
        Sinon, on utilise uniquement les poids mu déterministes.
        """
        if self.training:
            # Ajout du bruit multiplicatif aux poids et biais
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Mode évaluation : pas de bruit
            weight = self.weight_mu
            bias = self.bias_mu
        
        # Produit linéaire classique : x*W^T + b
        return F.linear(x, weight, bias)


class NoisyNetwork(nn.Module):

  def __init__(self, state_size, action_size, seed = 42):
    super(NoisyNetwork, self).__init__()
    # Seed pour reproductibilité
    self.seed = torch.manual_seed(seed)
    # Couche d'entrée -> 64 neurones
    self.fc1 = NoisyLinear(state_size, 128)
    # Couche cachée -> 64 neurones
    self.fc2 = NoisyLinear(128, 64)
    # Couche cachée -> 32 neurones
    self.fc3 = NoisyLinear(64, 32)
    # Couche de sortie : une Q-value par action possible
    self.fc4 = NoisyLinear(32, action_size)

  def forward(self, state):
    # Forward pass : passage dans le réseau
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.relu(x)
    return self.fc4(x)   # Retourne les Q-values

  def reset_noise(self):
    self.fc1.reset_noise()
    self.fc2.reset_noise()
    self.fc3.reset_noise()
    self.fc4.reset_noise()

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
    self.local_qnetwork = NoisyNetwork(state_size, action_size).to(self.device)

    # Réseau Q cible (target network) : utilisé pour le calcul des Q-cibles et mis à jour lentement
    self.target_qnetwork = NoisyNetwork(state_size, action_size).to(self.device)

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

  def act(self, state):
    # Convertir l'état en tenseur PyTorch et ajouter une dimension batch
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    # Mettre le réseau en mode évaluation (pas de dropout, pas de batchnorm)
    self.local_qnetwork.eval()
    with torch.no_grad():
      self.local_qnetwork.reset_noise()  # Re-générer le bruit
      # Obtenir les Q-values prédites pour chaque action
      action_values = self.local_qnetwork(state)
    # Repasser en mode entraînement
    self.local_qnetwork.train()

    return np.argmax(action_values.cpu().data.numpy()) 

  def learn(self, experiences, discount_factor):
    states, next_states, actions, rewards, dones = experiences

    # ========= DOUBLE DQN =========

    # 1️ Choix de l’action avec le réseau ONLINE
    next_actions = self.local_qnetwork(next_states).detach().argmax(1).unsqueeze(1)

    # 2️ Évaluation de cette action avec le réseau TARGET
    next_q_targets = self.target_qnetwork(next_states).detach().gather(1, next_actions)

    # 3️ Cible TD Double DQN
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)

    # 4️ Q(s,a) prédit par le réseau ONLINE
    q_expected = self.local_qnetwork(states).gather(1, actions)

    # 5️ Loss
    loss = F.mse_loss(q_expected, q_targets)

    # 6️ Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # 7️ Soft update du réseau cible
    self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)


  def soft_update(self, local_model, target_model, interpolation_parameter):
    # Met à jour le réseau cible pour qu'il suive lentement le réseau principal
    # θ_target = τ * θ_local + (1 - τ) * θ_target
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)


# Initializing the DQN agent
# Initialisation de l’agent avec les tailles d’entrée/sortie de l’environnement

agent = Agent(state_size, number_actions)

# Training the DQN agent
# Entraînement de l'agent sur un certain nombre d'épisodes

number_episodes = 2000
maximum_number_timesteps_per_episode = 1000

scores_on_100_episodes = deque(maxlen = 100)  # Moyenne glissante

import imageio

def save_episode_video(agent, env_name, episode_num):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)

    env.close()
    filename = f"Video/video_episode_{episode_num}.mp4"
    imageio.mimsave(filename, frames, fps=30)
    print(f"Vidéo enregistrée : {filename}")


for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0

  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state)
    next_state, reward, done, _, _ = env.step(action)

    # Mise à jour de l’agent
    agent.step(state, action, reward, next_state, done)

    state = next_state
    score += reward

    if done:
      break

  scores_on_100_episodes.append(score)
  
  if episode == 1:
    save_episode_video(agent, 'LunarLander-v3', episode)

  if episode % 10 == 0:
    print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")

  if episode % 100 == 0:
    print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    
  if episode % 250 == 0:  
    #tous les 500 episode on sauvegarde une video
    save_episode_video(agent, 'LunarLander-v3', episode)

  if episode % 200 == 0:
    # Sauvegarde des poids du réseau principal
    torch.save(agent.local_qnetwork.state_dict(), f'Save/checkpoint_{episode}.pth')
    print(f"Checkpoint sauvegardé pour l'épisode {episode}")


  # Condition pour arrêter si l'environnement est résolu
  if np.mean(scores_on_100_episodes) >= 200.0:
    print('\nEnvironment resolu dans {:d} episodes!\tScore Moyenne: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'Save/Final_checkpoint.pth')
    save_episode_video(agent, 'LunarLander-v3', episode)
    break

  

# Part 3 - Visualizing the results
# Partie 3 – Visualisation du comportement de l'agent sous forme de vidéo

import glob
import io
import base64
import imageio
from IPython.display import HTML, display

# Fonction pour enregistrer la vidéo de l’agent entraîné
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)

    env.close()
    imageio.mimsave('Video/Final_video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v3')

# Fonction pour afficher la vidéo dans un notebook
def show_video():
    mp4list = glob.glob('Video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()

