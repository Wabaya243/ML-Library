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

class Network(nn.Module):

  def __init__(self, state_size, action_size, seed = 42):
    super(Network, self).__init__()
    # Seed pour reproductibilité
    self.seed = torch.manual_seed(seed)
    # Couche d'entrée -> 64 neurones
    self.fc1 = nn.Linear(state_size, 256)
    # Couche cachée -> 64 neurones
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 128)
    self.fc4 = nn.Linear(128, 64)
    # Couche cachée -> 32 neurones
    self.fc5 = nn.Linear(64, 32)
    
    # couche Head Dueling
    # Valeur de l'état
    self.value_fc = nn.Linear(64, 1)
    # Avantage par action
    self.advantage_fc = nn.Linear(64, action_size)
    
  def forward(self, state):
    # Forward pass : passage dans le réseau
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.relu(x)
    x = self.fc4(x)
    x = F.relu(x)
    x = self.fc5(x)
    x = F.relu(x)
    
    # Valeur de l'état
    value = self.value_fc(x)
    # Avantage par action
    advantage = self.advantage_fc(x)
    # Combinaison Dueling
    q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
    return q_values   # Retourne les Q-values

# Partie 2 – Entraînement de l'IA

# Setting up the environment
# Configuration de l’environnement LunarLander-v3

import gymnasium as gym
env = gym.make('Taxi-v3')
state_size = env.observation_space.n  # ici c’est un entier (discret)
number_actions = env.action_space.n   # 6 actions

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


# Initializing the DQN agent
# Initialisation de l’agent avec les tailles d’entrée/sortie de l’environnement

agent = Agent(state_size, number_actions)

# Training the DQN agent
# Entraînement de l'agent sur un certain nombre d'épisodes

number_episodes = 4500
maximum_number_timesteps_per_episode = 1000

# Paramètres epsilon-greedy
epsilon_starting_value  = 1.0     # 100% exploration au début
epsilon_ending_value  = 0.01      # 1% exploration minimum
epsilon_decay_value  = 0.995      # Décroissance par épisode
epsilon = epsilon_starting_value

scores_on_100_episodes = deque(maxlen = 100)  # Moyenne glissante

import imageio

def save_episode_video(agent, env_name, episode_num):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    timestep = 0
    frames = []

    while not done:
        frame = env.render()
        # Enregistrer 1 frame sur 2
        if timestep % 10 == 0:
            frames.append(frame)
        timestep += 1
        state_one_hot = np.zeros(state_size)
        state_one_hot[state] = 1
        action = agent.act(state_one_hot)
        state, reward, done, _, _ = env.step(action)


    env.close()
    filename = f"Video/video_episode_{episode_num}.mp4"
    imageio.mimsave(filename, frames, fps=15)
    print(f"Vidéo enregistrée : {filename}")


for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0

  for t in range(maximum_number_timesteps_per_episode):
    # Convertir l'état discret en vecteur one-hot
    state_one_hot = np.zeros(state_size)
    state_one_hot[state] = 1
    
    action = agent.act(state_one_hot, epsilon)
    next_state, reward, done, _, _ = env.step(action)

    # Convertir next_state pour le replay memory
    next_state_one_hot = np.zeros(state_size)
    next_state_one_hot[next_state] = 1

    agent.step(state_one_hot, action, reward, next_state_one_hot, done)

    state = next_state  # pour la boucle, garder l'état original discret
    
    score += reward

    if done:
      break

  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

  #if episode == 1:
   # save_episode_video(agent, 'Taxi-v3', episode)

  print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")

  if episode % 100 == 0:
    print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    # Sauvegarde des poids du réseau principal
  if episode % 300 == 0:
    torch.save(agent.local_qnetwork.state_dict(), f'Save/checkpoint_{episode}.pth')
    print(f"Checkpoint sauvegardé pour l'épisode {episode}")
    
  if episode % 600 == 0:  
    #tous les 600 episode on sauvegarde une video
    save_episode_video(agent, 'Taxi-v3', episode)
    
  # Condition pour considérer l'environnement comme résolu
  # Si l'agent atteint un score moyen supérieur à 9.0, on peut considérer qu'il apprend correctement
  if np.mean(scores_on_100_episodes) >= 9.0:
        print('\nEnvironment resolu dans {:d} episodes!\tScore Moyenne: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
        torch.save(agent.local_qnetwork.state_dict(), 'Save/Final_checkpoint.pth')
    #    save_episode_video(agent, 'Taxi-v3', episode)
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
    timestep = 0

    while not done:
        frame = env.render()
        # Enregistrer 1 frame sur 2
        if timestep % 2 == 0:
            frames.append(frame)
        timestep += 1
        state_one_hot = np.zeros(state_size)
        state_one_hot[state] = 1
        action = agent.act(state_one_hot)
        state, reward, done, _, _ = env.step(action)

    env.close()
    imageio.mimsave('Video/Final_video.mp4', frames, fps=30)

show_video_of_model(agent, 'Taxi-v3')

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

