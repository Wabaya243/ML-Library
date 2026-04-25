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
from collections import deque
import cv2

import cv2

def preprocess(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    return state


# Partie 1 – Construction de l'IA

# Création de l’architecture du réseau neuronal : 3 couches fully-connected

class Network(nn.Module):

  def __init__(self, action_size, seed = 42):
    super(Network, self).__init__()
    # Seed pour reproductibilité
    self.seed = torch.manual_seed(seed)

    self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4) # (84 - 8) / 4 + 1 = 20
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # (20 - 4) / 2 + 1 = 9
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # (9 - 3) / 1 + 1 = 7

    self.fc1 = nn.Linear(64 * 7 * 7, 512)
    # Couche cachée -> 64 neurones
    self.fc2 = nn.Linear(512, 256)
    # Couche de sortie : une Q-value par action possible
    self.fc3 = nn.Linear(256, action_size)

  def forward(self, state):
    # Forward pass : passage dans le réseau
    x = state / 255.0

    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)

    # flatten
    x = x.view(x.size(0), -1)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x   # Retourne les Q-values


import gymnasium as gym
env = gym.make('CarRacing-v3', render_mode='human')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
ACTIONS = [
    np.array([0.0, 1.0, 0.0]),   # accélérer
    np.array([-1.0, 1.0, 0.0]),  # tourner gauche + accélérer
    np.array([1.0, 1.0, 0.0]),   # tourner droite + accélérer
    np.array([0.0, 0.0, 0.8]),   # frein
]
number_actions = len(ACTIONS)



print('State shape: ', state_shape)   # Format de l'état (8 valeurs)
print('State size: ', state_size)     # Nombre de caractéristiques

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
    states = torch.from_numpy(np.stack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.stack([e[3] for e in experiences if e is not None])).float().to(self.device) 
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

    return states, next_states, actions, rewards, dones

# Implementing the DQN class
# Implémentation de l’agent DQN

class Agent():

  def __init__(self, action_size):
    # Définir le device : GPU si disponible, sinon CPU
    self.device = device
    
    # Nombre d'actions possibles dans l'environnement
    self.action_size = action_size

    # Réseau Q principal (online network) : prédit les Q-values pour chaque action
    self.local_qnetwork = Network(action_size).to(self.device)

    # Réseau Q cible (target network) : utilisé pour le calcul des Q-cibles et mis à jour lentement
    self.target_qnetwork = Network(action_size).to(self.device)

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
    action_index = np.argmax(action_values.cpu().data.numpy()) if random.random() > epsilon else random.choice(range(len(ACTIONS)))
    return action_index, ACTIONS[action_index]
   
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

agent = Agent(number_actions)

# Training the DQN agent
# Entraînement de l'agent sur un certain nombre d'épisodes

number_episodes = 2000
maximum_number_timesteps_per_episode = env.spec.max_episode_steps
print(maximum_number_timesteps_per_episode)

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
    state = preprocess(state)
    state = np.expand_dims(state, axis=0)
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        action_index, action = agent.act(state)
        state, reward, done, _, _ = env.step(action)

    env.close()
    filename = f"Video/video_episode_{episode_num}.mp4"
    imageio.mimsave(filename, frames, fps=30)
    print(f"Vidéo enregistrée : {filename}")


for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  state = preprocess(state)
  state = np.expand_dims(state, axis=0)
  score = 0


  for t in range(maximum_number_timesteps_per_episode):
    action_index, action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    next_state = preprocess(next_state)
    next_state = np.expand_dims(next_state, axis=0)

    env.render()

    # Mise à jour de l’agent
    agent.step(state, action_index, reward, next_state, done)


    state = next_state
    score += reward

    if done:
      break

  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

 
  print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")

  if episode % 100 == 0:
    print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    
  if episode % 250 == 0:  
    #tous les 500 episode on sauvegarde une video
    save_episode_video(agent, 'CarRacing-v3', episode)

  if episode % 200 == 0:
    # Sauvegarde des poids du réseau principal
    torch.save(agent.local_qnetwork.state_dict(), f'Save/checkpoint_{episode}.pth')
    print(f"Checkpoint sauvegardé pour l'épisode {episode}")


  # Condition pour arrêter si l'environnement est résolu
  if np.mean(scores_on_100_episodes) >= 500.0:
    print('\nEnvironment resolu dans {:d} episodes!\tScore Moyenne: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'Save/Final_checkpoint.pth')
    save_episode_video(agent, 'CarRacing-v3', episode)
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
    state = preprocess(state)
    state = np.expand_dims(state, axis=0)
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        action_index, action = agent.act(state)
        state, reward, done, _, _ = env.step(action)

    env.close()
    imageio.mimsave('Video/Final_video.mp4', frames, fps=30)

show_video_of_model(agent, 'CarRacing-v3')

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

