
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque


# Partie 1 – Construction de l'IA

# Création de l’architecture du réseau neuronal : 3 couches fully-connected

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


# Partie 2 – Entraînement de l'IA

# Configuration de l’environnement LunarLander-v3

env = gym.make('LunarLander-v3')
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n

print('State shape: ', state_shape)   # Format de l'état (8 valeurs)
print('State size: ', state_size)     # Nombre de caractéristiques
print('Number of actions: ', number_actions) # 4 actions possibles


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Implémentation de l’agent DQN

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

# Training the DQN agent
# Entraînement de l'agent sur un certain nombre d'épisodes

number_episodes = 2000
maximum_number_timesteps_per_episode = 1000

scores_on_100_episodes = deque(maxlen = 100)  # Moyenne glissante

import imageio

def save_episode_video(agent, env_name, episode_num, max_frames=300):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    frames = []

    for _ in range(max_frames):
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
        if done:
            break

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
    agent.store_reward(reward)
    state = next_state
    score += reward

    if done:
      break

  agent.learn()
  scores_on_100_episodes.append(score)

  if episode == 1:
    save_episode_video(agent, 'LunarLander-v3', episode)

  
  print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")

  if episode % 100 == 0:
    print('\rEpisode {}\tScore Moyenne: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    
  if episode % 200 == 0:  
    #tous les 500 episode on sauvegarde une video
    save_episode_video(agent, 'LunarLander-v3', episode)

  if episode % 200 == 0:
    # Sauvegarde des poids du réseau principal
    torch.save(agent.policy.state_dict(), f'Save/checkpoint_{episode}.pth')
    print(f"Checkpoint sauvegardé pour l'épisode {episode}")


  # Condition pour arrêter si l'environnement est résolu
  if np.mean(scores_on_100_episodes) >= 200.0:
    print('\nEnvironment resolu dans {:d} episodes!\tScore Moyenne: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.policy.state_dict(), 'Save/Final_checkpoint.pth')
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
        state, reward, done, _, _ = env.step(action.item())

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

