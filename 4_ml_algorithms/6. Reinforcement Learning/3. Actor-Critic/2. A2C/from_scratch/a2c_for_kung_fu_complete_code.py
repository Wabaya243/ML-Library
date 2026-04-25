
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import ale_py
from gymnasium.wrappers import RecordVideo


import numpy as np
import cv2
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class Network(nn.Module):

  def __init__(self, action_size):
    super(Network, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels = 4,  out_channels = 32, kernel_size = (3,3), stride = 2)
    self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
    self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
    self.flatten = torch.nn.Flatten()
    
    self.fc1  = torch.nn.Linear(512, 256)
    self.fc2 = torch.nn.Linear(256, 128)
    self.fc2a = torch.nn.Linear(128, action_size)
    self.fc2s = torch.nn.Linear(128, 1)

  def forward(self, state):
    x = self.conv1(state)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)

    action_values = self.fc2a(x)
    state_value = self.fc2s(x).squeeze(-1)
    return action_values, state_value


# Wrapper pour prétraiter les observations d'un environnement Atari
class PreprocessAtari(ObservationWrapper):
    
    def __init__(self, env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4):
        """
        Initialisation du wrapper.
        - env : l'environnement Gym original
        - height, width : taille de l'image après redimensionnement
        - crop : fonction pour rogner l'image
        - dim_order : 'pytorch' ou 'tensorflow' pour l'ordre des dimensions
        - color : True si on garde les couleurs, False si on met en niveaux de gris
        - n_frames : nombre de frames à empiler (pour capturer le mouvement)
        """
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)   # taille cible de l'image
        self.crop = crop                  # fonction de découpe
        self.dim_order = dim_order        # ordre des dimensions
        self.color = color                # couleur ou gris
        self.frame_stack = n_frames       # nombre de frames empilées
        
        # nombre de canaux : 1 par frame si gris, 3 par frame si couleur
        n_channels = 3 * n_frames if color else n_frames
        
        # définition de la forme de l'observation selon le framework
        obs_shape = {
            'tensorflow': (height, width, n_channels),
            'pytorch': (n_channels, height, width)
        }[dim_order]
        
        # définition de l'espace d'observation normalisé entre 0 et 1
        self.observation_space = Box(0.0, 1.0, obs_shape)
        
        # buffer pour stocker les frames empilées
        self.frames = np.zeros(obs_shape, dtype=np.float32)

    def reset(self):
        """
        Réinitialise l'environnement et le buffer de frames.
        """
        self.frames = np.zeros_like(self.frames)  # reset du buffer
        obs, info = self.env.reset()              # reset de l'environnement original
        self.update_buffer(obs)                    # ajoute la première observation au buffer
        return self.frames, info                  # retourne les frames empilées et info

    def observation(self, img):
        """
        Prétraite une image de l'environnement.
        """
        img = self.crop(img)                       # rogne l'image
        img = cv2.resize(img, self.img_size)      # redimensionne
        
        if not self.color:
            # si on veut du gris et que l'image est couleur
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img.astype('float32') / 255.        # normalisation [0,1]
        
        # décale le buffer pour ajouter la nouvelle frame à la fin
        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)  # 3 canaux à décaler
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)  # 1 canal à décaler
        
        # ajoute la nouvelle frame au buffer
        if self.color:
            self.frames[-3:] = img
        else:
            self.frames[-1] = img
        
        return self.frames  # retourne les frames empilées

    def update_buffer(self, obs):
        """
        Met à jour le buffer avec une nouvelle observation.
        """
        self.frames = self.observation(obs)


# Fonction utilitaire pour créer l'environnement Atari prétraité
def make_env():
    env = gym.make("ALE/KungFuMaster-v5", render_mode='rgb_array')  # environnement original
    
    env = RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda episode_id: episode_id % 100 == 0
    )
    env = PreprocessAtari(
        env,
        height=42,
        width=42,
        crop=lambda img: img,
        dim_order='pytorch',
        color=False,
        n_frames=4
    )
    return env


env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)

# init de params

learning_rate = 1e-4
discount_factor = 0.99
number_environments = 20

# Implementation de la class A2C
import torch
import torch.nn.functional as F
import numpy as np

 # Calcul de la valeur cible pour le Critic
def compute_returns(rewards, dones, last_value, gamma):
    returns = []
    R = last_value
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
    return torch.stack(returns)


class Agent():
    
    def __init__(self, action_size):
        """
        Initialisation de l'agent.
        - action_size : nombre d'actions possibles dans l'environnement
        """
        # Choix du device : GPU si disponible sinon CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        
        # Création du réseau neuronal (Actor-Critic)
        self.network = Network(action_size).to(self.device)
        
        # Optimiseur Adam pour mettre à jour les paramètres du réseau
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state):
        """
        Sélectionne une action à partir de l'état actuel.
        - state : état courant (peut être un seul état ou un batch)
        """
        # Si on a un seul état (3D), on le transforme en batch de taille 1
        if state.ndim == 3:
          state = np.expand_dims(state, axis=0)
        
        # Convertit en tenseur PyTorch et envoie sur le device
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Passage dans le réseau : retourne action_values (logits) et value de l'état
        action_values, _ = self.network(state)
        
        # Calcul de la politique par softmax (probabilités des actions)
        policy = F.softmax(action_values, dim=-1)
        
        # Échantillonnage d'une action selon la politique
        return np.array([np.random.choice(len(p), p=p) for p in policy.detach().cpu().numpy()])

    def step(self, state, action, reward, next_state, done):
        """
        Mise à jour du réseau Actor-Critic à partir d'une transition.
        - state : état(s) actuel(s)
        - action : action(s) choisie(s)
        - reward : récompense(s) obtenue(s)
        - next_state : état(s) suivant(s)
        - done : bool indiquant si l'épisode est terminé
        """
        batch_size = state.shape[0]  # nombre d'exemples dans le batch
        
        # Conversion en tenseurs PyTorch et envoi sur le device
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)
        
        # Passage dans le réseau pour obtenir :
        # - action_values : logits pour les actions (Actor)
        # - state_value : estimation de la valeur de l'état (Critic)
        action_values, state_value = self.network(state)
        with torch.no_grad():
            _, next_state_value = self.network(next_state)
        
        # Calcul de la valeur cible pour le Critic
        returns = compute_returns(reward, done, next_state_value, discount_factor)
        
        # Calcul de l'avantage (Advantage) pour l'Actor
        advantage = returns - state_value
        # Normalisation de l'avantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        
        # Probabilités et log-probabilités pour le calcul de la loss de l'Actor
        probs = F.softmax(action_values, dim=-1)
        logprobs = F.log_softmax(action_values, dim=-1)
        
        # Entropie pour encourager l'exploration
        entropy = -torch.sum(probs * logprobs, axis=-1)
        
        # Extraction des log-probs correspondant aux actions réellement choisies
        batch_idx = np.arange(batch_size)
        logp_actions = logprobs[batch_idx, action]
        
        # Loss de l'Actor (policy gradient avec pénalité d'entropie)
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        
        # Loss du Critic (MSE entre la valeur cible et la valeur prédite)
        critic_loss = F.mse_loss(target_state_value.detach(), state_value)
        
        # Loss totale = Actor + Critic
        total_loss = actor_loss + critic_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# Initializing the A2C agent

agent = Agent(number_actions)

# Evaluating our A3C agent on a certain number of episodes

def evaluate(agent, env, n_episodes=1):
    """
    Évalue un agent sur un environnement donné.
    - agent : agent RL (Actor-Critic ici)
    - env : environnement Gym
    - n_episodes : nombre d'épisodes de test
    """
    episodes_rewards = []  # liste des récompenses totales par épisode

    for _ in range(n_episodes):
        state, _ = env.reset()  # reset de l'environnement
        total_reward = 0       # récompense cumulée de l'épisode

        while True:
            # L'agent choisit une action à partir de l'état courant
            action = agent.act(state)

            # On applique l'action dans l'environnement
            state, reward, done, info, _ = env.step(action[0])

            # On accumule la récompense
            total_reward += reward

            # Fin de l'épisode
            if done:
                break

        # On sauvegarde la récompense totale de l'épisode
        episodes_rewards.append(total_reward)

    return episodes_rewards


# Managing multiple environments simultaneously

class EnvBatch:

  def __init__(self, n_envs=10):
    """
    Crée un batch de n_envs environnements indépendants
    """
    self.envs = [make_env() for _ in range(n_envs)]

  def reset(self):
      """
      Réinitialise tous les environnements
      """
      _states = []
      for env in self.envs:
          _states.append(env.reset()[0])  # on ne garde que l'observation
      return np.array(_states)


  def step(self, actions):
    '''
        Exécute une action par environnement
        actions : tableau d'actions (une par env)
        zip(self.envs, actions) Associe chaque environnement à son action (env0, a0)
        env.step(a) for env, a in zip(...)]  (s0', r0, d0, i0, _)
        zip(*) → transposition   (s0', s1', s2'),   # next_states
    '''
    next_states, rewards, dones, infos, _ = map(
      np.array, 
     zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
    for i in range(len(self.envs)):
      if dones[i]:
        next_states[i] = self.envs[i].reset()[0]
    return next_states, rewards, dones, infos

# Training the A2C agent

import tqdm

env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()

with tqdm.trange(0, 200_001) as progress_bar:
  for i in progress_bar:
    batch_actions = agent.act(batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
    batch_rewards *= 0.01
    agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
    batch_states = batch_next_states
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))