
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


from collections import deque

class NStepBuffer:
    def __init__(self, n, gamma):
        # n : nombre de pas pour le retour n-step (ex: 3, 5)
        self.n = n
        
        # gamma : facteur d'actualisation (discount factor)
        self.gamma = gamma
        
        # Buffer temporaire qui stocke les transitions successives
        # Chaque élément = (state, action, reward, next_state, done)
        self.buffer = deque()

    def push(self, transition):
        # Ajoute une nouvelle transition à la fin du buffer
        # transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def is_ready(self):
        # Vérifie si on a accumulé au moins n transitions
        # Condition nécessaire pour calculer un retour n-step
        return len(self.buffer) >= self.n

    def pop(self):
        """
        Retourne une transition n-step :
        (state, action, n_step_return, next_state)
        """
        R = 0.0
        for i in range(self.n):
            R += (self.gamma ** i) * self.buffer[i][2]

        state, action, _, _, _ = self.buffer[0]
        _, _, _, next_state, done = self.buffer[self.n - 1]

        self.buffer.popleft()
        return state, action, R, next_state, done


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


class Network(nn.Module):

  def __init__(self, state_size, action_size, seed = 42):
    super(Network, self).__init__()
    # Seed pour reproductibilité
    self.seed = torch.manual_seed(seed)
    # Couche d'entrée -> 64 neurones
    self.fc1 = NoisyLinear(state_size, 256)
    # Couche cachée -> 64 neurones
    self.fc2 = NoisyLinear(256, 128)
    self.fc3 = NoisyLinear(128, 128)
    self.fc4 = NoisyLinear(128, 64)
    # Couche cachée -> 32 neurones
    self.fc5 = NoisyLinear(64, 32)
    # Couche de sortie : une Q-value par action possible
    self.fc6 = NoisyLinear(32, 16)

    self.value_fc = NoisyLinear(16, 1)
    self.advantage_fc = NoisyLinear(16, action_size)

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
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))

    value = self.value_fc(x)
    advantage = self.advantage_fc(x)

    q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
    return q_values

  def reset_noise(self):
        """Réinitialise le bruit factorisé de toutes les couches NoisyLinear."""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
        self.fc5.reset_noise()
        self.fc6.reset_noise()
        self.value_fc.reset_noise()
        self.advantage_fc.reset_noise()

    

import gymnasium as gym
env = gym.make('Taxi-v3')
state_size = env.observation_space.n  # ici c’est un entier (discret)
number_actions = env.action_space.n   # 6 actions

print('State size: ', state_size)     # Nombre de caractéristiques
print('Number of actions: ', number_actions) # 4 actions possibles


# Initialisation des hyperparamètres du DQN

learning_rate = 5e-4               # Taux d’apprentissage
minibatch_size = 100               # Taille d’un batch dans le Replay Memory
discount_factor = 0.99             # Gamma (réduction future)
replay_buffer_size = int(1e5)      # Taille du replay buffer
interpolation_parameter = 1e-3     # Tau pour le soft update du réseau cible


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# PER
class PrioritizedReplayMemory(object):

  def __init__(self, capacity, alpha=0.6):
    # Appareil utilisé (CPU ou GPU)
    self.device = device

    # Capacité maximale de la mémoire (nombre total de transitions stockables)
    self.capacity = capacity        

    # Liste qui stocke les transitions (state, action, reward, next_state, done)
    self.memory = []                

    # Alpha contrôle le degré de priorisation :
    # alpha = 0  → échantillonnage uniforme
    # alpha = 1  → priorisation totale basée sur le TD-error
    self.alpha = alpha

    # Tableau qui stocke la priorité associée à chaque transition
    # Même taille que la capacité maximale
    self.priorities = np.zeros((capacity,), dtype=np.float32 )

    # Position courante pour l'insertion (buffer circulaire)
    self.pos = 0


  def push(self, event):
    # Récupère la priorité maximale actuelle
    # Objectif : donner une forte priorité aux nouvelles transitions
    max_priority = self.priorities.max() if self.memory else 1.0

    # Si la mémoire n'a pas encore atteint sa capacité maximale
    if len(self.memory) < self.capacity:
      # On ajoute simplement la transition
      self.memory.append(event)
    else:
      # Sinon, on écrase la plus ancienne transition (FIFO circulaire)
      self.memory[self.pos] = event
    
    # Associe à la transition ajoutée la priorité maximale
    self.priorities[self.pos] = max_priority

    # Avance la position de manière circulaire
    self.pos = (self.pos + 1) % self.capacity
    

  def sample(self, batch_size, beta=0.4):
    
    # Récupère uniquement les priorités des transitions réellement stockées
    priorities = self.priorities[:len(self.memory)]

    # Applique la loi de priorité : p_i = priority_i ^ alpha
    probs = priorities ** self.alpha

    # Normalise pour obtenir une distribution de probabilité
    probs /= probs.sum()

    # Échantillonne des indices selon la distribution de probabilité
    indices = np.random.choice(len(self.memory), batch_size, p=probs)

    # Récupère les transitions correspondantes
    samples = [self.memory[i] for i in indices]

    # Nombre total de transitions stockées
    total = len(self.memory)

    # Calcul des poids d'importance sampling (correction du biais)
    # w_i = (N * P(i))^(-beta)
    weights = (total * probs[indices]) ** (-beta)

    # Normalisation des poids pour que max(w_i) = 1
    weights /= weights.max()

    # Décomposition des transitions en éléments séparés
    states, actions, rewards, next_states, dones = zip(*samples)

    # Conversion des batchs en tenseurs PyTorch utilisables par le réseau
    return (
            torch.tensor(states).float().to(self.device),        # États
            torch.tensor(next_states).float().to(self.device),  # États suivants
            torch.tensor(actions).long().unsqueeze(1).to(self.device),  # Actions
            torch.tensor(rewards).float().unsqueeze(1).to(self.device), # Rewards
            torch.tensor(dones).float().unsqueeze(1).to(self.device),   # Dones
            torch.tensor(weights).float().unsqueeze(1).to(self.device), # Poids IS
            indices                                                # Indices pour mise à jour
        )


  def update_priorities(self, indices, priorities):
        # Met à jour les priorités des transitions après l'apprentissage
        # En général : priority = |TD-error| + epsilon
        for idx, priority in zip(indices, priorities):
            # Mise à jour de la priorité associée à chaque transition
            self.priorities[idx] = priority



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
    self.memory = PrioritizedReplayMemory(replay_buffer_size)

     # Compteur pour décider quand entraîner le réseau (ici tous les 4 pas)
    self.t_step = 0

    self.beta = 0.4
    
    self.beta_increment = 1e-4
    
    self.n_step = 4

    self.n_step_buffer = NStepBuffer(self.n_step, discount_factor)

  def step(self, state, action, reward, next_state, done):
    # Ajouter la transition (state, action, reward, next_state, done) dans le Replay Memory
    self.n_step_buffer.push((state, action , reward, next_state, done))

    # 2️ Si le buffer contient n transitions → créer une transition n-step
    if self.n_step_buffer.is_ready():
        state_n, action_n, R_n, next_state_n, done_n = self.n_step_buffer.pop()
        self.memory.push((state_n, action_n, R_n, next_state_n, done_n))

    # Incrémenter le compteur de pas modulo 4
    self.t_step = (self.t_step + 1) % 4

    # Si c'est le 4ème pas, effectuer un apprentissage
    if self.t_step == 0:
      # S'assurer qu'on a assez d'expériences pour former un batch
      if len(self.memory.memory) > minibatch_size:
        
        # Échantillonner un batch aléatoire de transitions
        experiences = self.memory.sample(minibatch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Apprentissage à partir de ce batch
        self.learn(experiences, discount_factor)

  def act(self, state):
    # Convertir l'état en tenseur PyTorch et ajouter une dimension batch
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    # Mettre le réseau en mode évaluation (pas de dropout, pas de batchnorm)
    self.local_qnetwork.eval()
    with torch.no_grad():
      self.local_qnetwork.reset_noise()
      # Obtenir les Q-values prédites pour chaque action
      action_values = self.local_qnetwork(state)
    # Repasser en mode entraînement
    self.local_qnetwork.train()

    return np.argmax(action_values.cpu().data.numpy())

  def learn(self, experiences, discount_factor):
    # Déballer le batch d'expériences
    states, next_states, actions, rewards, dones, weight, indices = experiences

    # ========= DOUBLE DQN =========

    # 1️ Action choisie par le réseau ONLINE
    next_actions = self.local_qnetwork(next_states).detach().argmax(1).unsqueeze(1)

    # 2️ Q-value évaluée par le réseau TARGET
    next_q_targets = self.target_qnetwork(next_states).gather(1, next_actions)

    # 3️ Cible Double DQN
    q_targets = rewards + (discount_factor ** self.n_step) * next_q_targets * (1 - dones)

    # 4️ Q(s,a) prédit par le réseau ONLINE
    q_expected = self.local_qnetwork(states).gather(1, actions)


    # Calcul de la loss Pondéré
    td_errors = q_targets - q_expected
    loss = (td_errors ** 2 * weight).mean()

    #mise a jour de la priorités
    new_priority = td_errors.abs().detach().cpu().numpy() + 1e-6
    self.memory.update_priorities(indices=indices, priorities=new_priority)

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
    
    action = agent.act(state_one_hot)
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
    imageio.mimsave('Video/Final_video.mp4', frames, fps=15)

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

