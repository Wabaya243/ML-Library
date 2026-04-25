# importation des libraries
import numpy as np
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dueling Double DQN avec Noisy Nets

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

# Creation de la structure de la Neural Network
class NoisyNetwork(nn.Module):
    def __init__(self, input_size, nb_action):
        super(NoisyNetwork, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = NoisyLinear(input_size, 64)
        self.fc2 = NoisyLinear(64, 48)
        self.fc3 = NoisyLinear(48, 28)
        self.fc4 = NoisyLinear(28, 16)

        #couche head Dueling
        self.value_fc = NoisyLinear(16, 1)
        self.advantage_fc = NoisyLinear(16, nb_action)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # Valeur de l'état
        value = self.value_fc(x)
        # Avantage par action
        advantage = self.advantage_fc(x)
        # Combinaison Dueling
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
        self.value_fc.reset_noise()
        self.advantage_fc.reset_noise()

# Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)).to(device), samples)

# Deep Q Learning avec target network
class Dqn():
    def __init__(self, input_size, nb_action, gamma, tau=0.01):
        self.gamma = gamma
        self.tau = tau  # paramètre de soft update
        self.reward_window = []

        # Réseau Online
        self.online_model = NoisyNetwork(input_size, nb_action).to(device)
        # Réseau Target
        self.target_model = NoisyNetwork(input_size, nb_action).to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())  # initialisation
        self.target_model.eval()

        self.memory = ReplayMemory(150000)
        self.optimizer = optim.Adam(self.online_model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0).to(device)
        self.last_action = 0
        self.last_reward = 0
        self.input_size = input_size

    # selectionne une action epsilon-greedy
    def select_action(self, state, epsilon=0.0):
        with torch.no_grad():
            self.online_model.reset_noise()
            q_values = self.online_model(Variable(state).to(device))
        return q_values.max(1)[1].item()

    # apprentissage avec target network
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Q(s,a) prédit par le réseau ONLINE
        outputs = self.online_model(batch_state)\
            .gather(1, batch_action.unsqueeze(1))\
            .squeeze(1)

        # ========= DOUBLE DQN =========

        # 1️ Sélection de l’action avec le réseau ONLINE
        next_actions = self.online_model(batch_next_state)\
            .detach()\
            .max(1)[1]

        # 2️ Évaluation de cette action avec le réseau TARGET
        next_q_values = self.target_model(batch_next_state)\
            .detach()\
            .gather(1, next_actions.unsqueeze(1))\
            .squeeze(1)

        # 3️ Cible TD Double DQN
        target = batch_reward + self.gamma * next_q_values

        # 4️ Loss Huber
        td_loss = F.smooth_l1_loss(outputs, target)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        # 5️ Soft update du réseau cible
        self.soft_update()


    def soft_update(self):
        for target_param, online_param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    # mise à jour de l'état et apprentissage
    def update(self, reward, new_signal, epsilon=0.1):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0).to(device)
        self.memory.push((
            self.last_state, 
            new_state, 
            torch.LongTensor([int(self.last_action)]).to(device), 
            torch.Tensor([self.last_reward]).to(device)
        ))
        action = self.select_action(new_state, epsilon)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_reward = reward
        self.last_state = new_state

        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    # calcul du score moyen
    def score(self):
        return sum(self.reward_window)/ (len(self.reward_window) + 1.)

    # sauvegarde du modèle
    def save(self):
        torch.save(self.online_model.state_dict(), f'Save/Two_last_brain_input_size_{self.input_size}.pth')
        print("Model sauvegardé !")

    # chargement du modèle
    def load(self):
        filename = f"Save/Two_last_brain_input_size_{self.input_size}.pth"
        if os.path.isfile(filename):
            self.online_model.load_state_dict(torch.load(filename))
            self.target_model.load_state_dict(self.online_model.state_dict())
            self.online_model.to(device)
            self.target_model.to(device)
            print("Model chargé !")
