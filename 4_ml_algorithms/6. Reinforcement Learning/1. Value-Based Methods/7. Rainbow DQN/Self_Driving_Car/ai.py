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
        """Réinitialise le bruit de toutes les couches Noisy Linear."""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()
        self.fc4.reset_noise()
        self.value_fc.reset_noise()
        self.advantage_fc.reset_noise()


# Experience Replay
class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6):
        """
        Initialisation de la mémoire avec priorités.

        Args:
            capacity (int): taille maximale de la mémoire
            alpha (float): degré de priorité (0 = uniforme, 1 = totale priorité)
        """
        self.capacity = capacity              # Taille maximale de la mémoire
        self.memory = []                      # Liste pour stocker les transitions
        self.alpha = alpha                    # Contrôle l’importance de la priorité
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Tableau des priorités
        self.pos = 0                          # Position actuelle pour l'insertion (FIFO circulaire)

    def push(self, event):
        """
        Ajoute une transition dans la mémoire avec priorité maximale actuelle.

        Args:
            event (tuple): transition (state, next_state, action, reward)
        """
        # Priorité maximale actuelle pour que la nouvelle transition soit échantillonnée rapidement
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            # Ajouter la transition si la mémoire n'est pas encore pleine
            self.memory.append(event)
        else:
            # Remplacer la transition la plus ancienne (FIFO circulaire)
            self.memory[self.pos] = event

        # Mettre à jour la priorité de cette transition
        self.priorities[self.pos] = max_priority

        # Avancer la position dans la mémoire de manière circulaire
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Échantillonne un batch de transitions selon les priorités (PER).

        Args:
            batch_size (int): nombre de transitions à échantillonner
            beta (float): facteur de correction pour réduire le biais de priorité

        Returns:
            batch_state, batch_next_state, batch_action, batch_reward, weights, indices
        """
        # Récupérer les priorités actuelles
        priorities = self.priorities[:len(self.memory)]
        # Calculer la probabilité d'échantillonnage pour chaque transition
        probs = priorities ** self.alpha
        probs /= probs.sum()  # Normaliser pour obtenir une distribution de probabilité

        # Tirer des indices selon les probabilités
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        # Récupérer les transitions correspondantes
        samples = [self.memory[i] for i in indices]

        # Calculer les poids d'importance pour corriger le biais de priorité
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normaliser pour que le poids maximum soit 1
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Séparer les transitions en batchs pour les entrées du réseau
        batch_state, batch_next_state, batch_action, batch_reward = zip(*samples)
        batch_state = torch.cat(batch_state, 0).to(device)
        batch_next_state = torch.cat(batch_next_state, 0).to(device)
        batch_action = torch.cat(batch_action, 0).to(device)
        batch_reward = torch.cat(batch_reward, 0).to(device)

        return batch_state, batch_next_state, batch_action, batch_reward, weights, indices

    def update_priorities(self, indices, priorities):
        """
        Met à jour les priorités des transitions après le calcul du TD-error.

        Args:
            indices (list[int]): indices des transitions dans la mémoire
            priorities (list[float]): nouvelles priorités correspondantes
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# Deep Q Learning avec target network
class Dqn():
    def __init__(self, input_size, nb_action, gamma, tau=0.01):
        self.gamma = gamma
        self.tau = tau  # paramètre de soft update
        self.reward_window = []
        self.n_step = 3  # 3-5 steps
        self.n_step_buffer = NStepBuffer(self.n_step, self.gamma)

        # Réseau Online
        self.online_model = NoisyNetwork(input_size, nb_action).to(device)
        # Réseau Target
        self.target_model = NoisyNetwork(input_size, nb_action).to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())  # initialisation
        self.target_model.eval()

        self.memory = PrioritizedReplayMemory(150000)
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
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action, weights, indices):
        outputs = self.online_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
         # ========= DOUBLE DQN =========

        # 1️ Sélection de l’action avec le réseau ONLINE
        next_actions = self.online_model(batch_next_state).detach().max(1)[1]

        # 2️ Évaluation de cette action avec le réseau TARGET
        next_q_values = self.target_model(batch_next_state).detach().gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # 3️ Cible TD Double DQN
        target = batch_reward + (self.gamma ** self.n_step) * next_q_values

        # 4️ Loss pondéré
        td_loss = F.smooth_l1_loss(outputs, target)
        td_loss = (td_loss * weights).mean()

        #mise a jour des priorités
        td_errors = (outputs - target).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-5)

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        # Soft update du target network
        self.soft_update()

    def soft_update(self):
        for target_param, online_param in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    # mise à jour de l'état et apprentissage
    def update(self, reward, new_signal, epsilon=0.1):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0).to(device)

        self.n_step_buffer.push((self.last_state, self.last_action, reward, new_state, False))

        if self.n_step_buffer.is_ready():
            state, action, R, next_state, done = self.n_step_buffer.pop()
            self.memory.push((
                state, 
                next_state, 
                torch.LongTensor([int(action)]).to(device), 
                torch.Tensor([R]).to(device)
            ))  
        action = self.select_action(new_state, epsilon)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward, weights, indices = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action, weights, indices)  

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
