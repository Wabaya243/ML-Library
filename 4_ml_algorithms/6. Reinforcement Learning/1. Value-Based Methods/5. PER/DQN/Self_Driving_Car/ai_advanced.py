# importation des libraries
from seaborn import widgets
import numpy as np
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creation de la structure de la Neural Network
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 28)
        self.fc4 = nn.Linear(28, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values

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

        # Réseau Online
        self.online_model = Network(input_size, nb_action).to(device)
        # Réseau Target
        self.target_model = Network(input_size, nb_action).to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())  # initialisation
        self.target_model.eval()

        self.memory = ReplayMemory(150000)
        self.optimizer = optim.Adam(self.online_model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0).to(device)
        self.last_action = 0
        self.last_reward = 0
        self.input_size = input_size

    # selectionne une action epsilon-greedy
    def select_action(self, state, epsilon=0.1):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.online_model(Variable(state).to(device))
            return q_values.max(1)[1].item()
        else:
            return random.randrange(self.online_model.nb_action)

    # apprentissage avec target network
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.online_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Utilisation du target network pour calculer les q-values cibles
        next_outputs = self.target_model(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.gamma * next_outputs

        td_erros = target - outputs
        loss = (td_erros.pow(2) * widgets).mean() #PER pondéré
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update du target network
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
