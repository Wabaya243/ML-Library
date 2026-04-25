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

# Creation de la structure de la Neural Network
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 28)
        self.fc4 = nn.Linear(28, 16)

        #Dueling Head
        self.value_fc = nn.Linear(16, 1)
        self.advantage_fc = nn.Linear(16, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        #valeur de l'etat
        value = self.value_fc(x)

        #avantage de l'etat
        advantage = self.advantage_fc(x)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

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
        td_loss = F.smooth_l1_loss(outputs, target)

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
