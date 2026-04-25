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
        self.fc4 = nn.Linear(28, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


#implementation de l'experience replay 
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    #ajout d'une experience a la memoire et verifie qu'on depasse pas la capacity
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    #echantillonage d'une experience aleatoire de la memoire
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # on echantillonne batch_size experiences aleatoires de la memoire
        return map(lambda x: Variable(torch.cat(x, 0)).to(device), samples) # on melange les differentes listes d'elements de l'event (state, action, reward, next_state)

#implementation de la Deep Q Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] # une fenetre sur le 100 derniere recompense que l'ia a vu 
        self.model = Network(input_size, nb_action).to(device)
        self.memory = ReplayMemory(150000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0).to(device)
        self.last_action = 0
        self.last_reward = 0
        self.input_size = input_size

    # selectionne une action en fonction de l'etat actuel en utilisant une politique epsilon-greedy
    def select_action(self, state):
        with torch.no_grad():   # GPU friendly + pas de calcul de gradient
            probs = F.softmax(self.model(Variable(state).to(device))* 90, dim=1) # T=7
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]

    # effectue une etape d'entrainement sur le modele en utilisant une experience de la memoire
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # on prend la q-value correspondant a l'action effectuee
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # on prend la plus grande q-value pour chaque etat suivant (detach() pour ne pas calculer le gradient)
        
        target = self.gamma * next_outputs + batch_reward # on calcule la cible comme la recompense actuelle plus le gamma times la plus grande q-value pour chaque etat suivant
        td_loss = F.smooth_l1_loss(outputs, target) # on calcule la perte de perte de l'erreur entre la q-value predite et la cible
        
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        

    # met a jour l'etat actuel de l'ia avec l'etat suivant et l'action effectuee
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0).to(device)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]).to(device), torch.Tensor([self.last_reward]).to(device))) # on ajoute l'evenement (state, new_state, action, reward) a la memoire
        action = self.select_action(new_state) # on selectionne une action en fonction de l'etat actuel

        if len(self.memory.memory) > 100: # si on a plus de 100 experiences dans la memoire
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) # on echantillonne 100 experiences aleatoires de la memoire
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_reward = reward
        self.last_state = new_state
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    # calcule la moyenne des recompenses de la fenetre de recompenses
    def score(self):
        return sum(self.reward_window)/ (len(self.reward_window) + 1.)

    # fonction pour sauvegarder le poids
    def save(self):
        torch.save(self.model.state_dict(), 'Save/last_brain_input_size_'+str(self.input_size)+'.pth')
        print("model sauvegardé !")
        print(next(self.model.parameters()).device)

    def load(self):
        filename = f"Save/last_brain_input_size_{self.input_size}.pth"
        if os.path.isfile(filename):
            print("chargement du model !!")
            self.model.load_state_dict(torch.load('Save/last_brain_input_size_'+str(self.input_size)+'.pth'))
            self.model.to(device)
            print('model chargé !')
            