import gymnasium as gym
import numpy as np
import torch
from torch.nn import Module, Linear, LeakyReLU
import matplotlib.pyplot as plt
from tqdm import tqdm

# Given a time-series, une moyenne mobile exponentielle
def exp_moving_avg(arr, beta=0.9):
    '''
    arr : série temporelle (ex : rewards par épisode)
    beta : facteur de lissage (proche de 1 = plus lisse)
    '''

    # Nombre d'éléments dans la série temporelle
    n = arr.shape[0]
    
    # Création d'un tableau rempli de zéros pour stocker la moyenne mobile
    mov_avg = np.zeros(n)
    
    # Initialisation du premier élément de la moyenne mobile
    # (1 - beta) donne le poids accordé à la première valeur
    mov_avg[0] = (1 - beta) * arr[0]
    
    # Boucle sur tous les éléments à partir du deuxième
    for i in range(1, n):
        # Calcul récursif de la moyenne mobile exponentielle :
        # - beta * mov_avg[i-1] : mémoire du passé (ancienne moyenne)
        # - (1 - beta) * arr[i] : contribution de la valeur actuelle
        mov_avg[i] = beta * mov_avg[i-1] + (1 - beta) * arr[i]
    
    # Retourne le tableau contenant la moyenne mobile exponentielle
    return mov_avg


class PolicyNet(Module):
    def __init__(self, state_dim, n_actions, n_hidden=128):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.fc1 = Linear(state_dim, n_hidden)
        self.fc2 = Linear(n_hidden, n_hidden)
        self.fc3 = Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = LeakyReLU()(x)
        x = self.fc2(x)
        x = LeakyReLU()(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=0)
        return x

    def select_action(self, curr_state):
        action_probs = self.forward(torch.tensor(curr_state, dtype=torch.float32))
        action = np.random.choice(self.n_actions, p=action_probs.detach().numpy())
        return action

# la loss fonction
def loss_fn(probs, r):
    return -1 * torch.sum(r * torch.log(probs))

# les hyperparamètres
n_episodes = 700
max_episode_len = 500
n_hidden = 128
lr = 0.001
gamma = 0.99
n_actions = 2
state_dim = 4

# Tableau pour stocker la longueur de chaque épisode
history_episode_len = np.zeros(n_episodes)

# Création du réseau de policy (πθ)
# state_dim : dimension de l'état
# n_actions : nombre d'actions possibles
# n_hidden : taille de la couche cachée
agent = PolicyNet(state_dim, n_actions, n_hidden)

# Optimiseur Adam pour mettre à jour les paramètres du réseau
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

# Boucle principale sur les épisodes
for episode in range(n_episodes):
    
    # Création de l'environnement CartPole pour cet épisode
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Réinitialisation de l'environnement → état initial
    state, _ = env.reset()
    
    # Liste pour stocker les transitions (state, action, reward)
    transitions = []

    # Boucle sur les étapes de l'épisode
    for t in tqdm(range(max_episode_len), desc=f"Epoch : {episode}/{n_episodes}"):
        env.render()
        
        # Sélection d'une action à partir de la policy πθ(a|s)
        action = agent.select_action(state)
        
        # Exécution de l'action dans l'environnement
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        
        # Sauvegarde de la transition
        #  ici "t+1" joue le rôle de reward (longueur d'épisode)
        transitions.append((state, action, t + 1))
        
        # Vérifie si l'épisode est terminé
        finished = terminated or truncated
        if finished:
            break
        
    # Longueur réelle de l'épisode
    episode_len = len(transitions)
    
    # Stocke la longueur de l'épisode pour analyse/plot
    history_episode_len[episode] = episode_len
    
    # Création du batch des états à partir des transitions
    state_batch = torch.Tensor(np.array([s for (s, a, r) in transitions]))
    
    # Création du batch des actions exécutées
    action_batch = torch.Tensor([a for (s, a, r) in transitions])
    
    # Création du batch des rewards, inversé pour le calcul du retour
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
    
    # Passage des états dans le réseau pour obtenir πθ(a|s)
    pred_batch = agent(state_batch)
    
    # Extraction de la probabilité des actions réellement prises
    prob_batch = pred_batch.gather(
        dim=1,
        index=action_batch.long().view(-1, 1)
    ).squeeze()

    # Calcul des facteurs d'actualisation γ^t
    disc_return = torch.pow(gamma, torch.arange(episode_len).float()) * reward_batch
    
    # Normalisation du retour pour stabiliser l'apprentissage
    disc_return /= disc_return.max()

    # Calcul de la loss REINFORCE
    loss = loss_fn(prob_batch, disc_return)
    
    # Réinitialisation des gradients
    optimizer.zero_grad()
    
    # Calcul des gradients par backpropagation
    loss.backward()
    
    # Mise à jour des paramètres du réseau
    optimizer.step()
     
    # Affichage des informations de l'épisode
    print(f"Episode: {episode + 1},   Episode Length: {episode_len}")
        
    # Fermeture propre de l'environnement
    env.close()
    


# plot exponentially moving average of episodes lengths while training
plt.figure(figsize=(14,7))
plt.plot(exp_moving_avg(history_episode_len, 0.8))
plt.xlabel("Episode", fontsize=25)
plt.ylabel("Length", fontsize=25)
plt.grid(True)


# demonstration des actions de l'agent dans le simulator

n_episodes = 10

for episode in range(n_episodes):
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset()
    episode_ln = 0

    for t in range(max_episode_len):
        env.render()
        episode_ln += 1
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        if finished:
            break

    print(f"Episode: {episode + 1},   Episode Length: {episode_ln}")
    env.close()