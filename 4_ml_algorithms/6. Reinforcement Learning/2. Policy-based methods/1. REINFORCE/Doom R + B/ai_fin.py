import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import copy


# importation des packages pour OpenAI and Doom
import gymnasium as gym 

from gymnasium.wrappers import MaxAndSkipObservation, RecordVideo
# importation des autres fichier python
import image_preprocessing, env

from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)

        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k, obs_shape[0], obs_shape[1]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)



STACK_SIZE = 4
Image_Size = (1, 160, 160)



# on envoie les donné dans le GPU si dispo
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyCNN(nn.Module):
    def __init__(self, action_size):
        super(PolicyCNN, self).__init__()
        
        # la couche de convo
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        
        # la fonction d'activation
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=self.count_neurons(), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=action_size)

    def count_neurons(self):
       x = torch.zeros(1, 4, 160, 160)

       x = self.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
       x = self.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
       x = self.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))

       return x.view(1, -1).size(1)

    def forward(self, x):

        #la couche de convo
        x = self.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = self.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = self.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        
        # le Flattening
        x = x.view(x.size(0), -1)
        
        # La couche fully connected
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

class ValueCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self._count_neurons(), 512)
        self.fc2 = nn.Linear(512, 1)

    def _count_neurons(self):
        x = torch.zeros(1, 4, 160, 160)
        x = self.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.relu(F.max_pool2d(self.conv3(x), 2))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # V(s)


# Creation de l'agent

class REINFORCEAgent():

  def __init__(self, policy, value_network, seed = 42, learning_rate = 1e-4, gamma=0.99):
    # Définir le device : GPU si disponible, sinon CPU
    self.device = Device
    self.gamma = gamma

    self.policy = policy.to(self.device)
    self.value_network = value_network.to(self.device)

    # Optimiseur Adam pour mettre à jour les poids du réseau principal
    self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)
    self.value_optimizer = optim.Adam(self.value_network.parameters(), lr = learning_rate)

    self.saved_log_probs = []
    self.rewards = []
    self.states = []

  def act(self, state):
    # Convertir l'état en tenseur PyTorch et ajouter une dimension batch
    state = torch.from_numpy(state).float()
    
    # SUPPRESSION de la dimension grayscale
    state = state.squeeze(1)  # -> (4, 160, 160)

    # Ajout du batch
    state = state.unsqueeze(0).to(self.device)  # -> (1, 4, 160, 160)
    
    # Obtenir les probabilités des actions à partir de la politique
    probs = self.policy(state)
    # Créer une distribution catégorique à partir des probabilités
    dist = torch.distributions.Categorical(probs)

    # Choisir une action en fonction de la distribution catégorique
    action = dist.sample()
    # Enregistrer le log de la probabilité de l'action choisie
    self.saved_log_probs.append(dist.log_prob(action))    
    # Ajouter l'état à la liste des états visités
    self.states.append(state)

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

    # Concaténer tous les états pour calculer les Q-values
    states = torch.cat(self.states)
    # Calculer les Q-values pour chaque état
    values = self.value_network(states).squeeze()

    # Normalisation des retours (centrage et réduction de la variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    # le 1e-8 est pour éviter la division par zéro

    # Calcul des avantages (différence entre retours et Q-values)
    advantages = returns - values.detach()
    
    #calcul de la Actor loss (REINFORCE + Baseline)
    policy_loss = 0
    for log_prob, adv in zip(self.saved_log_probs, advantages):
        policy_loss += -log_prob * adv

    # Calcul de la Critic loss (MSE)
    value_loss = F.mse_loss(values, returns)


    # Rétropropagation
    self.value_optimizer.zero_grad()
    value_loss.backward()
    self.value_optimizer.step()

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

    # Reset du Buffers
    self.saved_log_probs.clear()
    self.rewards.clear()
    self.states.clear()
    
# Iniialisation de l'environement pour l'agent

r'''
print(os.listdir(scenarios_path))
['basic.cfg', 'basic.wad', 'cig.cfg', 'cig.wad', 'cig_with_unknown.wad', 'deadly_corridor.cfg', 
 'deadly_corridor.wad', 'deathmatch.cfg', 'deathmatch.wad', 'defend_the_center.cfg', 'defend_the_center.wad', 
 'defend_the_line.cfg', 'defend_the_line.wad', 'health_gathering.cfg', 'health_gathering.wad', 'health_gathering_supreme.cfg', 
 'health_gathering_supreme.wad', 'learning.cfg', 'multi.cfg', 'multi_deathmatch.wad', 'multi_duel.cfg', 'multi_duel.wad',
 'my_way_home.cfg', 'my_way_home.wad', 'oblige.cfg', 'predict_position.cfg', 'predict_position.wad', 'README.md', 'rocket_basic.cfg', 
 'rocket_basic.wad', 'simpler_basic.cfg', 'simpler_basic.wad', 'take_cover.cfg', 'take_cover.wad']
'''

# on init l'environement
doom_env = env.VizdoomEnv("deadly_corridor.cfg", difficulty=5)

print(f"Nombre de combinaisons d'actions : {len(doom_env.actions_list)}")
print("Exemples de combinaisons :")

for i, action_vec in enumerate(doom_env.actions_list[:10]):  # afficher seulement les 10 premières
    print(f"{i} : {action_vec}")

# on met le frameskip
doom_env = MaxAndSkipObservation(doom_env, skip=5)

# on fait le preprocessing de l'image
doom_env = image_preprocessing.PreprocessImage(doom_env, width=160, height=160, grayscale=True)

doom_env = FrameStack(doom_env, k=4)

# on enregistre la video
doom_env = RecordVideo(
    doom_env, 
    video_folder="Video_fine",
    episode_trigger= lambda ep: ep % 50 == 0
    )

number_actions = doom_env.action_space.n

# Création de l'IA

# Création de l'agent IA avec un cerveau (CNN) 
policy = PolicyCNN(number_actions)
value_network = ValueCNN()

# Charger les poids sauvegardés
checkpoint = torch.load("Save/checkpoint_100.pth")

policy.load_state_dict(checkpoint["policy"])
value_network.load_state_dict(checkpoint["value"])


agent = REINFORCEAgent(policy, value_network)


# Moyenne mobile des récompenses

# Classe permettant de calculer la moyenne des récompenses
# sur les 100 dernières étapes
class MA:
    def __init__(self, size):
        self.list_of_rewards = []  # historique des récompenses
        self.size = size           # taille de la fenêtre

    def add(self, rewards):
        # Ajoute une ou plusieurs récompenses
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)

        # On garde seulement les 'size' dernières valeurs
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        # Retourne la moyenne des récompenses stockées
        return np.mean(self.list_of_rewards)

ma = MA(100)


nb_epoch = 2000

# Boucle d'entraînement

# On entraîne le modèle sur plusieurs epochs
for epoch in range(nb_epoch):
    state, _ = doom_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        state, reward, done, _, _ = doom_env.step(action)
        agent.store_reward(reward)
        total_reward += reward

    agent.learn()
    ma.add(total_reward)

    print(f"Epoch {epoch} | Avg Reward: {ma.average()}")

    if epoch % 50 == 0:
        torch.save({
        "policy": policy.state_dict(),
        "value": value_network.state_dict()
       }, f"Save_fine/checkpoint_{epoch}.pth")


    if ma.average() > 1000:
        print("Bravo, environnement maîtrisé.")
        torch.save({
        "policy": policy.state_dict(),
        "value": value_network.state_dict()
       }, "Save_fine/final_checkpoint.pth")

        break