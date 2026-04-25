import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import copy, random
from collections import deque



# importation des packages pour OpenAI and Doom
import gymnasium as gym 


from gymnasium.wrappers import MaxAndSkipObservation, RecordVideo
# importation des autres fichier python
import experience_replay, image_preprocessing, env, PER



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
Image_Size = (4, 160, 160)



# on envoie les donné dans le GPU si dispo
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingCNN(nn.Module):
    def __init__(self, action_size):
        super(DuelingCNN, self).__init__()
        
        # la couche de convo
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        
        # la fonction d'activation
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=self.count_neurons(Image_Size), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        
        # Head Dueling
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        self.advantage_fc = nn.Linear(in_features=256, out_features=action_size)

    def count_neurons(self, image_dim):
       x = Variable(torch.rand(1, *image_dim))

       x = self.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
       x = self.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
       x = self.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
       
       return x.view(x.size(0), -1).size(1)

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
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A)
        Q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return Q_values



# Creation de l'agent

class AI:
    def __init__(self, brain, n_actions,
                 epsilon_start=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.995
                 ):
        
        self.brain = brain.to(Device)
        self.n_actions = n_actions

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


    def __call__(self, inputs):
        
        if isinstance(inputs, tuple):
            inputs = inputs[0]  # déballer si accident tuple (obs, info)
                
            # Cas Doom : (4, 1, H, W) → (4, H, W)
        if inputs.ndim == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        
        if inputs.ndim == 3:  # ajouter batch dimension
            inputs = np.expand_dims(inputs, axis=0)  # devient (1,C,H,W)
            
        # on transforme une liste des tableau a 2D a un gros tableau en 3D
        inputs = np.array(inputs, dtype=np.float32)
        # on transforme un array de numpy en tenseur de pytorch 
        inputs = torch.from_numpy(inputs).to(Device)
        
        # E-greedy
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.brain(inputs)
                action = torch.argmax(q_values, dim=1).item()
            
           
        # Copier sur CPU avant de convertir en numpy
        return np.array([[action]])
        
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
doom_env = MaxAndSkipObservation(doom_env, skip=4)

# on fait le preprocessing de l'image
doom_env = image_preprocessing.PreprocessImage(doom_env, width=160, height=160, grayscale=True)

doom_env = FrameStack(doom_env, k=4)

# on enregistre la video
doom_env = RecordVideo(
    doom_env, 
    video_folder="Video",
    episode_trigger= lambda ep: ep % 50 == 0
    )

number_actions = doom_env.action_space.n

# Création de l'IA

# Réseau de neurones convolutionnel qui va approximer la fonction Q
cnn = DuelingCNN(number_actions).to(Device)
cnn.load_state_dict(torch.load("Save/Test/checkpoint_1.pth"))

target_cnn = DuelingCNN(number_actions).to(Device)
target_cnn.load_state_dict(cnn.state_dict())



# Création de l'agent IA avec un cerveau (CNN) et un corps (Softmax)
agent = AI(
    brain=cnn,
    n_actions=number_actions,
    epsilon_start=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.996
)


# Experience Replay

# N-Step Progress : l'agent observe des séquences de 15 étapes consécutives
n_steps = experience_replay.NStepProgress(
    env=doom_env,
    ai=agent,
    n_step=20
)

# Mémoire de replay qui stocke jusqu'à 100 000 expériences
memory = PER.PERMemory(
    n_steps=n_steps,
    capacity=100000,
    alpha=0.6,
    beta_start=0.4,
    beta_frames=500000
)


# Eligibility Trace

def eligibilityTrace(batch):
    """
    Version optimisée de l'eligibility trace (forward view λ).
    Calcul des cibles Q pour un batch entier en évitant les triples boucles.
    """
    gamma = 0.99  # facteur de discount
    lam = 0.9     # paramètre lambda
    inputs = []   # états pour le training
    targets = []  # valeurs Q cibles pour le training

    # On parcourt chaque série de n-step dans le batch
    for series in tqdm(batch, desc="Elig. Trace Fast"):

        # Récupération des états, actions et récompenses
        states  = [step.state for step in series]
        actions = [step.action for step in series]
        rewards = [0.0 if r is None or np.isnan(r) else r for r in [step.reward for step in series]]
        done    = series[-1].done  # savoir si l'épisode est terminé
        T = len(series)

        # Transformation des états en tenseurs PyTorch
        states_tensor = torch.from_numpy(np.array(states, dtype=np.float32)).to(Device)
        
        states_tensor = states_tensor.squeeze(2)  
        # (T, 4, 1, 160, 160) en (T, 4, 160, 160)

        # Calcul des prédictions Q pour le réseau online et target
        with torch.no_grad():
            q_current = cnn(states_tensor)   # Q(s,a) prédit par le réseau online
            q_target  = target_cnn(states_tensor)  # Q(s,a) cible par le target network
            
            # Double DQN : action choisie par le online network
            next_actions = torch.argmax(q_current, dim=1)


        # Initialisation du retour G (si l'épisode est fini, G=0, sinon bootstrap avec le max du dernier état)
        if done:
            G = 0.0
        else:
            G = q_target[-1, next_actions[-1]].item()


        # On parcourt les steps en **reversed order** (de la fin vers le début)
        for t in reversed(range(T)):
            # Calcul du lambda-return : récompense immédiate + combinaison de Q cible et G précédent
            # (forward view simplifié mais équivalent)
            G = rewards[t] + gamma * ((1 - lam) * q_target[t, next_actions[t]].item() + lam * G)

            # On clone q_current pour ne pas modifier directement le réseau
            target = q_current[t].clone()

            # Mise à jour uniquement de la Q-value correspondant à l'action effectuée
            target[actions[t]] = G

            # Stockage pour le training
            inputs.append(states[t].squeeze(1))
            targets.append(target)

    # Conversion en tenseurs PyTorch
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)



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

# hyperparams
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0005)
nb_epoch = 500

# Boucle d'entraînement

# On entraîne le modèle sur plusieurs epochs
for epoch in range(nb_epoch):

    # L'agent joue 1000 étapes dans l'environnement
    # et stocke les expériences dans la mémoire
    memory.run_steps(700)
    
    # 500 × 1000 = 500 000 steps joués au total
    
    if epoch < 10:
        MAX_BATCH_PER_EPOCH = 2
    elif epoch < 60:
        MAX_BATCH_PER_EPOCH = 4
    else :
        MAX_BATCH_PER_EPOCH = 6  # ou 8 max
        
    '''
    4 batches
    chaque batch = 128 sequences
    une sequences = 15 steps
    4 x 128 x 15 = 7 680 transition / epoch
    trop d’updates = sur-apprentissage de bruit

    '''
    
    # Entraînement du réseau à partir des expériences mémorisées
    # On prélève des batches de 128 séquences
    for i, (batch, indices, weights) in enumerate(tqdm(memory.sample_batch(128), desc=f"Epoch : {epoch}/{nb_epoch}")):
        
        if i >= MAX_BATCH_PER_EPOCH:
            break

        # Calcul des entrées et des cibles avec Eligibility Trace
        inputs, targets = eligibilityTrace(batch)

        # Conversion en Variables PyTorch (utile pour le calcul du gradient)
        inputs, targets = Variable(inputs).to(Device), Variable(targets).to(Device)

        # Prédictions du réseau (valeurs Q)
        predictions = cnn(inputs)

        # Calcul de l'erreur entre prédictions et cibles
        weights = torch.from_numpy(weights).unsqueeze(1).to(Device)  # (batch_size, 1)
        loss_error = (F.mse_loss(predictions, targets, reduction='none') * weights).mean()


        # Remise à zéro des gradients précédents
        optimizer.zero_grad()

        # Rétropropagation de l'erreur
        loss_error.backward()

        # Mise à jour des poids du réseau
        optimizer.step()
            
        # Mise à jour des priorités
        td_errors = (targets - predictions).detach().cpu().abs().max(axis=1)[0]  # max sur les actions
        memory.update_priorities(indices, td_errors)

    # Récupération des récompenses obtenues sur les dernières étapes
    rewards_steps = n_steps.rewards_steps()
    if len(rewards_steps) == 0:
        rewards_steps = [0]  # éviter nan au début
    ma.add(rewards_steps)


    # Ajout des récompenses à la moyenne mobile
    ma.add(rewards_steps)

    # Calcul de la récompense moyenne
    avg_reward = ma.average()
    

    # Affichage des performances de l'agent
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    
    agent.epsilon = max(
    agent.epsilon_min,
    agent.epsilon * agent.epsilon_decay
    )

    print(f"Epoch {epoch} | epsilon = {agent.epsilon:.3f}")

    if epoch % 15 == 0:
        target_cnn.load_state_dict(cnn.state_dict())

    if epoch % 10 == 0:
        torch.save(cnn.state_dict(), f"Save/model_{epoch}.pth")
        print("model sauvegardé !")
    
    if avg_reward > 1000.0:
        print("Bravo c'est gagné!")
        break
