# Proximal Policy Optimization (PPO)

# Check out this link for the complete model explanation: https://spinningup.openai.com/en/latest/algorithms/ppo.html

# Importing the libraries

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# Setting the hyperparameters

hidden_dim = 256                # Taille de la couche fully-connected
lr_actor = 3e-4                 # Learning rate de l'acteur
lr_critic = 1e-3                # Learning rate du critique
gamma = 0.99                    # Facteur de discount
gae_lambda = 0.95               # Lambda pour le GAE (Generalized Advantage Estimation)
ppo_epochs = 10                 # Nombre d'epochs PPO par update
mini_batch_size = 64            # Taille des mini-batchs
ppo_clip = 0.2                  # Clipping PPO (ratio policy)
buffer_size = 2048              # Nombre de transitions stockées avant update
update_timestep = buffer_size  # Fréquence de mise à jour PPO
action_std = 0.5                # Écart-type pour l'exploration continue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Building the Actor-Critic Network

class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()

        # ===== Couches convolutionnelles communes =====
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Couche fully connected
        self.fc1 = nn.Linear(64 * 8 * 8, hidden_dim)

        # ===== Acteur =====
        self.fc_actor = nn.Linear(hidden_dim, num_actions)

        # Log de l'écart-type (politique gaussienne continue)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # ===== Critique =====
        self.fc_critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Passage dans le CNN
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Aplatissement
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # ===== Acteur =====
        action_mean = self.fc_actor(x)

        # Variance de la loi gaussienne
        action_var = torch.exp(self.log_std.expand_as(action_mean))
        cov_mat = torch.diag_embed(action_var)

        # ===== Critique =====
        value = self.fc_critic(x)

        return action_mean, cov_mat, value


# Implementing the Memory Buffer

class Memory:
    """
    Buffer temporaire utilisé par PPO pour stocker
    les trajectoires avant la mise à jour du réseau.
    """

    def __init__(self):
        self.actions = []        # Actions prises
        self.states = []         # États observés
        self.logprobs = []       # Log-probabilités des actions
        self.rewards = []        # Récompenses reçues
        self.is_terminals = []   # Indique si l'état est terminal (done)

    def clear_memory(self):
        """ Vide complètement le buffer après un update PPO """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Building the PPO Agent

class PPO:
    def __init__(self, num_actions):
        # Réseau Actor-Critic principal (politique courante)
        self.policy = ActorCritic(num_actions).to(device)

        # Optimiseur Adam pour mettre à jour les paramètres du réseau
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)

        # Ancienne politique (π_old), utilisée pour le calcul du ratio PPO
        self.policy_old = ActorCritic(num_actions).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Fonction de perte pour la valeur d’état (critic)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # Pas de calcul de gradient lors du choix de l’action
        with torch.no_grad():
            # Récupération des paramètres de la distribution d’actions
            action_mean, action_var, _ = self.policy_old(state)

            # Distribution gaussienne multivariée pour les actions continues
            dist = MultivariateNormal(action_mean, action_var)

            # Échantillonnage d’une action
            action = dist.sample()

            # Log-probabilité de l’action (utilisée par PPO)
            action_logprob = dist.log_prob(action)

        # Retour de l’action et de sa log-probabilité
        return action.squeeze(0).detach().cpu().numpy(), action_logprob.detach()

    def update(self, memory):
        #  Calcul des récompenses cumulées (Monte Carlo) 
        rewards = []
        discounted_reward = 0

        # Parcours inverse pour calculer G_t
        for reward, is_terminal in zip(
                reversed(memory.rewards),
                reversed(memory.is_terminals)):

            # Si l’épisode est terminé, on remet le cumul à zéro
            if is_terminal:
                discounted_reward = 0

            # Discount factor γ
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Conversion en tenseur
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Normalisation des récompenses (stabilité de l’apprentissage)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        #  Conversion de la mémoire en tenseurs 
        old_states = torch.stack(memory.states).detach().to(device)
        old_actions = torch.stack(memory.actions).detach().to(device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(device)


        #  Optimisation PPO sur plusieurs epochs
        for _ in range(ppo_epochs):

            # Évaluation de la politique courante
            action_means, action_vars, state_values = self.policy(old_states)
            
            state_values = state_values.squeeze(-1)

            # Distribution d’actions
            dists = MultivariateNormal(action_means, action_vars)

            # Log-probabilités des actions précédentes
            logprobs = dists.log_prob(old_actions)

            # Entropie (favorise l’exploration)
            dist_entropy = -logprobs.mean()

            # Ratio PPO : πθ(a|s) / πθ_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Avantage (Advantage Function)
            advantages = rewards - state_values.detach()

            # Surrogate loss (objectif PPO)
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios,
                1 - ppo_clip,
                1 + ppo_clip
            ) * advantages

            # Perte totale PPO (Actor + Critic + Entropie)
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # Descente de gradient
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Mise à jour de la politique ancienne
        self.policy_old.load_state_dict(self.policy.state_dict())


# Prétraitement des états

def preprocess_state(state):
    # Conversion en float32 et normalisation des pixels
    state = np.ascontiguousarray(state, dtype=np.float32) / 255.0
    
    # HWC → CHW
    state = np.transpose(state, (2, 0, 1))

    # Conversion en tenseur PyTorch
    state = torch.from_numpy(state).to(device)

    # Ajout de la dimension batch
    return state.unsqueeze(0)


# Initialisation de l’environnement

env = gym.make('CarRacing-v3')

# Mémoire PPO (stocke états, actions, récompenses, etc.)
memory = Memory()

# Création de l’agent PPO
ppo = PPO(env.action_space.shape[0])


# Boucle d’entraînement
state, _ = env.reset()
state = preprocess_state(state)
episode_reward = 0
episode_rewards = []   # Liste des récompenses par épisode

for t in range(1, update_timestep + 1):

    # Sélection de l’action par la politique
    action, action_logprob = ppo.select_action(state)

    # Interaction avec l’environnement
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    next_state = preprocess_state(next_state)
    
    episode_reward += reward


    # Stockage dans la mémoire
    memory.states.append(state.squeeze(0))
    memory.actions.append( torch.tensor(action, dtype=torch.float32, device=device)     )
    memory.logprobs.append(action_logprob)
    memory.rewards.append(reward)
    memory.is_terminals.append(done)

    # Passage à l’état suivant
    state = next_state

    # Réinitialisation si l’épisode est terminé
    if done:
        episode_rewards.append(episode_reward)
        episode_reward = 0
    
        state, _ = env.reset()
        state = preprocess_state(state)

    # Mise à jour PPO toutes les update_timestep étapes
    if t % update_timestep == 0:
        ppo.update(memory)
        memory.clear_memory()

    if len(episode_rewards) >= 10:
        mean_last_10 = np.mean(episode_rewards[-10:])
        print(f"Moyenne des 10 derniers épisodes : {mean_last_10:.2f}")

