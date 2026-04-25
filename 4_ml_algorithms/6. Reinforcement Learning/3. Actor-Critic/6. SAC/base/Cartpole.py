import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F

LEARNING_RATE = 0.002      # Taux d’apprentissage de l’optimiseur
GAMMA = 0.99               # Facteur de discount des récompenses futures
EPSILON = 0.2              # Paramètre de clipping PPO
UPDATE_INTERVAL = 20       # Nombre d’épisodes avant mise à jour du réseau
PPO_EPOCHS = 4             # Nombre d’epochs d’optimisation PPO
BATCH_SIZE = 5             # Taille des mini-batchs

# Création de l’environnement CartPole
env = gym.make("CartPole-v1")
print(f"Max episode steps: {env.spec.max_episode_steps}")

# Modèle Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Couches partagées entre l’acteur et le critique
        # Elles extraient des caractéristiques communes de l’état
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        
        # Réseau Actor (politique)
        # Il prédit la probabilité de chaque action
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)  # Distribution de probabilité sur les actions
        )
        
        # Réseau Critic (fonction de valeur)
        # Il estime la valeur de l’état V(s)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self):
        # Non utilisé ici car on sépare act() et evaluate()
        raise NotImplementedError
    
    def act(self, state):
        """
        Sélectionne une action à partir de l’état courant
        """
        # Conversion de l’état en tenseur PyTorch
        state = torch.FloatTensor(state)
        
        # Extraction des caractéristiques communes
        shared_features = self.shared(state)
        
        # Calcul des probabilités d’actions et de la valeur de l’état
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        
        # Création de la distribution catégorielle
        dist = Categorical(action_probs)
        
        # Échantillonnage d’une action
        action = dist.sample()
        
        # Retourne :
        # - l’action (int)
        # - le log-prob de l’action (pour PPO)
        # - la valeur estimée de l’état
        return action.item(), dist.log_prob(action), value
    
    def evaluate(self, states, actions):
        """
        Évalue un batch d’états et d’actions
        Utilisé lors de la mise à jour PPO
        """
        # Conversion en tenseur si nécessaire
        if isinstance(states, list):
            states = torch.FloatTensor(states)
        
        # Passage dans les couches partagées
        shared_features = self.shared(states)
        
        # Calcul des probabilités d’actions et des valeurs
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        
        # Création de la distribution
        dist = Categorical(action_probs)
        
        # Conversion des actions en tenseur si nécessaire
        if isinstance(actions, list):
            actions = torch.tensor(actions)
            
        # Calcul du log-prob des actions prises
        log_probs = dist.log_prob(actions)
        
        # Calcul de l’entropie (favorise l’exploration)
        entropy = dist.entropy()
        
        # Retourne :
        # - log_probs : pour le ratio PPO
        # - value : estimation V(s)
        # - entropy : régularisation exploration
        return log_probs, value.squeeze(), entropy

    
    
# Mémoire SAC (Replay Buffer)
# Le ReplayBuffer stocke les transitions (s, a, r, s', done)
# afin de casser la corrélation temporelle lors de l’apprentissage
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size   # Taille maximale du buffer
        self.ptr = 0               # Pointeur circulaire
        self.size = 0              # Nombre réel d’éléments stockés
        
        # Buffers pour stocker les données des transitions
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32)
    
    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une transition dans le replay buffer
        """
        # Stockage de l’état courant
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        
        # Stockage de l’action effectuée
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        
        # Stockage de la récompense reçue
        self.rewards[self.ptr] = torch.tensor([reward], dtype=torch.float32)
        
        # Stockage de l’état suivant
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32)
        
        # Indique si l’épisode est terminé
        self.dones[self.ptr] = torch.tensor([float(done)], dtype=torch.float32)
        
        # Mise à jour du pointeur (buffer circulaire)
        self.ptr = (self.ptr + 1) % self.max_size
        
        # Mise à jour de la taille effective du buffer
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un batch de transitions
        """
        # Tirage aléatoire d’indices valides
        ind = torch.randint(0, self.size, size=(batch_size,))
        
        # Retourne un batch (s, a, r, s', done)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )
    
    def __len__(self):
        """
        Retourne le nombre d’éléments stockés dans le buffer
        """
        return self.size

    
# Initialisation des dimensions

# Dimension de l’espace d’état
state_dim = env.observation_space.shape[0]

# Dimension de l’espace d’action
# Ici actions discrètes (CartPole : gauche / droite)
action_dim = env.action_space.n


# Réseaux SAC (Soft Actor-Critic)

# Contrairement à PPO, SAC utilise :
# - un réseau de politique (Actor)
# - deux réseaux Q (Critics) pour réduire le biais de surestimation

# Réseau de politique (Actor)
class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACPolicy, self).__init__()
        
        # Réseau fully-connected pour prédire
        # la moyenne et l'écart-type de la politique gaussienne
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Sorties : mean et log_std
        )
        
        self.action_dim = action_dim
        
        # Bornes pour stabiliser l’écart-type
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        """
        Calcule une action continue et son log-probability
        """
        # Passage dans le réseau actor
        x = self.actor(state)
        
        # Séparation de la moyenne et du log de l'écart-type
        mean, log_std = x.chunk(2, dim=-1)
        
        # Contraindre log_std pour éviter instabilités numériques
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # Création d'une distribution gaussienne
        normal = torch.distributions.Normal(mean, std)
        
        # Échantillonnage avec le reparameterization trick
        # (permet la rétropropagation)
        x_t = normal.rsample()
        
        # Application de tanh pour borner les actions entre [-1, 1]
        action = torch.tanh(x_t)
        
        # Calcul du log-prob corrigé (changement de variable tanh)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state):
        """
        Retourne une action pour l’interaction avec l’environnement
        """
        # Ajout d’une dimension batch
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Calcul de l’action
        action, _ = self.forward(state)
        
        # Retour au format numpy
        return action.detach().cpu().numpy()[0]
    

# Réseau Critic (Q-networks)
class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACCritic, self).__init__()
        
        # Architecture du premier Q-network (Q1)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Architecture du second Q-network (Q2)
        # Permet de réduire le biais de surestimation
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """
        Calcule Q1(s,a) et Q2(s,a)
        """
        # Concaténation état + action
        x = torch.cat([state, action], dim=1)
        
        # Calcul des deux valeurs Q
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """
        Utilisé lors de la mise à jour de la politique
        """
        x = torch.cat([state, action], dim=1)
        return self.q1(x)

# Hyperparamètres SAC
ALPHA = 0.2                  # Coefficient d'entropie (exploration)
TARGET_UPDATE_INTERVAL = 1   # Fréquence de mise à jour du réseau cible
GAMMA = 0.99                 # Facteur de discount
TAU = 0.005                  # Coefficient de soft update

# Initialisation des réseaux
policy = SACPolicy(state_dim, action_dim)
critic = SACCritic(state_dim, action_dim)
critic_target = SACCritic(state_dim, action_dim)

# Synchronisation initiale des poids du réseau cible
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

# Optimiseurs
policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

print(policy_optimizer)

# Initialisation du Replay Buffer
buffer = ReplayBuffer(
    max_size=100000,
    state_dim=state_dim,
    action_dim=action_dim
)


# Conversion entre actions discrètes et continues

# CartPole utilise des actions discrètes (0 ou 1)
# SAC est conçu pour des actions continues
# → on fait donc une conversion entre les deux espaces

def convert_discrete_to_continuous(discrete_action, action_dim):
    """
    Convertit une action discrète en action continue dans [-1, 1]
    Exemple :
        0 → -1
        1 →  1
    """
    return np.array([2.0 * discrete_action / (action_dim - 1) - 1.0])

def convert_continuous_to_discrete(continuous_action, action_dim):
    """
    Convertit une action continue dans [-1, 1] vers une action discrète
    """
    # S'assurer que l'action reste dans les bornes
    continuous_action = np.clip(continuous_action, -1.0, 1.0)
    
    # Remappage vers l'espace discret [0, action_dim - 1]
    discrete_action = int(
        (continuous_action[0] + 1.0) / 2.0 * (action_dim - 1) + 0.5
    )
    
    return discrete_action

# Boucle d'entraînement SAC
def train_sac(env, policy, critic, critic_target,
              policy_optimizer, critic_optimizer, buffer,
              max_episodes=1000, max_steps=500, batch_size=64):
    
    # Stocke la récompense totale de chaque épisode
    episode_rewards = []
    
    # Boucle sur les épisodes
    for episode in range(max_episodes):
        
        # Réinitialisation de l'environnement
        state, _ = env.reset()
        episode_reward = 0
        
        # Boucle sur les étapes de l’épisode
        for step in range(max_steps):
            
            # Sélection de l'action

            # L’agent choisit une action continue
            continuous_action = policy.get_action(state)
            
            # Conversion vers une action discrète pour CartPole
            discrete_action = convert_continuous_to_discrete(
                continuous_action, action_dim
            )
            
            # Interaction avec l’environnement
            next_state, reward, done, truncated, _ = env.step(discrete_action)
            
            # Fin d’épisode (terminé ou tronqué)
            done = done or truncated
            
            # Stockage de la transition dans le replay buffer
            buffer.push(state, continuous_action, reward, next_state, done)
            
            # Mise à jour de l’état courant
            state = next_state
            episode_reward += reward
            
            # Entraînement (si assez de données)
            if len(buffer) > batch_size:
                
                # Échantillonnage d’un batch aléatoire
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Mise à jour du Critic
                with torch.no_grad():
                    # Action suivante selon la politique actuelle
                    next_actions, next_log_probs = policy(next_states)
                    
                    # Valeurs Q cibles
                    next_q1, next_q2 = critic_target(next_states, next_actions)
                    
                    # Minimum des deux Q + terme d'entropie
                    next_q = torch.min(next_q1, next_q2) - ALPHA * next_log_probs
                    
                    # Cible TD
                    target_q = rewards + (1 - dones) * GAMMA * next_q
                
                # Valeurs Q courantes
                current_q1, current_q2 = critic(states, actions)
                
                # Perte du critic (double Q-learning)
                critic_loss = (
                    F.mse_loss(current_q1, target_q) +
                    F.mse_loss(current_q2, target_q)
                )
                
                # Optimisation du critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Mise à jour de l’Actor

                # Mise à jour retardée (selon TARGET_UPDATE_INTERVAL)
                if step % TARGET_UPDATE_INTERVAL == 0:
                    
                    # Actions générées par la politique
                    actions_new, log_probs = policy(states)
                    
                    # Q-value associée
                    q1_new = critic.q1_forward(states, actions_new)
                    
                    # Perte de la politique (max Q + entropie)
                    policy_loss = (ALPHA * log_probs - q1_new).mean()
                    
                    # Optimisation de la politique
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()
                    
                    # Soft update du Critic cible
                    for target_param, param in zip(
                        critic_target.parameters(),
                        critic.parameters()
                    ):
                        target_param.data.copy_(
                            TAU * param.data +
                            (1 - TAU) * target_param.data
                        )
            
            # Fin de l’épisode
            if done:
                break
        
        # Sauvegarde de la récompense totale
        episode_rewards.append(episode_reward)
        
        # Affichage des performances
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
                
    return episode_rewards



import imageio

# Visualisation de l’historique des récompenses
def plot_rewards(rewards_history, avg_window=10):
    """
    Affiche l’évolution des récompenses au cours de l’entraînement
    avec une moyenne glissante
    """
    # Calcul de la moyenne mobile des récompenses
    avg_rewards = []
    for i in range(len(rewards_history)):
        start_idx = max(0, i - avg_window + 1)
        avg_rewards.append(np.mean(rewards_history[start_idx:i+1]))
    
    # Création du graphique
    plt.figure(figsize=(10, 5))
    plt.plot(
        rewards_history,
        label="Récompense totale par épisode",
        alpha=0.5
    )
    plt.plot(
        avg_rewards,
        label=f"Moyenne des récompenses (sur {avg_window} épisodes)",
        linewidth=2,
        color='red'
    )
    
    # Annotation du graphique
    plt.xlabel("Épisode")
    plt.ylabel("Récompense")
    plt.title("Progression de l’entraînement (SAC)")
    plt.legend()
    
    # Sauvegarde et affichage
    plt.savefig("sac_training_progress.png")
    plt.show()


# Visualisation d’épisodes (compatible SAC et API Gym)
def visualize_episode(policy, env_name="CartPole-v1", num_episodes=10):
    """
    Joue plusieurs épisodes avec la politique entraînée
    et retourne les frames du meilleur épisode
    """
    # Création de l’environnement avec rendu RGB
    # new_step_api=True pour gérer les différences d’API Gym
    eval_env = gym.make(
        env_name,
        render_mode="rgb_array",
        new_step_api=True
    )
    
    best_frames = []     # Frames du meilleur épisode
    best_score = -np.inf
    
    # Boucle sur plusieurs épisodes d’évaluation
    for _ in range(num_episodes):
        
        # Réinitialisation de l’environnement
        reset_result = eval_env.reset()
        
        # Gestion des différentes versions de l’API Gym
        if isinstance(reset_result, tuple):
            # Nouvelle API : (state, info)
            if len(reset_result) == 2:
                state, _ = reset_result
            else:
                state = reset_result[0]
        else:
            # Ancienne API : state seul
            state = reset_result
        
        frames = []
        total_reward = 0
        done = False
        
        # Boucle de l’épisode
        while not done:
            # Sauvegarde de l’image courante
            frames.append(eval_env.render())
            
            # Sélection de l’action continue par la politique SAC
            continuous_action = policy.get_action(state)
            
            # Conversion vers une action discrète (CartPole)
            action_dim = eval_env.action_space.n
            discrete_action = convert_continuous_to_discrete(
                continuous_action,
                action_dim
            )
            
            # Exécution de l’action
            step_result = eval_env.step(discrete_action)
            
            # Gestion des différentes signatures de step()
            if len(step_result) == 5:
                # Nouvelle API : (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # Ancienne API : (next_state, reward, done, info)
                next_state, reward, done, _ = step_result
            else:
                raise ValueError(
                    f"Format inattendu pour step(): {step_result}"
                )
            
            total_reward += reward
            state = next_state
        
        # Conservation du meilleur épisode
        if total_reward > best_score:
            best_score = total_reward
            best_frames = frames
    
    eval_env.close()
    return best_frames


# Conversion action continue → action discrète
def convert_continuous_to_discrete(continuous_action, action_dim):
    """
    Convertit une action continue [-1, 1]
    vers une action discrète de CartPole
    """
    continuous_action = np.clip(continuous_action, -1.0, 1.0)
    discrete_action = int(
        (continuous_action[0] + 1.0) / 2.0 * (action_dim - 1) + 0.5
    )
    return discrete_action


# Évaluation finale et sauvegarde GIF
def evaluate_and_save_gif(policy, rewards):
    """
    Affiche les performances et sauvegarde un GIF
    du comportement de l’agent
    """
    # Affichage de l’historique des récompenses
    plot_rewards(rewards)
    
    # Visualisation du comportement de l’agent
    try:
        frames = visualize_episode(policy)
        
        # Suppression des dimensions inutiles si nécessaire
        frames = [np.squeeze(frame) for frame in frames]
        
        # Sauvegarde du GIF (durée en ms par frame)
        imageio.mimsave(
            "cartpole_sac_policy.gif",
            frames,
            duration=33.33,
            loop=0
        )
        
        print("GIF sauvegardé : cartpole_sac_policy.gif")
    
    except Exception as e:
        print(f"Impossible de sauvegarder la visualisation : {e}")
        
        # Affichage détaillé de l’erreur
        import traceback
        traceback.print_exc()


rewards = train_sac(env, policy, critic, critic_target, policy_optimizer, critic_optimizer, buffer)
        
# Sauvegarder l'Actor (policy)
torch.save(policy.state_dict(), "Save/sac_actor.pth")

# Sauvegarder le Critic (double Q)
torch.save(critic.state_dict(), "Save/sac_critic.pth")

policy.load_state_dict(torch.load("Save/sac_actor.pth"))
policy.eval()  # mettre en mode évaluation


# Final evaluation
eval_env = gym.make("CartPole-v1", render_mode='human')
eval_rewards = []

for episode in range(10):
    state, _ = eval_env.reset()
    total_reward = 0
    done = False
    env.render()
    
    while not done:
        continuous_action = policy.get_action(state)
        discrete_action = convert_continuous_to_discrete(continuous_action, action_dim)
        
        next_state, reward, terminated, truncated, _ = eval_env.step(discrete_action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    eval_rewards.append(total_reward)

eval_env.close()
print(f"Evaluation average reward: {np.mean(eval_rewards):.2f}")