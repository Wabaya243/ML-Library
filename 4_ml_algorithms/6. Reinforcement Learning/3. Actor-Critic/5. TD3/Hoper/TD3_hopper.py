from logger import Logger
from buffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import gymnasium as gym
import os


class ActorNetwork(nn.Module):
    """
    Réseau d'acteur simple pour un agent SAC/Actor-Critic.
    - Prend en entrée l'état de l'environnement
    - Produit une action continue bornée dans les limites de l'environnement
    """
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        
        # Réseau entièrement connecté (fully-connected)
        # 3 couches : input -> hidden -> hidden -> output
        # Dernière couche Tanh pour borner l'action entre -1 et 1
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), hidden_dim),  # couche d'entrée
            nn.ReLU(),                                               # activation ReLU
            nn.Linear(hidden_dim, hidden_dim),                       # couche cachée
            nn.ReLU(),                                               # activation ReLU
            nn.Linear(hidden_dim, np.prod(action_space.shape)),      # couche de sortie
            nn.Tanh()                                                # borne l'action [-1, 1]
        )
        
        # Buffers pour transformer l'action [-1,1] vers les limites réelles de l'environnement
        # action_scale : échelle pour correspondre à l'amplitude (high - low) / 2
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2, dtype=torch.float32)
        )
        
        # action_bias : décalage pour recentrer l'action (high + low) / 2
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32)
        )
        
    def forward(self, observation):
        """
        Passage avant du réseau
        - observation : état actuel (tensor)
        - retourne : action continue transformée dans les limites de l'environnement
        """
        # Calcul de l'action brute [-1,1] via le réseau
        action = self.network(observation)
        
        # Mise à l'échelle et recentrage selon les limites de l'environnement
        return action * self.action_scale + self.action_bias


# Critic (Q-network) simple
class CriticNetwork(nn.Module):
    """
    Réseau Q pour SAC : prend en entrée l'état et l'action
    et renvoie la valeur Q(s,a) d’un critic unique.
    """
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        
        # Couche fully-connected : concat état + action
        self.network = nn.Sequential(
            nn.Linear(
                np.prod(observation_space.shape) + np.prod(action_space.shape),
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Sortie : Q-value scalaire
        )
        
    def forward(self, observation, action):
        # Concaténation état + action sur la dernière dimension
        return self.network(torch.cat([observation, action], dim=1))


# Ensemble de Critic pour SAC (Q1 et Q2)
class CriticEnsemble(nn.Module):
    """
    Permet de créer plusieurs critics (Q1, Q2) pour réduire le biais
    de surestimation du Q-learning.
    """
    def __init__(self, observation_space, action_space, hidden_dim, num_critics=2):
        super().__init__()
        self.critics = nn.ModuleList([
            CriticNetwork(observation_space, action_space, hidden_dim)
            for _ in range(num_critics)
        ])
        
    def forward(self, obs, action):
        """
        Renvoie une liste de Q-values pour chaque critic
        """
        return [critic(obs, action) for critic in self.critics]



def gaussian_noise(shape: tuple, mu=0.0, sigma=0.1, clip=None):
    """
    Génère un bruit gaussien pour l'exploration ou le bruit ajouté à la politique.
    
    Paramètres :
        shape : tuple → la forme du tenseur de bruit à générer
        mu    : float → moyenne du bruit (par défaut 0.0)
        sigma : float → écart-type du bruit (par défaut 0.1)
        clip  : float | None → limite absolue pour tronquer le bruit autour de mu
    
    Retour :
        Tensor de bruit gaussien avec éventuellement un clipping
    """
    # Génération du bruit gaussien
    noise = mu + torch.randn(*shape) * sigma
    
    # Tronquer le bruit si clip est défini
    if clip is not None:
        noise = torch.clamp(noise, min=mu-clip, max=mu+clip)
    
    return noise


class TD3Config:
    """
    Configuration centralisée pour l'agent TD3.
    Contient tous les hyperparamètres utilisés pour l'entraînement.
    """
    # Informations générales
    env_name: str             = 'Hopper-v5'  # Nom de l'environnement Gym
    agent_name: str           = 'TD3'        # Nom de l'agent
    device: str               = 'cpu'        # Device pour PyTorch ('cpu' ou 'cuda')
    checkpoint: bool          = False        # Sauvegarde périodique des poids
    num_checkpoints: int      = 20           # Nombre de checkpoints / logs
    verbose: bool             = True         # Affiche les logs pendant l'entraînement
    total_steps: int          = 1_000_000    # Nombre total de steps d'entraînement
    target_reward: int | None = 1000         # Récompense cible pour early stopping
    learning_starts: int      = 5000         # Début de l'apprentissage après ce nombre de steps
    
    # Hyperparamètres classiques RL
    gamma: float              = 0.99         # Facteur de discount des récompenses futures
    lr: float                 = 3e-4         # Learning rate pour les optimizers
    hidden_dim: int           = 256          # Dimension des couches cachées des réseaux Actor/Critic
    buffer_capacity: int      = 100_000      # Taille maximale du replay buffer
    batch_size: int           = 64           # Taille des mini-batchs pour l'apprentissage
    num_steps: int            = 3            # n-step return pour le critic
    tau: float                = 0.005        # Coefficient de soft update du critic cible
    grad_norm_clip: float     = 40.0         # Clip global pour les gradients
    
    # Hyperparamètres spécifiques TD3
    noise_explore: float      = 0.1          # Écart-type du bruit gaussien pour l'exploration
    noise_policy: float       = 0.2          # Écart-type du bruit ajouté à la politique lors de la cible
    noise_clip: float         = 0.5          # Limite maximale du bruit ajouté à la politique
    actor_delay: int          = 2            # Nombre de pas critic par pas actor (delayed policy update)

    
    


class TD3:
    def __init__(self):
        """
        Initialisation de l'agent TD3 :
        - Actor / Target Actor
        - Critic / Target Critics
        - Optimizers et ReplayBuffer
        """
        config = TD3Config()
        self.device = config.device
        
        # Création de l'environnement Gym
        self.env = gym.make(config.env_name)
        observation_space, action_space = self.env.observation_space, self.env.action_space
        
        # Actor et Target Actor pour la politique
        self.actor = ActorNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.soft_update(self.actor, self.target_actor, 1.0)  # Copie initiale des poids
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        
        # CriticEnsemble (Q1/Q2) et Target Critics
        self.critics = CriticEnsemble(observation_space, action_space, config.hidden_dim).to(self.device)
        self.target_critics = CriticEnsemble(observation_space, action_space, config.hidden_dim).to(self.device)
        self.soft_update(self.critics, self.target_critics, 1.0)  # Copie initiale des poids
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.lr)
        
        # Replay buffer pour stocker les transitions
        self.buffer = ReplayBuffer(config.buffer_capacity, config.num_steps, config.gamma)
        self.config = config
        
    def checkpoint(self, steps):
        """
        Sauvegarde les poids de l'acteur sur disque
        """
        if not os.path.exists('models'):
            os.makedirs('models')
        checkpoint_path = f"models/{self.config.agent_name}_{self.config.env_name}_{steps}.pth"
        torch.save(self.actor.state_dict(), checkpoint_path)

    def soft_update(self, online, target, tau):
        """
        Soft update des réseaux cibles :
        target = tau * online + (1-tau) * target
        """
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            
    def select_action(self, observation, add_noise=False):
        """
        Sélection d'une action selon la politique Actor
        - add_noise=True : ajoute du bruit pour exploration (TD3)
        """
        with torch.no_grad():
            # Conversion observation en tensor batch
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(observation_tensor).squeeze(0)
            
            if add_noise:
                # Ajout du bruit gaussien pour exploration
                noise = self.actor.action_scale * \
                    gaussian_noise(action.shape, 0.0, self.config.noise_explore).to(self.config.device)
                action = torch.clamp(action + noise, min=-self.actor.action_scale, max=self.actor.action_scale)
            
            # Conversion en numpy pour interaction avec l'environnement
            return action.cpu().numpy()
            
    def learn(self, train_actor: bool):
        """
        Effectue un pas d'apprentissage TD3 :
        - Mise à jour des Critics
        - Mise à jour du Actor avec delayed update
        """
        # Sample d'un batch depuis le replay buffer
        observations, actions, rewards, next_observations, terminations =\
            self.buffer.sample(self.config.batch_size)
        
        # Conversion en tensors pour PyTorch
        observations      = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
        actions           = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
        rewards           = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).view(self.config.batch_size,  1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
        terminations      = torch.tensor(np.array(terminations), dtype=torch.float32, device=self.device).view(self.config.batch_size,  1)

        # Update Critics
        with torch.no_grad():
            # Ajout d'un bruit gaussien tronqué à l'action cible pour TD3
            clipped_noise = self.actor.action_scale * gaussian_noise(
                actions.shape[1:], 0.0, self.config.noise_policy, self.config.noise_clip
            ).to(self.config.device)
            
            # Calcul de l'action cible avec bruit
            next_state_actions = torch.clamp(
                self.target_actor(next_observations) + clipped_noise,
                min=-self.actor.action_scale, max=self.actor.action_scale
            )
            
            # Calcul des Q-values cibles (min des deux critics)
            next_state_q = torch.minimum(*self.target_critics(next_observations, next_state_actions))
            target_q = rewards + self.config.gamma ** self.config.num_steps * (1.0 - terminations) * next_state_q  
        
        # Q-values actuelles pour le batch
        current_action_qs = self.critics(observations, actions)
        
        # Perte critic = MSE entre Q(s,a) et Q-target pour Q1/Q2
        critic_loss = torch.sum(torch.stack([F.mse_loss(q, target_q) for q in current_action_qs]))

        # Optimisation des critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        # ==============================
        # Update Actor (Delayed)
        # ==============================
        if train_actor:
            # Utilise Q1 pour la mise à jour du policy gradient
            current_action_q = self.critics.critics[0](observations, self.actor(observations))
            actor_loss = -(current_action_q).mean()  # Gradient ascent sur Q-value

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_norm_clip)
            self.actor_optimizer.step()

            # Soft update des réseaux cibles
            self.soft_update(self.actor, self.target_actor, self.config.tau)
            self.soft_update(self.critics, self.target_critics, self.config.tau)
            
    def train(self):
        """
        Boucle principale d'entraînement TD3
        - Interaction avec l'environnement
        - Stockage dans le replay buffer
        - Apprentissage critic/actor
        - Optionnel checkpoint
        """
        if self.config.verbose:
            print(f"Training {self.config.agent_name} agent...\n")
        
        # Initialisation du logger
        logger = Logger(total_steps=self.config.total_steps, num_checkpoints=self.config.num_checkpoints)
        
        # Reset de l'environnement
        observation, _ = self.env.reset()
        
        # Boucle principale sur les steps
        for step in range(1, self.config.total_steps + 1):

            # Sélection d'action
            if step > self.config.learning_starts:
                action = self.select_action(observation, add_noise=True)
            else:
                # Actions aléatoires avant d'avoir assez de données
                action = self.env.action_space.sample()
            
            # Step dans l'environnement
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            
            # Mise à jour du logger
            logger.log(reward, terminated, truncated)
            
            # Stockage de la transition
            self.buffer.add((observation, action, reward, next_observation, terminated, truncated))
            
            # Apprentissage si assez de transitions
            if len(self.buffer) > self.config.batch_size and step >= self.config.learning_starts:
                train_actor = step % self.config.actor_delay == 0  # Delayed actor update
                self.learn(train_actor)

            # Reset si épisode terminé
            if terminated or truncated:
                next_observation, _ = self.env.reset()
            observation = next_observation
                
            # Affichage des logs
            if self.config.verbose:
                logger.print_logs()
                
            # Sauvegarde périodique
            if self.config.checkpoint and step % logger.checkpoint_interval == 0:
                self.checkpoint(step)
                
            # Early stopping si la reward cible est atteinte
            if self.config.target_reward is not None and len(logger.episode_returns) >= 20:
                mean_reward = np.mean(logger.episode_returns[-20:])
                if mean_reward >= self.config.target_reward:
                    if self.config.verbose:
                        print("\nTarget reward achieved. Training stopped.")
                    break

        if self.config.verbose:
            print("\nTraining complete.")
        
        return logger.logs

    

td3 = TD3()

td3_logs = td3.train()


# Actionneur (policy)
torch.save(td3.actor.state_dict(), "Save/td3_actor_manual.pth")


# Charger un modèle
td3.actor.load_state_dict(torch.load("Save/td3_actor_manual.pth"))
td3.actor.eval()  # passer en mode évaluation 


# Créer un environnement d’évaluation (mode humain ou rgb_array)
eval_env = gym.make(td3.config.env_name, render_mode='human')

eval_rewards = []

for episode in range(10):  # 10 épisodes d'évaluation
    reset_result = eval_env.reset()
    if isinstance(reset_result, tuple):
        state, _ = reset_result
    else:
        state = reset_result

    total_reward = 0
    done = False

    while not done:
        # Action déterministe pour évaluation (stochastic=False)
        action = td3.select_action(state, stochastic=False)

        # Step dans l'environnement
        step_result = eval_env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            next_state, reward, done, _ = step_result
        else:
            raise ValueError(f"Format inattendu pour step(): {step_result}")

        state = next_state
        total_reward += reward

    eval_rewards.append(total_reward)
    print(f"Episode {episode+1}: reward = {total_reward:.2f}")

eval_env.close()

print(f"\nAverage evaluation reward over 10 episodes: {np.mean(eval_rewards):.2f}")
