from logger import Logger
from buffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import gymnasium as gym
import os


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


# Réseau Actor (Soft Actor) pour SAC
class SoftActorNetwork(nn.Module):
    """
    Réseau de politique pour SAC.
    Prédit :
    - la moyenne et le log std de la distribution gaussienne
    - échantillonne des actions continues transformées avec tanh
    """
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * np.prod(action_space.shape))  # mean + log_std
        )
        
        # Buffers pour transformer l’action en limites [low, high] de l'environnement
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32)
        )
        
    def forward(self, observation, a_min=-5, a_max=2):
        """
        Calcule la moyenne et log_std transformé pour la politique
        """
        # Extraction de mean et log_std
        mean, log_std = torch.chunk(self.network(observation), 2, dim=-1)
        
        # Transformation tanh pour borner log_std
        log_std = torch.tanh(log_std)
        
        # Mise à l’échelle et translation du log_std
        log_std = a_min + 0.5 * (a_max - a_min) * (log_std + 1)
        return mean, log_std
    
    def get_action(self, observation, a_min=-5, a_max=2):
        """
        Échantillonne une action à partir de la politique SAC
        et calcule la log-probabilité corrigée pour la transformation tanh
        """
        # Passage forward
        mean, log_std = self.forward(observation, a_min, a_max)
        
        # Distribution normale
        dist = Normal(mean, log_std.exp())
        
        # Échantillonnage avec reparameterization trick
        sampled_action = dist.rsample()
        
        # Transformation tanh
        tanh_action = torch.tanh(sampled_action)
        
        # Mise à l’échelle et translation vers les bornes de l’environnement
        action = self.action_scale * tanh_action + self.action_bias
        
        # Log-probabilité corrigée pour tanh
        log_prob = dist.log_prob(sampled_action)
        log_prob -= torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


# Configuration SAC
class SACConfig:
    """
    Configuration centralisée pour SAC
    """
    env_name: str             = 'Hopper-v5'      # Nom de l’environnement
    agent_name: str           = 'SAC'            # Nom de l’agent
    device: str               = 'cpu'            # Device torch
    checkpoint: bool          = False            # Sauvegarde périodique des poids
    num_checkpoints: int      = 20               # Nombre de checkpoints pour logs
    verbose: bool             = True             # Affichage des logs
    total_steps: int          = 1_000_000        # Nombre total de steps d’entraînement
    target_reward: int | None = 1000             # Récompense cible pour early stopping
    learning_starts: int      = 5000             # Commencer l’apprentissage après ces steps
    gamma: float              = 0.99             # Discount factor
    lr: float                 = 3e-4             # Learning rate
    hidden_dim: int           = 256              # Dimension cachée des réseaux Actor et Critic
    buffer_capacity: int      = 100_000          # Capacité max du replay buffer
    batch_size: int           = 64               # Batch size pour le learner
    num_steps: int            = 3                # Nombre de steps pour n-step returns
    tau: float                = 0.005            # Soft update coefficient pour Critic cible
    grad_norm_clip: float     = 40.0             # Clip global des gradients
    actor_delay: int          = 2                # Steps de critic par step d’actor

    # Paramètre spécifique SAC
    alpha: float              = 0.2              # Coefficient d’entropie SAC


    
# Classe principale SAC
class SAC:
    """
    Soft Actor-Critic (SAC) implementation.
    - CriticEnsemble pour Q1/Q2
    - SoftActorNetwork pour la politique
    - ReplayBuffer avec n-step support
    """
    def __init__(self):
        # Charger la configuration
        config = SACConfig()
        self.device = config.device
        
        # Initialiser l'environnement
        self.env = gym.make(config.env_name)
        observation_space, action_space = self.env.observation_space, self.env.action_space
        
        # Actor (Soft Policy)
        self.actor = SoftActorNetwork(observation_space, action_space, config.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        
        # Critic (Q1/Q2)
        self.critics = CriticEnsemble(observation_space, action_space, config.hidden_dim).to(self.device)
        self.target_critics = CriticEnsemble(observation_space, action_space, config.hidden_dim).to(self.device)
        self.soft_update(self.critics, self.target_critics, 1.0)  # copie complète des poids
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.lr)
        
        # Replay buffer (avec n-step)
        self.buffer = ReplayBuffer(config.buffer_capacity, config.num_steps, config.gamma)
        self.config = config
        
    # Sauvegarde de checkpoint
    def checkpoint(self, steps):
        """
        Sauvegarde les poids de l'acteur sur disque
        """
        if not os.path.exists('models'):
            os.makedirs('models')
        checkpoint_path = f"models/{self.config.agent_name}_{self.config.env_name}_{steps}.pth"
        torch.save(self.actor.state_dict(), checkpoint_path)

    # Soft update du critic cible
    def soft_update(self, online, target, tau):
        """
        Soft update : target = tau * online + (1-tau) * target
        """
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            
    # Sélection d'action
    def select_action(self, observation, stochastic=True):
        """
        Sélectionne une action avec la politique SAC
        - stochastic=True : action échantillonnée
        - stochastic=False : action déterministe (mean)
        """
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, log_std = self.actor(observation_tensor)
            if stochastic:
                dist = Normal(mean, log_std.exp())
                action = dist.rsample()
            else:
                action = mean
            # Transformation tanh + mise à l’échelle selon l'environnement
            action = self.actor.action_scale * torch.tanh(action) + self.actor.action_bias
            return action.squeeze(0).cpu().numpy()
            
    # Une étape d’apprentissage
    def learn(self, train_actor: bool):
        """
        Effectue un pas d’apprentissage SAC
        - update du critic
        - update du actor si train_actor=True
        """
        # Échantillonner batch du buffer
        observations, actions, rewards, next_observations, terminations =\
            self.buffer.sample(self.config.batch_size)
        
        # Conversion en tensors
        observations      = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
        actions           = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
        rewards           = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).view(self.config.batch_size, 1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.device).view(self.config.batch_size, -1)
        terminations      = torch.tensor(np.array(terminations), dtype=torch.float32, device=self.device).view(self.config.batch_size, 1)

        # Update Critic
        with torch.no_grad():
            # Actions prochaines selon l'acteur
            next_state_actions, log_probs = self.actor.get_action(next_observations)
            
            # Q-target avec entropie
            next_state_qs = self.target_critics(next_observations, next_state_actions)
            next_state_q = torch.minimum(*next_state_qs) - self.config.alpha * log_probs
            target_q = rewards + self.config.gamma ** self.config.num_steps * (1.0 - terminations) * next_state_q  
        
        # Q-values actuelles
        current_action_qs = self.critics(observations, actions)
        critic_loss = torch.sum(torch.stack([F.mse_loss(q, target_q) for q in current_action_qs]))

        # Optimisation critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        # Update Actor
        if train_actor:
            current_actions, log_probs = self.actor.get_action(observations)
            
            # Loss actor avec entropie
            current_action_q = torch.minimum(*self.critics(observations, current_actions))
            actor_loss = ((self.config.alpha * log_probs) - current_action_q).mean()

            # Optimisation actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_norm_clip)
            self.actor_optimizer.step()

        # Soft update du critic cible
        self.soft_update(self.critics, self.target_critics, self.config.tau)
            
    # Boucle d’entraînement principale
    def train(self):
        """
        Entraîne l'agent SAC selon la configuration
        """
        if self.config.verbose:
            print(f"Training {self.config.agent_name} agent...\n")
        
        # Initialisation du Logger
        logger = Logger(total_steps=self.config.total_steps, num_checkpoints=self.config.num_checkpoints)
        
        # Réinitialisation de l'environnement
        observation, _ = self.env.reset()
        
        # Boucle principale
        for step in range(1, self.config.total_steps + 1):
            # Sélection d'action et interaction avec l'environnement
            action = self.select_action(observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            
            # Mise à jour du logger
            logger.log(reward, terminated, truncated)
            
            # Stockage dans le replay buffer
            self.buffer.add((observation, action, reward, next_observation, terminated, truncated))
            
            # Apprentissage si assez de transitions
            if len(self.buffer) > self.config.batch_size and step >= self.config.learning_starts:
                train_actor = step % self.config.actor_delay == 0
                self.learn(train_actor)

            # Réinitialisation si épisode terminé
            if terminated or truncated:
                next_observation, _ = self.env.reset()
            observation = next_observation
                
            # Affichage du logger
            if self.config.verbose:
                logger.print_logs()
                
            # Sauvegarde périodique
            if self.config.checkpoint and step % logger.checkpoint_interval == 0:
                self.checkpoint(step)
                
            # Arrêt anticipé si objectif atteint
            if self.config.target_reward is not None and len(logger.episode_returns) >= 20:
                mean_reward = np.mean(logger.episode_returns[-20:])
                if mean_reward >= self.config.target_reward:
                    if self.config.verbose:
                        print("\nTarget reward achieved. Training stopped.")
                    break

        # Fin de l’entraînement
        if self.config.verbose:
            print("\nTraining complete.")
        
        return logger.logs
    def evaluate_policy(self, num_episodes=5, render=False):
        """
        Évalue la politique actuelle sur eval_env.
        Retourne la récompense moyenne.
        """
        total_rewards = []
    
        for _ in range(num_episodes):
            reset_result = self.eval_env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
    
            done = False
            episode_reward = 0
            frames = []
    
            while not done:
                action = self.select_action(state, stochastic=False)  # action déterministe pour eval
                step_result = self.eval_env.step(action)
    
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    next_state, reward, done, _ = step_result
                else:
                    raise ValueError(f"Format inattendu pour step(): {step_result}")
    
                episode_reward += reward
                if render:
                    frames.append(self.eval_env.render())
                state = next_state
    
            total_rewards.append(episode_reward)
    
        avg_reward = np.mean(total_rewards)
        return avg_reward

    


    
sac = SAC()

# Activer la sauvegarde périodique
# sac.config.checkpoint = True


sac_logs = sac.train()

# Actionneur (policy)
torch.save(sac.actor.state_dict(), "Save/sac_actor_manual_hopper.pth")
torch.save(sac.critics.state_dict(), "Save/sac_critic_manual_hopper.pth")

# Charger un modèle
sac.actor.load_state_dict(torch.load("Save/sac_actor_manual_hopper.pth"))
sac.actor.eval()  # passer en mode évaluation 


# Créer un environnement d’évaluation (mode humain ou rgb_array)
eval_env = gym.make(sac.config.env_name, render_mode='human')

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
        action = sac.select_action(state, stochastic=False)

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


r'''
torch.save({
    'actor_state_dict': sac.actor.state_dict(),
    'critics_state_dict': sac.critics.state_dict(),
    'actor_optimizer_state_dict': sac.actor_optimizer.state_dict(),
    'critic_optimizer_state_dict': sac.critic_optimizer.state_dict(),
    'config': sac.config,
}, "sac_full_manual.pth")



checkpoint = torch.load("sac_full_manual.pth")
sac.actor.load_state_dict(checkpoint['actor_state_dict'])
sac.critics.load_state_dict(checkpoint['critics_state_dict'])
sac.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
sac.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
sac.config = checkpoint['config']

'''