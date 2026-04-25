import numpy as np
import torch
from torch import nn
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from collections import deque
import time

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape),  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, np.prod(action_space.shape)),
            nn.Tanh()
        )
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32)
        )
        
    def forward(self, observation):
        action = self.network(observation)
        return action * self.action_scale + self.action_bias
    
class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape) + np.prod(action_space.shape),  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=1))



# ======================================================
# Ornstein-Uhlenbeck Noise (exploration continue)
# ======================================================

class OrnsteinUhlenbeckNoise:
    """
    Générateur de bruit d'Ornstein-Uhlenbeck.
    Utilisé pour ajouter un bruit corrélé temporellement
    aux actions (souvent en DDPG).
    """

    def __init__(self, size: int, mu=0.0, sigma=0.1, theta=0.15):
        self.size  = size
        self.mu    = mu * torch.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.reset()

    def reset(self):
        """Réinitialise le bruit à sa moyenne."""
        self.state = self.mu.clone()

    def sample(self):
        """Génère un nouvel échantillon de bruit."""
        dx = self.theta * (self.mu - self.state) \
             + self.sigma * torch.randn(self.size)
        self.state += dx
        return self.state.clone()


# ======================================================
# Replay Buffer avec support n-step
# ======================================================

class ReplayBuffer:
    """
    Buffer de rejouabilité pour stocker les transitions
    et échantillonner des minibatchs pour l'apprentissage.
    """

    def __init__(self, capacity, num_steps=1, gamma=0.99):
        self.buffer         = deque(maxlen=capacity)
        self.n_step_buffer  = deque(maxlen=num_steps)
        self.num_steps      = num_steps
        self.gamma          = gamma

    def add(self, transition):
        """
        Ajoute une transition au buffer.
        Transition attendue (nouvelle API Gym) :
        (s, a, r, s', terminated, truncated)
        """
        assert len(transition) == 6, \
            "Format attendu : (s, a, r, s', terminated, truncated)"

        # Cas 1-step classique
        if self.num_steps == 1:
            observation, action, reward, next_observation, terminated, _ = transition
            self.buffer.append(
                (observation, action, reward, next_observation, terminated)
            )
            return

        # Cas n-step
        self.n_step_buffer.append(transition)

        # Calcul de la récompense n-step
        n_step_reward = 0.0
        for _, _, reward, _, _, _ in reversed(self.n_step_buffer):
            n_step_reward = reward + self.gamma * n_step_reward

        # Première transition du buffer n-step
        observation, action, _, _, _, _ = self.n_step_buffer[0]

        # Dernière transition
        _, _, _, final_observation, terminated, truncated = transition

        # Ajout au buffer principal si n-step atteint
        if len(self.n_step_buffer) == self.num_steps:
            self.buffer.append(
                (observation, action, n_step_reward, final_observation, terminated)
            )

        # Si épisode terminé, on vide le buffer temporaire
        if terminated or truncated:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        """Échantillonne un minibatch aléatoire."""
        observations, actions, rewards, next_observations, terminations = zip(
            *random.sample(self.buffer, batch_size)
        )
        return observations, actions, rewards, next_observations, terminations

    def __len__(self):
        return len(self.buffer)


# Soft update des réseaux cibles

def soft_update(online, target, tau):
    """
    Mise à jour douce des paramètres du réseau cible :
    target = tau * online + (1 - tau) * target
    """
    for online_param, target_param in zip(online.parameters(),
                                          target.parameters()):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )


class Logger:
    """
    Logger utilisé pour suivre l'entraînement :
    - nombre total de pas
    - retours par épisode
    - longueur des épisodes
    - vitesse d'entraînement (FPS)
    """

    def __init__(self, total_steps: int, num_checkpoints: int):
        # Compteurs globaux
        self.current_step     = 0
        self.current_episode  = 1
        self.current_return   = 0.0
        self.current_length   = 0

        # Historique des épisodes
        self.episode_returns  = []
        self.episode_lengths = []

        # Logs personnalisés (ex: losses)
        self.custom_logs      = {}
        self.custom_log_keys  = []

        # Gestion du temps
        self.start_time            = time.time()
        self.last_checkpoint_time  = self.start_time
        self.last_checkpoint_step  = 0

        # Paramètres globaux
        self.total_steps       = total_steps
        self.num_checkpoints   = num_checkpoints
        self.checkpoint_interval = max(1, total_steps // num_checkpoints)

        # Affichage
        self.header_printed = False
        self.log_interval   = 100  # Affichage tous les X pas
        self.window         = 20   # Fenêtre pour moyenne glissante

    def log(self, reward: float, termination: bool, truncation: bool, **kwargs):
        """
        Met à jour les statistiques à chaque pas de l'environnement.

        Args:
            reward      : récompense obtenue
            termination : fin naturelle de l'épisode
            truncation  : fin forcée (timeout)
            kwargs      : logs supplémentaires (ex: loss)
        """
        self.current_step   += 1
        self.current_return += reward
        self.current_length += 1

        # Fin d'épisode
        if termination or truncation:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_episode += 1
            self.current_return  = 0.0
            self.current_length  = 0

        # Logs personnalisés
        for key, value in kwargs.items():
            if key not in self.custom_log_keys:
                self.custom_log_keys.append(key)
            self.custom_logs[key] = value

    def print_logs(self):
        """
        Affiche l'état de l'entraînement :
        progression, récompense moyenne, FPS, temps écoulé.
        """
        if self.current_step % self.log_interval == 0 and self.episode_returns:
            elapsed_time = time.time() - self.start_time

            # FPS depuis le dernier checkpoint
            steps_since_checkpoint = self.current_step - self.last_checkpoint_step
            time_since_checkpoint  = time.time() - self.last_checkpoint_time
            fps = steps_since_checkpoint / max(time_since_checkpoint, 1e-6)

            # Statistiques
            progress = 100 * self.current_step / self.total_steps
            mean_reward = np.mean(self.episode_returns[-self.window:])
            mean_length = np.mean(self.episode_lengths[-self.window:])

            # Format temps hh:mm:ss
            h, r = divmod(int(elapsed_time), 3600)
            m, s = divmod(r, 60)
            formatted_time = f"{h:02}:{m:02}:{s:02}"

            # Impression de l'en-tête (une seule fois)
            if not self.header_printed:
                header = (
                    f"{'Progress':>8} | {'Step':>8} | {'Episode':>8} | "
                    f"{'Mean Rew':>8} | {'Mean Len':>8} | {'FPS':>6} | {'Time':>8}"
                )
                for key in self.custom_log_keys:
                    header += f" | {key:>{len(key)}}"
                print(header)
                self.header_printed = True

            # Ligne de log
            line = (
                f"{progress:>7.1f}% | "
                f"{self.current_step:>8,} | "
                f"{self.current_episode:>8,} | "
                f"{mean_reward:>8.2f} | "
                f"{mean_length:>8.1f} | "
                f"{fps:>6.0f} | "
                f"{formatted_time:>8}"
            )

            for key in self.custom_log_keys:
                value = self.custom_logs.get(key, 0)
                line += f" | {value:>{len(key)}}"

            print(f"\r{line}", end="")

        # Mise à jour du checkpoint
        if self.current_step % self.checkpoint_interval == 0:
            print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step

    @property
    def logs(self):
        """Retourne toutes les statistiques finales."""
        return {
            "total_steps": self.current_step,
            "total_episodes": self.current_episode - 1,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "best_reward": np.max(self.episode_returns) if self.episode_returns else None,
            "total_duration": time.time() - self.start_time,
            "mean_fps": self.current_step / (time.time() - self.start_time + 1e-6),
            "custom_logs": self.custom_logs
        }

    
    

class DDPG:
    """
    Implémentation complète de l'algorithme DDPG
    (Deep Deterministic Policy Gradient).
    """

    def __init__(self):
        config = DDPGConfig()
        self.device = config.device

        # Environnement
        self.env = gym.make(config.env_name)
        obs_space, act_space = self.env.observation_space, self.env.action_space

        # Réseau Actor
        self.actor = ActorNetwork(obs_space, act_space, config.hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(obs_space, act_space, config.hidden_dim).to(self.device)
        self.soft_update(self.actor, self.target_actor, 1.0)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)

        # Réseau Critic
        self.critic = CriticNetwork(obs_space, act_space, config.hidden_dim).to(self.device)
        self.target_critic = CriticNetwork(obs_space, act_space, config.hidden_dim).to(self.device)
        self.soft_update(self.critic, self.target_critic, 1.0)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        # Replay buffer + bruit OU
        self.buffer = ReplayBuffer(config.buffer_capacity, config.num_steps, config.gamma)
        self.noise_generator = OrnsteinUhlenbeckNoise(
            size=np.prod(act_space.shape),
            sigma=config.noise_sigma,
            theta=config.noise_theta
        )

        self.config = config

        
    def checkpoint(self, steps):
        "Saves model weights to disk."
        if not os.path.exists('models'):
            os.makedirs('models')
        checkpoint_path = f"models/{self.config.agent_name}_{self.config.env_name}_{steps}.pth"
        torch.save(self.actor.state_dict(), checkpoint_path)

    def soft_update(self, online, target, tau):
        "Performs a soft update of the target network parameters."
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
            
    def select_action(self, observation, add_noise=False):
        "Selects an action using the current policy with optional noise."
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(observation_tensor).squeeze(0)
            if add_noise:
                noise = self.actor.action_scale * self.noise_generator.sample().to(self.device)
                action = torch.clamp(action + noise, min=-self.actor.action_scale, max=self.actor.action_scale)
            return action.cpu().numpy()

    def learn(self):
        "Perform a single learning step."
        # Sample and format experience data
        observations, actions, rewards, next_observations, terminations =\
            self.buffer.sample(self.config.batch_size)
        
        observations      = torch.tensor(np.array(observations), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size, -1)
        actions           = torch.tensor(np.array(actions), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size, -1)
        rewards           = torch.tensor(np.array(rewards), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size,  1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size, -1)
        terminations      = torch.tensor(np.array(terminations), dtype=torch.float32, 
                                         device=self.device).view(self.config.batch_size,  1)

        # Critic loss and param update
        with torch.no_grad():
            next_state_q = self.target_critic(next_observations, self.target_actor(next_observations))
            target_q = rewards + self.config.gamma ** self.config.num_steps * (1.0 - terminations) * next_state_q  
        current_action_q = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_action_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_norm_clip)
        self.critic_optimizer.step()

        # Actor loss and param update
        current_action_q = self.critic(observations, self.actor(observations))
        actor_loss = -(current_action_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_norm_clip)
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.actor, self.target_actor, self.config.tau)
        self.soft_update(self.critic, self.target_critic, self.config.tau)

    def train(self):
        "Trains DDPG agent based on the provided configuration."
        if self.config.verbose:
            print(f"Training {self.config.agent_name} agent...\n")
        
        # Initialise Logger
        logger = Logger(total_steps=self.config.total_steps, num_checkpoints=self.config.num_checkpoints)
        
        # Reset environment
        observation, _ = self.env.reset()
        
        # Main training loop
        for step in range(1, self.config.total_steps + 1):
            
            # Select action
            if step > self.config.learning_starts:
                action = self.select_action(observation, add_noise=True)
            else:
                # Random if not yet learning
                action = self.env.action_space.sample()
                
            # Environment step
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            
            # Update logs
            logger.log(reward, terminated, truncated)
            
            # Push experience to buffer
            self.buffer.add((observation, action, reward, next_observation, terminated, truncated))
            
            # Perform learning step
            if len(self.buffer) > self.config.batch_size and step >= self.config.learning_starts:
                self.learn()

            # Reset environment and noise if episode ended
            if terminated or truncated:
                next_observation, _ = self.env.reset()
                self.noise_generator.reset()
            observation = next_observation
                
            # Print training info if verbose
            if self.config.verbose:
                logger.print_logs()
                
            # Save weights if checkpointing
            if self.config.checkpoint and step % logger.checkpoint_interval == 0:
                self.checkpoint(step)
                
            # Check stopping condition
            if self.config.target_reward is not None and len(logger.episode_returns) >= 20:
                mean_reward = np.mean(logger.episode_returns[-20:])
                if mean_reward >= self.config.target_reward:
                    if self.config.verbose:
                        print("\nTarget reward achieved. Training stopped.")
                    break

        # Training ended
        if self.config.verbose:
            print("\nTraining complete.")
        
        return logger.logs
    


    
class DDPGConfig:
    env_name: str             = 'Pendulum-v1'  # Environment name
    agent_name: str           =      'DDPG'  # Agent name
    device: str               =      'cuda' if torch.cuda.is_available() else 'cpu'  # Torch device
    checkpoint: bool          =       False  # Periodically save model weights
    num_checkpoints: int      =          20  # Number of checkpoints/printing logs to create
    verbose: bool             =        True  # Verbose printing
    total_steps: int          =     100_000  # Total training steps
    target_reward: int | None =        -200  # Target reward used for early stopping
    learning_starts: int      =        5000  # Begin learning after this many steps
    gamma: float              =        0.99  # Discount factor
    lr: float                 =        3e-4  # Learning rate
    hidden_dim: int           =         256  # Actor and critic network hidden dim
    buffer_capacity: int      =     100_000  # Maximum replay buffer capacity
    batch_size: int           =          64  # Batch size used by learner
    num_steps: int            =           3  # Number of steps to unroll Bellman equation by
    tau: float                =       0.005  # Soft target network update interpolation coefficient
    grad_norm_clip: float     =        40.0  # Global gradient clipping value
    noise_sigma: float        =         0.1  # OU noise standard deviation
    noise_theta: float        =        0.15  # OU noise reversion rate    
    
    
ddpg = DDPG()
ddpg_logs = ddpg.train()


def plot_rewards(logs, window=5):
    rewards = logs['episode_returns']
    moving_avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward per Episode', lw=3, c='#636EFA')
    plt.plot(moving_avg_rewards, label=f'{window}-Episode Moving Average', lw=3, c='#636EFA', ls='--', alpha=0.5)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episodic Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rewards(ddpg_logs, window=5)

