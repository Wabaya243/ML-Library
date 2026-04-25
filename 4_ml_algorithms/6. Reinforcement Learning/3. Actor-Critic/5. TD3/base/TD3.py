import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_shape, action_dim):
        super().__init__()
        
        hidden_size = 256
        
        self.dense_1 = nn.Linear(state_shape, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.dense_3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = torch.tanh(self.dense_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_shape, action_dim):
        super().__init__()
        
        hidden_size = 256
        
        # Q1 architecture
        self.dense_1 = nn.Linear(state_shape + action_dim, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.dense_3 = nn.Linear(hidden_size, 1)
        
        # Q2 architecture
        self.dense_4 = nn.Linear(state_shape + action_dim, hidden_size)
        self.dense_5 = nn.Linear(hidden_size, hidden_size)
        self.dense_6 = nn.Linear(hidden_size, 1)

    def forward(self, x, x_actions):
        x = torch.cat([x, x_actions], dim=1)

        q1 = F.relu(self.dense_1(x))
        q1 = F.relu(self.dense_2(q1))
        q1 = self.dense_3(q1)

        q2 = F.relu(self.dense_4(x))
        q2 = F.relu(self.dense_5(q2))
        q2 = self.dense_6(q2)

        return q1, q2

    def forward_q1(self, x, x_actions):
        x = torch.cat([x, x_actions], dim=1)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x
    

class ReplayBuffer:
    def __init__(self, max_len, state_shape, action_dim, device):
        self.device = device
        
        # Nombre maximum d'éléments pouvant être stockés dans le buffer
        self.max_len = max_len
        
        # Buffers pour stocker les états, actions, récompenses, états suivants et flags "done"
        self.state_buffer = torch.zeros((max_len, state_shape), dtype=torch.float32).to(device)
        self.action_buffer = torch.zeros((max_len, action_dim), dtype=torch.float32).to(device)
        self.reward_buffer = torch.zeros(max_len, dtype=torch.float32).to(device)
        self.next_state_buffer = torch.zeros((max_len, state_shape), dtype=torch.float32).to(device)
        self.done_buffer = torch.zeros(max_len, dtype=torch.float32).to(device)
        
        # Pointeur indiquant où écrire la prochaine transition
        self.ptr = 0
        
        # Taille actuelle du buffer (nombre d'éléments réellement stockés)
        self.size = 0

    def __len__(self):
        # Retourne la taille actuelle du buffer
        return self.size

    def append(self, state, action, reward, next_state, done):
        """
        Ajoute une transition (state, action, reward, next_state, done) dans le buffer.
        Si le buffer est plein, il écrase les anciennes transitions de manière circulaire.
        """
        self.state_buffer[self.ptr] = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.action_buffer[self.ptr] = torch.tensor(action, dtype=torch.float32).to(self.device)
        self.reward_buffer[self.ptr] = torch.tensor(reward, dtype=torch.float32).to(self.device)
        self.next_state_buffer[self.ptr] = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        self.done_buffer[self.ptr] = torch.tensor(done, dtype=torch.float32).to(self.device)

        # Avancer le pointeur circulairement
        self.ptr = (self.ptr + 1) % self.max_len
        # Mettre à jour la taille réelle du buffer
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un batch de transitions du buffer.
        Retourne : states, actions, rewards, next_states, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        states = self.state_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_states = self.next_state_buffer[indices]
        dones = self.done_buffer[indices]
        return states, actions, rewards, next_states, dones
    
    
class GaussianNoiseGenerator:
    def __init__(self, sigma=0.1):
        # Écart-type du bruit gaussien
        self.sigma = sigma
    
    def sample(self, *args):
        """
        Génère un bruit gaussien de forme spécifiée par args
        (par exemple, shape=(batch_size, action_dim))
        """
        return self.sigma * np.random.randn(*args)


class Logger:
    def __init__(self):
        # Nombre de pas depuis le début de l'épisode courant
        self.steps = 0
        # Nombre total de pas (peut être défini par l'entraînement)
        self.total_steps = 0
        # Imprimer une nouvelle ligne toutes les 'new_line_every' étapes
        self.new_line_every = 25000
        # Récompense cumulée dans l'épisode courant
        self.cumulative_reward = 0
        # Nombre de pas dans l'épisode courant
        self.current_episode_length = 0
        # Historique des récompenses par épisode
        self.episode_rewards = []
        # Historique des longueurs des épisodes
        self.episode_lengths = []
        # Moyenne mobile des récompenses sur les 50 derniers épisodes
        self.episode_rewards_ma = 0
        # Moyenne mobile des longueurs d'épisodes sur les 50 derniers épisodes
        self.episode_lengths_ma = 0
        
    def log(self, reward, done):
        """
        Enregistre la récompense à chaque étape et met à jour l'historique 
        à la fin de l'épisode (si done=True)
        """
        self.cumulative_reward += reward
        self.current_episode_length += 1

        if done:
            # Ajouter les statistiques de l'épisode terminé
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_lengths.append(self.current_episode_length)
            # Calculer la moyenne mobile sur les 50 derniers épisodes
            self.episode_rewards_ma = np.mean(self.episode_rewards[-50:])
            self.episode_lengths_ma = np.mean(self.episode_lengths[-50:])
            # Réinitialiser les compteurs pour le nouvel épisode
            self.cumulative_reward = 0
            self.current_episode_length = 0
        
        # Incrémenter le nombre de pas
        self.steps += 1
        
    def print_logs(self):
        """
        Affiche les statistiques d'entraînement actuelles :
        - Steps effectués
        - Récompense moyenne par épisode
        - Longueur moyenne des épisodes
        """
        # Décider si on imprime sur la même ligne ou sur une nouvelle ligne
        end_char = "\n" if self.steps % self.new_line_every == 0 else "\r"
        print(
            f"Step: {self.steps}/{self.total_steps} | "
            f"Avg reward per episode: {self.episode_rewards_ma:.4f} | "
            f"Avg steps per episode: {self.episode_lengths_ma:.2f}", 
            end=end_char
        )


# Configuration des hyperparamètres pour l'agent RL
config = {
    'learning_rate_actor': 0.001,   # Taux d'apprentissage de la politique (actor)
    'learning_rate_critic': 0.001,  # Taux d'apprentissage de la fonction de valeur (critic)
    'tau': 0.005,                   # Taux de mise à jour des réseaux cibles (soft update)
    'buffer_max_length': 500000,    # Taille maximale du replay buffer
    'batch_size': 256,              # Nombre de transitions échantillonnées par batch
    'start_timesteps': 10000,       # Nombre de pas initiaux avec politique aléatoire
    'updates_per_step': 1,          # Nombre de mises à jour par pas de l'environnement
    'policy_freq': 2,               # Fréquence des mises à jour différées de la politique
    'gamma': 0.99,                  # Facteur d'actualisation des récompenses (discount factor)
    'sigma': 0.2,                   # Écart-type du bruit gaussien pour exploration
    'clip_noise': 0.5,              # Amplitude maximale du bruit appliqué aux actions cibles
    'action_low': -1,               # Valeur minimale possible pour les actions
    'action_high': 1                # Valeur maximale possible pour les actions
}


class TD3Agent:
    def __init__(self, env, Actor, Critic, config):
        """
        Initialisation de l'agent TD3.
        
        args:
            env (gym.Env) : L'environnement gym
            Actor (torch.nn.Module) : Classe du réseau de politique (actor)
            Critic (torch.nn.Module) : Classe du réseau de valeur (critic)
            config (dict) : Dictionnaire contenant les hyperparamètres TD3
        """
        self.env = env
        
        # Dimensions des états et actions
        self.state_shape = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Device (GPU si disponible, sinon CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialiser les réseaux Actor et Critic
        self.actor = Actor(self.state_shape, self.action_dim).to(self.device)
        self.critic = Critic(self.state_shape, self.action_dim).to(self.device)

        # Réseaux cibles pour TD3 (Actor et Critic)
        self.actor_target = Actor(self.state_shape, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_shape, self.action_dim).to(self.device)
        
        # Charger la configuration depuis le dictionnaire
        self.learning_rate_actor = config['learning_rate_actor']
        self.learning_rate_critic = config['learning_rate_critic']
        self.tau = config['tau']
        self.buffer_max_length = config['buffer_max_length']
        self.batch_size = config['batch_size']
        self.start_timesteps = config['start_timesteps']
        self.updates_per_step = config['updates_per_step']
        self.policy_freq = config['policy_freq']
        self.gamma = config['gamma']
        self.sigma = config['sigma']
        self.clip_noise = config['clip_noise']
        self.action_low = config['action_low']
        self.action_high = config['action_high']

        # Optimizers pour actor et critic
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        
        # Replay buffer pour stocker les transitions
        self.buffer = ReplayBuffer(self.buffer_max_length, self.state_shape, self.action_dim, self.device)

        # Générateur de bruit gaussien pour exploration
        self.gaussian_noise = GaussianNoiseGenerator(self.sigma)
        
        # Copier les paramètres initiaux dans les réseaux cibles
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Logger pour suivre récompenses et longueurs d'épisodes
        self.logger = Logger()
        
    def train(self, max_timesteps):
        """
        Entraîne l'agent pour un nombre maximum de timesteps.
        """
        # Réinitialiser l'environnement
        state, info = self.env.reset()
        
        # Définir le nombre total de pas pour le logger
        self.logger.total_steps = max_timesteps
        
        for current_timestep in range(max_timesteps):
            
            # Phase d'exploration avec actions aléatoires
            if len(self.buffer) < self.start_timesteps:
                action = self.env.action_space.sample()
            else:
                # Sinon, choisir action via l'actor + bruit gaussien pour exploration
                action = self.act(state)
                noise = self.gaussian_noise.sample(self.action_dim)
                action = np.clip(action + noise, a_min=self.action_low, a_max=self.action_high)

            # Effectuer l'action dans l'environnement
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Stocker la transition dans le replay buffer
            self.buffer.append(state, action, reward, next_state, done)
            
            # Logger et afficher progression
            self.logger.log(reward, done)
            self.logger.print_logs()
            
            # Réinitialiser l'état si épisode terminé
            if done:
                state, info = self.env.reset()
            else:
                state = next_state
            
            # Ne pas apprendre tant que la phase d'exploration n'est pas terminée
            if current_timestep < self.start_timesteps:
                continue
                
            # Échantillonner un batch de transitions depuis le buffer
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            # Calculer la cible pour le critic
            with torch.no_grad():
                # Générer bruit pour actions cibles
                noise = torch.tensor(
                    self.gaussian_noise.sample(self.batch_size, self.action_dim),
                    dtype=torch.float,
                    device=self.device
                ).clip(-self.clip_noise, self.clip_noise)
                
                # Calculer actions cibles + bruit, puis clipper
                next_actions = torch.clip(self.actor_target(next_states) + noise, min=self.action_low, max=self.action_high)

                # Prédictions Q pour les actions cibles
                targets_q1, targets_q2 = self.critic(next_states, next_actions)
                targets_q = torch.min(targets_q1, targets_q2).squeeze(-1)
                targets_q = rewards + self.gamma * (1 - dones) * targets_q

            # Prédictions Q actuelles
            pred_q1, pred_q2 = self.critic(states, actions)

            # Calcul de la perte du critic
            loss_critic = F.mse_loss(pred_q1.squeeze(-1), targets_q) + F.mse_loss(pred_q2.squeeze(-1), targets_q)

            # Backprop pour critic
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            # Mise à jour différée de la politique (Actor)
            if current_timestep % self.policy_freq == 0:
                # Calcul de la perte de l'actor
                loss_actor = -self.critic.forward_q1(states, self.actor(states)).mean()
                
                # Backprop Actor
                self.optimizer_actor.zero_grad()
                loss_actor.backward()
                self.optimizer_actor.step()
                
                # Mise à jour des paramètres cibles (soft update)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def act(self, state):
        """
        Prédit une action pour un état donné sans calculer les gradients.
        """
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            return self.actor(state_tensor).detach().cpu().numpy()
        

# Wrappers pour gym
class ObservationScalingWrapper(gym.ObservationWrapper):
    """
    Wrapper pour normaliser certaines observations entre -1 et +1.
    Exemple : on réduit la vitesse angulaire d'un facteur 0.125
    """
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation[2] *= 0.125  # Réduction de la vitesse angulaire
        return observation
    

class RewardScalingWrapper(gym.RewardWrapper):
    """
    Wrapper pour scaler les récompenses avec un facteur donné.
    """
    def __init__(self, env, scaling_factor=1/10.0):
        super().__init__(env)
        self.scaling_factor = scaling_factor

    def reward(self, reward):
        return reward * self.scaling_factor
    

class ActionScalingWrapper(gym.ActionWrapper):
    """
    Wrapper pour scaler les actions : ici on les double.
    Utile si l'environnement attend une plage différente.
    """
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return action * 2

from bokeh.plotting import figure, show
from bokeh.models import NumeralTickFormatter
from bokeh.io import output_notebook

# Création et wrapping de l'environnement
env = gym.make('Pendulum-v1')

# On applique les wrappers dans l'ordre :
# ObservationScalingWrapper : normalisation des observations
# RewardScalingWrapper : réduction de l'amplitude des récompenses
# ActionScalingWrapper : ajuste les actions à la plage attendue par l'environnement
env = ActionScalingWrapper(RewardScalingWrapper(ObservationScalingWrapper(env)))

# Création et entraînement de l'agent TD3
agent_pendulum = TD3Agent(env, Actor, Critic, config)

# Entraînement pour 75 000 steps
agent_pendulum.train(75000)

# Fonction pour visualiser l'entraînement
def plot_training(logger, moving_average=2000, title=""):
    """
    Trace les récompenses obtenues pendant l'entraînement.
    Affiche une moyenne mobile et un intervalle ±1 écart-type.
    
    Args:
        logger : instance de Logger utilisée pour stocker les récompenses
        moving_average : fenêtre de moyenne mobile
        title : titre additionnel pour le graphique
    """
    episode_rewards = np.array(logger.episode_rewards)
    episode_lengths = np.array(logger.episode_lengths)
    total_steps = np.sum(episode_lengths)

    # Crée un tableau répété des récompenses pour chaque step
    x = np.arange(total_steps)
    y = np.repeat(episode_rewards, episode_lengths)

    # Moyenne mobile pour lisser la courbe
    window = np.ones(moving_average) / moving_average
    moving_avg = np.convolve(y, window, mode='same')

    # Calcul de l'écart-type glissant
    rolling_std = np.sqrt(np.convolve(y ** 2, window, mode='same') - moving_avg ** 2)
    lower_band = moving_avg - rolling_std
    upper_band = moving_avg + rolling_std

    # Création du graphique
    p = figure(
        title="Training Performance" + title,
        x_axis_label='Steps',
        y_axis_label='Reward',
        x_range=(0, total_steps),
        width=800,
        height=500
    )
    # Moyenne mobile en ligne
    p.line(x[:len(moving_avg)], moving_avg, line_width=2, line_color='#636EFA')
    # Bande ±1 écart-type
    p.varea(x=x[:len(moving_avg)], y1=lower_band, y2=upper_band, fill_alpha=0.2)

    # Formattage
    p.xaxis.formatter = NumeralTickFormatter(format="0")
    p.title.text_font_size = '16pt'

    output_notebook(hide_banner=True)
    show(p)

# Remise à l'échelle des récompenses pour visualisation
# On remet les récompenses à leur amplitude originale (multiplié par 10)
agent_pendulum.logger.episode_rewards = [r * 10 for r in agent_pendulum.logger.episode_rewards]

# Tracer les récompenses obtenues pendant l'entraînement
plot_training(agent_pendulum.logger, moving_average=2000, title=": Pendulum")

# Wrappers spécifiques pour un autre environnement (ex : Lander)
class ObservationScalingWrapper(gym.ObservationWrapper):
    """
    Wrapper pour normaliser les observations entre -1 et +1.
    Ajuste les positions, vitesses et angles pour un meilleur apprentissage.
    """
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation[0] *= 0.6666       # Coordonnée x
        observation[1] *= 0.6666       # Coordonnée y
        observation[2] *= 0.2          # Vitesse x
        observation[3] *= 0.2          # Vitesse y
        observation[4] *= (1./3.14156) # Angle
        observation[5] *= 0.2          # Vitesse angulaire
        return observation
    
class RewardScalingWrapper(gym.RewardWrapper):
    """
    Wrapper pour réduire la magnitude des récompenses.
    """
    def __init__(self, env, scaling_factor=1./10):
        super().__init__(env)
        self.scaling_factor = scaling_factor

    def reward(self, reward):
        return reward * self.scaling_factor

    
    
env = gym.make('LunarLander-v2', continuous=True)
env = RewardScalingWrapper(ObservationScalingWrapper(env))

agent_lunar = TD3Agent(env, Actor, Critic, config)
agent_lunar.train(250000)


# Scale rewards back
agent_lunar.logger.episode_rewards = [r * 10 for r in agent_lunar.logger.episode_rewards]

# Plot rewards during training
plot_training(agent_lunar.logger, moving_average=5000, title=": LunarLander")


class ObservationScalingWrapper(gym.ObservationWrapper):
    "Scale observations between the range of -1 to +1"
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation[0] *= 1. / 3.14       # hull angle speed
        observation[1] *= 0.2             # angular velocity
        observation[2] *= 0.2             # horizontal speed
        observation[3] *= 0.2             # vertical speed
        observation[4] *= 1. / 3.14       # position of joints 1
        observation[5] *= 0.2             # joints angular speed 1
        observation[6] *= 1.0 / 3.14      # position of joints 2
        observation[7] *= 0.2             # joints angular speed 2
        observation[8] *= 0.2             # position of joints 3
        observation[9] *= 1. / 3.14       # joints angular speed 3
        observation[10] *= 0.2            # position of joints 4
        observation[11] *= 1. / 3.142     # joints angular speed 4
        observation[12] *= 0.2            # position of joints 5
        observation[13] *= 0.2            # joints angular speed 5
        return observation
    
class RewardScalingWrapper(gym.RewardWrapper):
    "Scale the reward by the given factor."
    def __init__(self, env, scaling_factor=1./5):
        super().__init__(env)
        self.scaling_factor = scaling_factor

    def reward(self, reward):
        return reward * self.scaling_factor


config['batch_size'] = 128
config['learning_rate_critic'] = 0.0001
config['learning_rate_actor'] = 0.0001

env = gym.make('BipedalWalker-v3')
env = RewardScalingWrapper(ObservationScalingWrapper(env))

agent_walker = TD3Agent(env, Actor, Critic, config)
agent_walker.train(500000)


agent_walker.logger.episode_rewards = [r * 5 for r in agent_walker.logger.episode_rewards]
plot_training(agent_walker.logger, moving_average=5000, title=": BipedalWalker")


