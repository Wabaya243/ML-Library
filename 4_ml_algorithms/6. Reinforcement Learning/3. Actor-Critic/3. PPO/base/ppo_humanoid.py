import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
import imageio
import os
import pickle
import matplotlib.animation as animation
from gymnasium.wrappers import RecordVideo



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        # Nombre de dimensions de l'action
        self.action_dim = action_dim

        # Écart-type initial des actions (exploration)
        self.action_std = action_std_init

        # ACTOR (la politique)
        
        # Prend un état en entrée
        # Retourne la moyenne de la distribution des actions
        # Ici, les actions sont continues dans [-1, 1]
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Limite les actions entre -1 et 1 (ex: Humanoid)
        )
        
        
        # CRITIC (valeur de l'état)
        
        # Estime V(s) = "à quel point cet état est bon"
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)  # Valeur scalaire de l'état
        )
        
    # Met à jour l'écart-type des actions (utile pour réduire l'exploration)
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        
    # Forward non utilisé car on sépare actor / critic
    def forward(self):
        raise NotImplementedError
        
    # Choix d'une action
    def act(self, state):
        # Moyenne des actions selon la politique
        action_mean = self.actor(state)

        # Écart-type identique pour chaque dimension de l'action
        action_std = torch.ones_like(action_mean) * self.action_std
        
        # Distribution gaussienne des actions
        dist = Normal(action_mean, action_std)
        
        # Échantillonnage d'une action (exploration)
        action = dist.sample()

        # Log-probabilité de l'action (utilisée par PPO)
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        # Detach pour ne pas stocker le graphe de calcul
        return action.detach(), action_logprob.detach()
        
    # Évaluation pour l'entraînement PPO
    def evaluate(self, state, action):
        # Moyenne des actions avec la politique actuelle
        action_mean = self.actor(state)
        
        # Distribution gaussienne
        action_std = torch.ones_like(action_mean) * self.action_std
        dist = Normal(action_mean, action_std)
        
        # Log-probabilités des actions déjà prises
        action_logprobs = dist.log_prob(action).sum(dim=-1)

        # Entropie → encourage l'exploration
        dist_entropy = dist.entropy().sum(dim=-1)

        # Valeur estimée de l'état V(s)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    

class PPO:
    def __init__(self, state_dim, action_dim, action_std_init, lr, gamma, K_epochs, eps_clip):
        # Facteur de discount (importance du futur)
        self.gamma = gamma

        # Paramètre de clipping PPO
        self.eps_clip = eps_clip

        # Nombre d'itérations d'optimisation par update
        self.K_epochs = K_epochs
        
        # Mémoire de trajectoire
        self.buffer = RolloutBuffer()
        
        # Politique actuelle
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Ancienne politique (stabilité PPO)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Perte MSE pour le critic
        self.MseLoss = nn.MSELoss()
    
    # Sélection d'action
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        # Stockage pour l'apprentissage
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.cpu().numpy().flatten()
    
    
    # Mise à jour PPO
    def update(self):
        # Calcul des retours Monte Carlo
        rewards = []
        discounted_reward = 0

        # Parcours à l'envers pour gérer les épisodes
        for reward, is_terminal in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalisation des récompenses (stabilité)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Récupération des données stockées
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        
        # Optimisation de la politique
        for _ in range(self.K_epochs):
            # Évaluation avec la nouvelle politique
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            
            state_values = torch.squeeze(state_values)

            # Avantage A(s,a) = R - V(s)
            advantages = rewards - state_values.detach()
            
            # Ratio PPO
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Fonctions objectif PPO
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios,
                                1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            
            # Perte totale PPO
            loss = (
                -torch.min(surr1, surr2)                 # Actor
                + 0.5 * self.MseLoss(state_values, rewards)  # Critic
                - 0.01 * dist_entropy                   # Exploration
            )
            
            # Descente de gradient
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Mise à jour de l'ancienne politique
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Vidage du buffer
        self.buffer.clear()
        
        
class RolloutBuffer:
    def __init__(self):
        # Actions prises
        self.actions = []

        # États observés
        self.states = []

        # Log-probabilités des actions
        self.logprobs = []

        # Récompenses reçues
        self.rewards = []

        # Indique si l'épisode est terminé
        self.is_terminals = []
    
    # Réinitialisation du buffer après update
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def train():
    """Train a PPO agent on Humanoid and export data for external visualization."""
    import json
    from datetime import datetime
    
    # HalfCheetah-v4, Walker2d-v4, Humanoid-v4
    env_name = 'HalfCheetah-v4'
    
    train_env = gym.make(env_name, render_mode='rgb_array')
    
    train_env = RecordVideo(
        train_env,
        video_folder="Video",
        episode_trigger= lambda ep: ep % 100 == 0
        )
    
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Hyperparameters
    max_ep_len = 1000              # Durée maximale d’un épisode
    max_training_episodes = 3000   # Nombre total d’épisodes d’entraînement
    update_timestep = 4000         # Nombre de pas avant de mettre à jour la politique default : ( 2 x max_ep_len)
    action_std = 0.6               # Écart-type de la loi normale des actions, Trop petit l’agent explore peu, Trop grand → actions incohérentes
    K_epochs = 10                  # Nombre de fois où on réutilise le même batch
    eps_clip = 0.2                 # La nouvelle politique ne doit pas être trop différente de l’ancienn
    gamma = 0.99                   # Discount factor
    lr = 0.0003                    # Learning rate
    
    # Initialize PPO agent
    ppo = PPO(state_dim, action_dim, action_std, lr, gamma, K_epochs, eps_clip)
  
    # Variables for logging
    time_step = 0
    i_episode = 0
    
    # Record rewards
    ep_rewards = []
    
    # For final episode data collection
    final_episode_states = []
    final_episode_actions = []
    final_episode_rewards = []
    final_episode_infos = []
    
    # Training loop
    while i_episode < max_training_episodes:
        state, _ = train_env.reset()
        current_ep_reward = 0
        
        # For collecting episode data
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_infos = []
        
        for t in range(max_ep_len):
            # Record state
            if i_episode == max_training_episodes - 1:
                episode_states.append(state.tolist())
            
            # Select action
            action = ppo.select_action(state)
            
            # Record action
            if i_episode == max_training_episodes - 1:
                episode_actions.append(action.tolist())
            
            # Take action
            state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Record reward and info
            if i_episode == max_training_episodes - 1:
                episode_rewards.append(float(reward))
                # Extract qpos information if available
                if hasattr(train_env.unwrapped, 'get_body_com'):
                    # Store positions of important body parts if available
                    body_positions = {}
                    for body_part in ['head', 'torso', 'left_foot', 'right_foot', 'left_hand', 'right_hand']:
                        try:
                            pos = train_env.unwrapped.get_body_com(body_part)
                            body_positions[body_part] = pos.tolist()
                        except:
                            pass
                    episode_infos.append(body_positions)
                else:
                    episode_infos.append({})
            
            # Save to PPO buffer
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            # PPO update
            if time_step % update_timestep == 0:
                ppo.update()
            
            if done:
                break
        
        # Store final episode data
        if i_episode == max_training_episodes - 1:
            final_episode_states = episode_states
            final_episode_actions = episode_actions
            final_episode_rewards = episode_rewards
            final_episode_infos = episode_infos
        
        i_episode += 1
        ep_rewards.append(current_ep_reward)
        
        print(f"Episode: {i_episode}, Reward: {current_ep_reward:.2f}")
    
    # Close training environment
    train_env.close()
    
    # Visualize training results
    plt.figure(figsize=(10, 5))
    plt.plot(ep_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Progress - {env_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig('humanoid_training_progress.png')
    plt.show()
    
    # Save model
    torch.save(ppo.policy.state_dict(), f"Save/{env_name}_model.pth")
    
    # Export final episode data to JSON for external visualization
    if final_episode_states:
        final_episode_data = {
            "states": final_episode_states,
            "actions": final_episode_actions,
            "rewards": final_episode_rewards,
            "infos": final_episode_infos,
            "metadata": {
                "environment": env_name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "episodes": max_training_episodes,
                "state_dim": state_dim,
                "action_dim": action_dim
            }
        }
        
        # Save to JSON
        with open('Save/final_episode_data.json', 'w') as f:
            json.dump(final_episode_data, f)
        
        print("\nFinal episode data saved to 'Save/final_episode_data.json'")
        
        # Create data visualizations
        # Reward plot
        plt.figure(figsize=(10, 4))
        plt.plot(final_episode_rewards)
        plt.title('Rewards Over Final Episode')
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig('final_episode_rewards.png')
        plt.show()
        
        # State features
        plt.figure(figsize=(12, 8))
        states_array = np.array(final_episode_states)
        
        # Plot a few key state dimensions
        key_dims = min(8, states_array.shape[1])
        for i in range(key_dims):
            plt.plot(states_array[:, i], label=f'State {i}')
        
        plt.title('Key State Features Over Final Episode')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('final_episode_states.png')
        plt.show()
        
        # Action visualization
        plt.figure(figsize=(12, 8))
        actions_array = np.array(final_episode_actions)
        
        # Plot a few key action dimensions
        key_dims = min(8, actions_array.shape[1])
        for i in range(key_dims):
            plt.plot(actions_array[:, i], label=f'Action {i}')
        
        plt.title('Key Action Values Over Final Episode')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('final_episode_actions.png')
        plt.show()
    
    print("\nTraining completed.")
    print(f"Model saved to '{env_name}_model.pth'")
    print("Visualization files saved.")
    print("\nTo generate a GIF from this data, download the JSON file")
    print("and use a separate script with OpenGL support to visualize it.")
    
    return ep_rewards


train()