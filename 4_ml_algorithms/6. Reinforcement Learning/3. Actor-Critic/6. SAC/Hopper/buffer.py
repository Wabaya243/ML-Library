import random
from collections import deque

class ReplayBuffer:
    """
    Replay Buffer avec support des retours n-step.
    Compatible avec la nouvelle API Gym :
    (observation, action, reward, next_observation, terminated, truncated)
    """
    def __init__(self, capacity, num_steps=1, gamma=0.99):
        # Buffer principal (mémoire d’expérience)
        self.buffer = deque(maxlen=capacity)
        
        # Paramètres du n-step learning
        self.num_steps = num_steps    # Nombre de pas pour le retour n-step
        self.gamma = gamma            # Facteur de discount
        
        # Buffer temporaire pour calculer les n-step returns
        self.n_step_buffer = deque(maxlen=num_steps)
        
    def add(self, transition):
        """
        Ajoute une transition au buffer
        et gère la logique n-step si activée
        """
        # Vérification du format de la transition
        assert len(transition) == 6, \
            "Utiliser la nouvelle API Gym : (s, a, r, s', terminated, truncated)"
        
        # Cas standard : 1-step (DQN classique)
        if self.num_steps == 1:
            observation, action, reward, next_observation, terminated, truncated = transition
            
            # Stockage direct dans le buffer principal
            self.buffer.append(
                (observation, action, reward, next_observation, terminated)
            )
        
        # Cas n-step learning
        else:
            # Ajout au buffer temporaire
            self.n_step_buffer.append(transition)
            
            # Informations de la transition finale
            _, _, _, final_observation, final_termination, final_truncation = transition
            
            # Calcul de la récompense cumulée n-step
            n_step_reward = 0.0
            for _, _, reward, _, _, _ in reversed(self.n_step_buffer):
                n_step_reward = n_step_reward * self.gamma + reward
            
            # État et action initiaux (au premier step)
            observation, action, _, _, _, _ = self.n_step_buffer[0]

            # Si le buffer n-step est plein, on stocke dans le buffer principal
            if len(self.n_step_buffer) == self.num_steps:
                self.buffer.append(
                    (
                        observation,
                        action,
                        n_step_reward,
                        final_observation,
                        final_termination
                    )
                )
            
            # Si l’épisode se termine, on vide le buffer n-step
            if final_termination or final_truncation:
                self.n_step_buffer.clear()
                
    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un batch d’expériences
        """
        observations, actions, rewards, next_observations, terminations = \
            zip(*random.sample(self.buffer, batch_size))
        
        return observations, actions, rewards, next_observations, terminations
        
    def __len__(self):
        """
        Retourne la taille actuelle du buffer
        """
        return len(self.buffer)
