# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

# Making the AI progress on several (n_step) steps
class NStepProgress:
    """
    Cette classe permet de générer des transitions n-step
    à partir d'un environnement et d'une politique (IA).
    """

    def __init__(self, env, ai, n_step):
        # Politique / réseau de neurones qui choisit les actions
        self.ai = ai
        # Liste pour stocker les récompenses totales par épisode
        self.rewards = []
        # Environnement (ex: OpenAI Gym)
        self.env = env
        # Nombre d'étapes pour le n-step learning
        self.n_step = n_step

    def __iter__(self):
        # Réinitialisation de l'environnement
        state, info = self.env.reset()
        
        # Historique des transitions
        history = deque()
        # Récompense cumulée sur un épisode
        reward = 0.0

        while True:
            # state est un array (C,H,W)
            action = self.ai(state)[0][0]  
            
            # step retourne (obs, reward, terminated, truncated, info)
            next_state, r, terminated, truncated, info = self.env.step(action)
            is_done = terminated or truncated  # Gymnasium sépare terminé et tronqué

            # Accumulation de la récompense
            reward += r

            # Sauvegarde de la transition courante
            history.append(
                Step(state=state, action=action, reward=r, done=is_done)
            )

            # On limite la taille de l'historique à n_step + 1
            while len(history) > self.n_step + 1:
                history.popleft()

            # Si on a assez d'étapes, on renvoie une séquence n-step
            if len(history) == self.n_step + 1:
                yield tuple(history)

            # Mise à jour de l'état courant
            state = next_state

            # Si l'épisode est terminé
            if is_done:
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
      
                self.rewards.append(reward)
                reward = 0.0
                state, info = self.env.reset()  # reset -> (obs, info)
                history.clear()

    def rewards_steps(self):
        """
        Retourne les récompenses cumulées par épisode
        et réinitialise la liste.
        """
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


# ================================
# Experience Replay
# ================================

class PERMemory:
    def __init__(self, n_steps, capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # contrôle l’importance de la priorisation
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)
            self.buffer.append(entry)
            # priorité initiale : max existante ou 1 si vide
            max_prio = max(self.priorities, default=1.0)
            self.priorities.append(max_prio)
            samples -= 1

    def sample_batch(self, batch_size):
        N = len(self.buffer)
        if N == 0:
            return []

        # Calcul des probabilités normalisées
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Tirage aléatoire avec probabilités
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalisation

        return batch, indices, weights

    def update_priorities(self, indices, td_errors, eps=1e-6):
        for idx, td in zip(indices, td_errors):
            if hasattr(td, "item"):
                td = td.item()
            self.priorities[idx] = abs(td) + eps

