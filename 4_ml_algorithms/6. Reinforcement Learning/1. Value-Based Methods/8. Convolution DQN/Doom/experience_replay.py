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

class ReplayMemory:
    """
    Mémoire de rejeu (Experience Replay) permettant
    de stocker et d'échantillonner des transitions n-step.
    """

    def __init__(self, n_steps, capacity=10000):
        # Taille maximale de la mémoire
        self.capacity = capacity
        # Générateur de transitions n-step
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        # Buffer circulaire pour stocker les expériences
        self.buffer = deque()

    def sample_batch(self, batch_size):
        """
        Crée un itérateur qui renvoie des mini-batchs aléatoires
        de taille batch_size.
        """
        ofs = 0
        # Copie du buffer pour mélange aléatoire
        vals = list(self.buffer)
        np.random.shuffle(vals)

        # Génération des batchs
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        """
        Récupère un certain nombre de transitions n-step
        et les ajoute dans la mémoire.
        """
        while samples > 0:
            # Récupération de n étapes consécutives
            entry = next(self.n_steps_iter)
            # Ajout dans le buffer
            self.buffer.append(entry)
            samples -= 1

        # On s'assure de ne pas dépasser la capacité maximale
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()
