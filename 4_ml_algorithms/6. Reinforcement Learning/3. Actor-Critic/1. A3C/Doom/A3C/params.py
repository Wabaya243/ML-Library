
# Regroupement de tous les paramètres (que l'on peut modifier pour expérimenter)
class Params():
    def __init__(self):
        self.lr = 7e-4                  # Taux d'apprentissage (learning rate)
        self.gamma = 0.99                 # Facteur de réduction (discount factor)
        self.tau = 1.                     # Paramètre tau pour l'estimation de la valeur
        self.seed = 42                    # Graine aléatoire pour la reproductibilité
        self.num_processes = 10         # Nombre de processus (agents) en parallèle
        self.num_steps = 20               # Nombre d'étapes par mise à jour
        self.max_episode_length = 10000   # Longueur maximale d’un épisode
        self.env_name = 'ALE/Breakout-v5'     
