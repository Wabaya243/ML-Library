"""
gridworld_q_bonus234.py
Q-Learning tabulaire 4x4 (ou taille personnalisée) avec :
- obstacles
- epsilon decay (linear / exponential / adaptive)
- heatmap des Q-values (max over actions)
- politique (flèches) et chemin greedy vers le goal

Exécuter: python gridworld_q_bonus234.py
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from collections import deque

# -----------------------
#   ENVIRONNEMENT GridWorld
# -----------------------
class GridWorld:
    def __init__(
        self,
        size=4,
        start=(0,0),
        goal=None,
        obstacles=None,
        step_penalty=-1,
        obstacle_penalty=-5,
        goal_reward=10
    ):
        """
        Environnement GridWorld personnalisé.

        size : taille de la grille (size x size)
        start : position de départ (x, y)
        goal : position d'arrivée (par défaut en bas à droite)
        obstacles : liste ou set de cases interdites
        step_penalty : récompense négative pour chaque mouvement
        obstacle_penalty : grosse pénalité si l'agent tente d'aller sur un obstacle
        goal_reward : récompense obtenue quand l'agent atteint l'objectif
        """

        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size-1, size-1)
        self.obstacles = set(obstacles) if obstacles else set()
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        self.goal_reward = goal_reward
        self.reset()

    def reset(self):
        """
        Réinitialise l'agent à la position de départ.
        Retourne l'état initial (tuple (x,y)).
        """
        self.pos = tuple(self.start)
        return self.pos

    def step(self, action):
        """
        Exécute une action et renvoie :
        - le nouvel état,
        - la récompense,
        - un booléen done indiquant si l'épisode est terminé.

        Action :
            0 = HAUT
            1 = BAS
            2 = GAUCHE
            3 = DROITE

        Règles :
        - L'agent ne sort jamais de la grille.
        - S'il essaie d'aller sur un obstacle, il reste sur place
          et reçoit obstacle_penalty.
        - Atteindre le but donne goal_reward et termine l'épisode.
        - Sinon, chaque déplacement donne step_penalty.
        """

        x, y = self.pos

        if action == 0:     # HAUT
            nx = max(x - 1, 0); ny = y
        elif action == 1:   # BAS
            nx = min(x + 1, self.size - 1); ny = y
        elif action == 2:   # GAUCHE
            nx = x; ny = max(y - 1, 0)
        elif action == 3:   # DROITE
            nx = x; ny = min(y + 1, self.size - 1)
        else:
            raise ValueError("Action invalide")

        new_pos = (nx, ny)

        # Cas 1 : Collision avec un obstacle
        if new_pos in self.obstacles:
            # On ne bouge pas et on pénalise fortement
            reward = self.obstacle_penalty
            done = False
            return self.pos, reward, done

        # Cas 2 : Déplacement normal
        self.pos = new_pos

        # Cas 3 : Arrivée à l'objectif
        if self.pos == self.goal:
            return self.pos, self.goal_reward, True

        # Cas 4 : Simple pas normal
        return self.pos, self.step_penalty, False

    def state_to_index(self, state):
        """
        Convertit un état (x,y) en un index entier :
        index = x * size + y

        Permet d'utiliser une Q-table 1D pour les états.
        """
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError(f"State must be (x,y), got: {state}")

        x, y = state
        return x * self.size + y

    def index_to_state(self, idx):
        """
        Convertit un index de Q-table en état (x,y).
        Utile pour visualiser la politique.
        """
        x = idx // self.size
        y = idx % self.size
        return (x, y)

# -----------------------
#   MÉTHODES DE DÉCROISSANCE D’EPSILON
# -----------------------

def linear_decay(epsilon_start, epsilon_min, episode, max_episodes):
    """
    Décroissance linéaire :
    Fait décroître epsilon de façon régulière entre epsilon_start et epsilon_min
    tout au long des épisodes.

        epsilon = epsilon_start + slope * episode
        où slope = (epsilon_min - epsilon_start) / max_episodes

    On retourne toujours un epsilon ≥ epsilon_min.
    """
    slope = (epsilon_min - epsilon_start) / max_episodes
    return max(epsilon_min, epsilon_start + slope * episode)


def exponential_decay(epsilon_start, epsilon_min, episode, decay_rate):
    """
    Décroissance exponentielle :
    epsilon = epsilon_start * exp(-decay_rate * episode)

    Plus le decay_rate est grand, plus epsilon diminue vite.

    On retourne toujours un epsilon ≥ epsilon_min.
    """
    return max(epsilon_min, epsilon_start * np.exp(-decay_rate * episode))


class AdaptiveEpsilon:
    """
    Décroissance adaptative :
    Le système surveille la progression de l'apprentissage via une moyenne glissante
    des récompenses des derniers épisodes.

    - Si la performance augmente clairement → on accélère la décroissance de epsilon
      (moins d’exploration, on exploite plus).
    - Si la performance stagne → on ralentit la décroissance
      (plus d’exploration pour chercher une meilleure stratégie).

    Paramètres :
        start : valeur initiale de epsilon
        min_eps : epsilon minimal
        window : taille de la fenêtre pour la moyenne glissante des récompenses
        improve_threshold : amélioration minimale du score moyen considérée comme un progrès
        decay_fast : coefficient de décroissance rapide
        decay_slow : coefficient de décroissance lente

    Méthode :
        update(reward) : met à jour epsilon à la fin de chaque épisode
    """

    def __init__(self, start, min_eps, window=50, improve_threshold=0.1,
                 decay_fast=0.995, decay_slow=0.999):

        self.epsilon = start
        self.min_eps = min_eps

        self.window = window
        self.recent_rewards = deque(maxlen=window)

        self.prev_avg = None
        self.improve_threshold = improve_threshold
        self.decay_fast = decay_fast
        self.decay_slow = decay_slow


    def update(self, episode_reward):
        """
        Met à jour epsilon après chaque épisode en analysant :
         - la performance actuelle
         - l'amélioration de la moyenne glissante

        Retourne la nouvelle valeur de epsilon.
        """

        # Ajouter la récompense actuelle à la fenêtre
        self.recent_rewards.append(episode_reward)

        # Pas encore assez de données → décroissance lente
        if len(self.recent_rewards) < self.window:
            self.epsilon = max(self.min_eps, self.epsilon * self.decay_slow)
            return self.epsilon

        # Calcul de la récompense moyenne
        avg = np.mean(self.recent_rewards)

        if self.prev_avg is None:
            # Première estimation → décroissance lente
            self.prev_avg = avg
            self.epsilon = max(self.min_eps, self.epsilon * self.decay_slow)
            return self.epsilon

        # Si la performance s'améliore significativement
        if (avg - self.prev_avg) > self.improve_threshold:
            # On accélère la décroissance (moins d’exploration)
            self.epsilon = max(self.min_eps, self.epsilon * self.decay_fast)
        else:
            # Peu ou pas d’amélioration → on explore davantage
            self.epsilon = max(self.min_eps, self.epsilon * self.decay_slow)

        self.prev_avg = avg
        return self.epsilon

# -----------------------
#   TRAINING FUNCTION
# -----------------------
def train_q_learning(env,
                     alpha=0.1,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_min=0.05,
                     episodes=1000,
                     max_steps_per_episode=100,
                     decay_type='exponential',
                     linear_end_episode=None,
                     exp_decay_rate=0.005,
                     adaptive_params=None,
                     seed=0):

    # Fixer la graine pour rendre les résultats reproductibles
    random.seed(seed)
    np.random.seed(seed)

    # Nombre total d'états et d'actions
    n_states = env.size * env.size
    n_actions = 4  # HAUT, BAS, GAUCHE, DROITE

    # Q-table initialisée à zéro
    Q = np.zeros((n_states, n_actions))

    # epsilon initial
    eps = epsilon_start

    # Dans le cas du decay adaptatif, création d’un objet AdaptiveEpsilon
    adaptive = None
    if decay_type == 'adaptive':
        params = adaptive_params or {}
        adaptive = AdaptiveEpsilon(start=epsilon_start,
                                   min_eps=epsilon_min,
                                   **params)

    rewards_history = []

    # BOUCLE PRINCIPALE D’APPRENTISSAGE Q-LEARNING
    for ep in range(episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        done = False
        total_reward = 0

        # Boucle interne : interactions agent <-> environnement
        for step in range(max_steps_per_episode):

            # -----------------------
            #   1. POLITIQUE EPSILON-GREEDY
            # -----------------------
            if random.random() < eps:
                # Exploration : choisir une action aléatoire
                action = random.randint(0, n_actions - 1)
            else:
                # Exploitation : choisir la meilleure action selon Q-table
                action = int(np.argmax(Q[state_idx]))

            # -----------------------
            #   2. Exécuter l’action
            # -----------------------
            new_state, reward, done = env.step(action)
            new_state_idx = env.state_to_index(new_state)

            # -----------------------
            #   3. Mise à jour Q-learning
            # -----------------------
            # TD target : r + γ max(Q(s'))
            td_target = reward + gamma * np.max(Q[new_state_idx])

            # TD error : [target - valeur actuelle]
            td_error = td_target - Q[state_idx][action]

            # Mise à jour : Q(s,a) ← Q(s,a) + α * TD_error
            Q[state_idx][action] += alpha * td_error

            # Avancer dans l’état suivant
            state_idx = new_state_idx
            total_reward += reward

            if done:
                break

        # -----------------------
        #   4. Mise à jour d’epsilon (décroissance)
        # -----------------------
        if decay_type == 'linear':
            # Décroissance linéaire sur les épisodes
            end = linear_end_episode if linear_end_episode is not None else episodes
            eps = linear_decay(epsilon_start, epsilon_min, ep, end)

        elif decay_type == 'exponential':
            # Décroissance exponentielle
            eps = exponential_decay(epsilon_start, epsilon_min, ep, exp_decay_rate)

        elif decay_type == 'adaptive':
            # Décroissance adaptative basée sur les récompenses moyennes
            eps = adaptive.update(total_reward)

        else:
            # Petite décroissance multiplicative par défaut
            eps = max(epsilon_min, eps * 0.995)

        rewards_history.append(total_reward)

    return Q, rewards_history

# -----------------------
#   VISUALISATIONS
# -----------------------
def plot_heatmap_maxQ(Q, env, fname='heatmap_maxQ.png', show=False):
    """Affiche une heatmap (carte thermique) des valeurs max Q(s,a) pour chaque état.
       Les obstacles sont masqués (affichés différemment)."""

    # On calcule max_a Q(s,a) pour chaque état, puis on remet en forme en grille NxN
    maxQ = np.max(Q, axis=1).reshape((env.size, env.size))

    # Création d'un masque booléen pour repérer les obstacles
    mask = np.zeros_like(maxQ, dtype=bool)
    for (ox, oy) in env.obstacles:
        mask[ox, oy] = True

    # On remplace la valeur des obstacles par NaN => matplotlib les affichera différemment
    maxQ_masked = np.where(mask, np.nan, maxQ)

    # Création de la figure
    plt.figure(figsize=(6,5))

    # imshow affiche la grille de valeurs maxQ
    im = plt.imshow(maxQ_masked, origin='upper', interpolation='nearest')

    # Barre de couleurs indiquant l'échelle des valeurs Q
    plt.colorbar(im, label='max_a Q(s,a)')

    plt.title('Heatmap : valeurs max Q pour chaque état (obstacles masqués)')

    # On place S (start) et G (goal)
    sx, sy = env.start
    gx, gy = env.goal
    plt.text(sy, sx, 'S', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')
    plt.text(gy, gx, 'G', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')

    # Dessiner les obstacles en noir semi-transparent
    for ox, oy in env.obstacles:
        plt.gca().add_patch(
            plt.Rectangle((oy-0.5, ox-0.5), 1, 1, color='black', alpha=0.6)
        )

    # Pour que (0,0) soit en haut à gauche, comme dans la grille
    plt.gca().invert_yaxis()

    # Sauvegarde de l'image
    plt.savefig(fname, bbox_inches='tight', dpi=150)

    if show:
        plt.show()

    plt.close()


def plot_policy_arrows(Q, env, fname='policy_arrows.png', show=False):
    """Affiche la politique gloutonne (greedy policy) avec des flèches pour chaque état.
       Les obstacles sont masqués (pas de flèche)."""

    n_states = env.size * env.size

    # Pour chaque cellule, on récupère l'action argmax_a Q(s,a)
    best_actions = np.argmax(Q, axis=1).reshape((env.size, env.size))

    # Dictionnaire action -> vecteur flèche
    # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    action_to_vec = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}

    # Grilles U (déplacement horizontal) et V (vertical)
    U = np.zeros((env.size, env.size))
    V = np.zeros((env.size, env.size))

    for x in range(env.size):
        for y in range(env.size):
            # Si obstacle → pas de flèche
            if (x,y) in env.obstacles:
                U[x,y] = 0
                V[x,y] = 0
                continue

            # Action choisie dans cet état
            a = int(best_actions[x,y])
            dx, dy = action_to_vec[a]

            # quiver utilise (U horizontal, V vertical), inversé par rapport à notre grille
            U[x,y] = dy
            V[x,y] = -dx

    # Affichage
    plt.figure(figsize=(6,6))
    plt.title('Politique greedy (flèches)')

    # Quiver dessine des flèches dans la grille
    plt.quiver(np.arange(env.size), np.arange(env.size), U, V,
               pivot='middle', scale=1, scale_units='xy')

    # Obstacles en carré noir
    for ox, oy in env.obstacles:
        plt.gca().add_patch(
            plt.Rectangle((oy-0.5, ox-0.5), 1, 1, color='black', alpha=0.6)
        )

    # Start et Goal
    sx, sy = env.start
    gx, gy = env.goal
    plt.text(sy, sx, 'S', ha='center', va='center',
             fontsize=14, color='red', fontweight='bold')
    plt.text(gy, gx, 'G', ha='center', va='center',
             fontsize=14, color='green', fontweight='bold')

    # Correction orientation visuelle
    plt.xlim(-0.5, env.size-0.5)
    plt.ylim(-0.5, env.size-0.5)
    plt.gca().invert_yaxis()

    plt.savefig(fname, bbox_inches='tight', dpi=150)

    if show:
        plt.show()

    plt.close()


def get_greedy_path(Q, env, max_steps=100):
    """Suit la politique greedy depuis l'état initial jusqu'au but.
       Renvoie la liste des états visités.
       Si une boucle est détectée (l'agent revient sur un état déjà visité), on arrête."""

    path = []

    # On crée une copie de l'environnement pour ne pas modifier l'original
    env_copy = GridWorld(
        size=env.size, start=env.start, goal=env.goal,
        obstacles=env.obstacles,
        step_penalty=env.step_penalty,
        obstacle_penalty=env.obstacle_penalty,
        goal_reward=env.goal_reward
    )

    state = env_copy.reset()
    path.append(state)

    # Pour repérer les boucles
    visited = set([state])

    for _ in range(max_steps):

        idx = env_copy.state_to_index(state)

        # Action greedy dans cet état
        action = int(np.argmax(Q[idx]))

        # On applique l'action dans l'environnement
        new_state, _, done = env_copy.step(action)

        path.append(new_state)

        # Si but atteint → fin
        if done:
            return path

        # Détection de boucle : si déjà visité, on arrête
        if new_state in visited:
            return path

        visited.add(new_state)
        state = new_state

    return path


def plot_path(path, env, fname='path.png', show=False):
    """Trace graphiquement le chemin greedy trouvé par l'agent."""

    plt.figure(figsize=(4,4))
    plt.title('Chemin greedy (suivi de la politique)')

    # Extraire les coordonnées
    xs = [s[0] for s in path]
    ys = [s[1] for s in path]

    # tracer les points parcourus
    plt.plot(ys, xs, marker='o')

    # Obstacles en noir
    for ox, oy in env.obstacles:
        plt.gca().add_patch(
            plt.Rectangle((oy-0.5, ox-0.5), 1, 1, color='black', alpha=0.6)
        )

    # Start et Goal
    sx, sy = env.start
    gx, gy = env.goal
    plt.text(sy, sx, 'S', ha='center', va='center',
             fontsize=12, color='red', fontweight='bold')
    plt.text(gy, gx, 'G', ha='center', va='center',
             fontsize=12, color='green', fontweight='bold')

    # Correction orientation
    plt.gca().invert_yaxis()
    plt.xlim(-0.5, env.size-0.5)
    plt.ylim(-0.5, env.size-0.5)

    plt.savefig(fname, bbox_inches='tight', dpi=150)

    if show:
        plt.show()

    plt.close()


# -----------------------
#   MAIN / EXEMPLE USAGE
# -----------------------
def main():
    # ---------------------------------------------------------------------
    # 1) Lecture des arguments passés via la ligne de commande
    # Exemple : python main.py --size 8 --episodes 5000 --decay linear
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=6,
                        help='Taille de la grille (NxN)')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Nombre total d\'épisodes pour l\'apprentissage')
    parser.add_argument('--decay', type=str, default='exponential',
                        choices=['linear','exponential','adaptive'],
                        help='Stratégie de réduction du epsilon (exploration)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire pour reproductibilité')
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # 2) Définition d’un ensemble d’obstacles dans la grille
    #    Chaque obstacle est défini par (ligne, colonne)
    # ---------------------------------------------------------------------
    obstacles = {(1,1), (1,2), (2,3), (3,1)}

    # Création de l'environnement GridWorld
    env = GridWorld(
        size=args.size,
        start=(0,0),
        goal=(args.size-1, args.size-1),
        obstacles=obstacles
    )

    # ---------------------------------------------------------------------
    # 3) Hyperparamètres de Q-learning
    # ---------------------------------------------------------------------
    alpha = 0.1             # Learning rate
    gamma = 0.99            # Discount factor
    epsilon_start = 1.0     # Exploration au début
    epsilon_min = 0.05      # Exploration minimale
    episodes = args.episodes

    # ---------------------------------------------------------------------
    # Paramètres spécifiques selon le type de decay choisi
    # ---------------------------------------------------------------------
    decay_type = args.decay
    exp_decay_rate = 0.002            # utile pour exponential
    linear_end_episode = episodes     # epsilon atteint min à la fin
    adaptive_params = {
        'window':50,
        'improve_threshold':0.1,
        'decay_fast':0.992,
        'decay_slow':0.998
    }

    # Info affichée au début
    print(f"Training Q-learning: size={env.size}, obstacles={env.obstacles}, "
          f"decay={decay_type}, episodes={episodes}")

    # ---------------------------------------------------------------------
    # 4) Entraînement Q-Learning
    # ---------------------------------------------------------------------
    Q, rewards = train_q_learning(
        env,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        episodes=episodes,
        max_steps_per_episode=200,
        decay_type=decay_type,
        linear_end_episode=linear_end_episode,
        exp_decay_rate=exp_decay_rate,
        adaptive_params=adaptive_params,
        seed=args.seed
    )

    # ---------------------------------------------------------------------
    # 5) Visualisation : Heatmap des max-Q
    # ---------------------------------------------------------------------
    plot_heatmap_maxQ(Q, env, fname='images/heatmap_maxQ.png', show=True)

    # ---------------------------------------------------------------------
    # 6) Visualisation : Politique greedy (flèches)
    # ---------------------------------------------------------------------
    plot_policy_arrows(Q, env, fname='images/policy_arrows.png', show=True)

    # ---------------------------------------------------------------------
    # 7) Calcul du chemin greedy (à partir de la politique apprise)
    # ---------------------------------------------------------------------
    path = get_greedy_path(Q, env, max_steps=200)
    print("Chemin greedy trouvé (états):", path)

    # Visualisation du chemin sur la grille
    plot_path(path, env, fname='images/path.png', show=True)

    # ---------------------------------------------------------------------
    # 8) Visualisation : évolution des récompenses
    # ---------------------------------------------------------------------
    plt.figure(figsize=(8,3))
    plt.plot(rewards)
    plt.title('Récompense totale par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.grid(True)
    plt.savefig('images/rewards_history.png', dpi=150, bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------------
# Point d’entrée du script
# -------------------------------------------------------------------------
if __name__ == '__main__':
    main()

