from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
from gymnasium.core import ObservationWrapper
from vizdoom import DoomGame, ScreenFormat, ScreenResolution, scenarios_path, GameVariable, Button


# ================================================================================
# PARTIE 1 : ENVIRONNEMENT VIZDOOM PERSONNALISÉ
# ================================================================================

class VizDoomEnv(gym.Env):
    """
    Environnement Gymnasium personnalisé pour VizDoom
    
    Caractéristiques :
    - Résolution : 640x480 RGB
    - Actions : 11 combinaisons de mouvements et tirs
    - Reward shaping : encourage l'exploration, les kills, et pénalise le spam
    - Cooldown sur les tirs pour éviter le spam
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 35}

    def __init__(self, cfg_file="deadly_corridor.cfg", render_mode="rgb_array", difficulty=4):
        super().__init__()
        self.render_mode = render_mode

        # ────── Configuration du jeu VizDoom ──────
        cfg_path = os.path.join(scenarios_path, cfg_file)
        self.game = DoomGame()
        self.game.load_config(cfg_path)
        self.game.set_screen_format(ScreenFormat.RGB24)
        
        # Résolution augmentée pour plus de détails
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        
        # Difficulté du jeu (1=facile, 5=cauchemar)
        self.game.set_doom_skill(difficulty)
        
        # pour ne pas afficher les fenetres gain de vitesse
        self.game.set_window_visible(False)
        
        self.game.init()
        
        # ────── Variables de suivi pour le reward shaping ──────
        self.last_hitcount = 0      # Nombre de coups réussis
        self.last_killcount = 0     # Nombre d'ennemis tués
        self.last_y = 0             # Position Y (profondeur/avancement)
        self.last_x = 0             # Position X (latéral)
        
        # ────── Système de cooldown pour les tirs ──────
        self.attack_cooldown = 0
        self.attack_cooldown_max = 10  # Nombre de steps avant tir \"gratuit\"
        
        # ────── Configuration des actions ──────
        # Boutons autorisés pour l'agent
        self.buttons = [
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD,
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.ATTACK,
        ]

        # Indices réels dans VizDoom
        self.button_indices = [
            self.game.get_available_buttons().index(b)
            for b in self.buttons
        ]

        # Index du bouton ATTACK dans l'espace réduit (0-6)
        self.attack_local_idx = self.buttons.index(Button.ATTACK)

        # Index réel du bouton ATTACK dans VizDoom
        self.attack_vizdoom_idx = self.button_indices[self.attack_local_idx]

        # Taille réelle attendue par VizDoom
        self.full_button_count = self.game.get_available_buttons_size()

        # ────── Liste des actions discrètes ──────
        # Chaque ligne représente une combinaison de boutons
        # Format : [FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK, UNUSED]
        self.actions_list = [
            # Ne rien faire
            [0,0,0,0,0,0,0,0],

            # Déplacements purs
            [1,0,0,0,0,0,0,0],  # avancer
            [0,1,0,0,0,0,0,0],  # reculer
            [0,0,1,0,0,0,0,0],  # strafe gauche
            [0,0,0,1,0,0,0,0],  # strafe droite

            # Rotations
            [0,0,0,0,1,0,0,0],  # tourner gauche
            [0,0,0,0,0,1,0,0],  # tourner droite

            # Tir pur
            [0,0,0,0,0,0,1,0],  # tirer seul

            # Déplacement + tir
            [1,0,0,0,0,0,1,0],  # avancer + tirer
            [0,1,0,0,0,0,1,0],  # reculer + tirer
            [0,0,1,0,0,0,1,0],  # strafe gauche + tirer
            [0,0,0,1,0,0,1,0],  # strafe droite + tirer
            [0,0,0,0,1,0,1,0],  # tourner gauche + tirer
            [0,0,0,0,0,1,1,0],  # tourner droite + tirer

            # Combinaison avancée
            [1,0,0,0,1,0,1,0],  # avancer + tourner gauche + tirer
            [1,0,0,0,0,1,1,0],  # avancer + tourner droite + tirer
        ]

        # Espace d'actions : nombre d’actions dynamique
        self.action_space = spaces.Discrete(len(self.actions_list))
       
        # Espace d'observation : images RGB 640x480
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )

    def step(self, action):
        """
        Exécute une action dans l'environnement
        
        Returns:
            observation: Image RGB de l'état actuel
            reward: Récompense calculée avec reward shaping
            terminated: True si l'épisode est terminé
            truncated: False (pas de troncature)
            info: Dictionnaire d'informations additionnelles
        """
        # ────── Conversion de l'action discrète en action VizDoom ──────
        local_action = self.actions_list[action]
        
        doom_action = [0] * self.full_button_count
        for local_idx, value in enumerate(local_action):
            if value == 1:
                doom_idx = self.button_indices[local_idx]
                doom_action[doom_idx] = 1
                

        # ────── Exécution de l'action ──────
        reward = self.game.make_action(doom_action)
        
        # ────── Récupération des variables de jeu ──────
        hitcount = self.game.get_game_variable(GameVariable.HITCOUNT)
        killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
        ammo = self.game.get_game_variable(GameVariable.AMMO1)
        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        
        # ────── REWARD SHAPING ──────
        
        # 1. Pénaliser le tir selon le contexte
        if local_action[self.attack_local_idx] == 1:
            if self.attack_cooldown > 0:
                reward -= 0.05 * self.attack_cooldown  # malus pour spam
            else:
                self.attack_cooldown = self.attack_cooldown_max

        # tir sans munition
        if ammo <= 0 and local_action[self.attack_local_idx] == 1:
            reward -= 0.5

        # Pénalise avancer sans résultat
        if local_action[0] == 1 and hitcount == self.last_hitcount:
            reward -= 1.0

        # Pénalise absence de rotation (très important) force l’exploration visuelle
        if local_action[4] == 0 and local_action[5] == 0:
            reward -= 0.4

        if hitcount > self.last_hitcount and local_action[self.attack_local_idx] == 1:
            reward += 0.8  # tir utile
            

        # recomprense de survie
        reward += 0.001

        # 2. Récompenser les hits et kills
        reward += 8.0 * (hitcount - self.last_hitcount)
        reward += 13.0 * (killcount - self.last_killcount)
        
        # 3. Encourager l'avancement dans le niveau
        # Encourager l'exploration sur X et Y (distance parcourue)
        dx = x - self.last_x
        dy = y - self.last_y
        dist = np.sqrt(dx*dx + dy*dy)
        reward += 0.1 * dist  # petit bonus proportionnel à la distance parcourue

        # maluse pour evier qu'il tourne en rond 
        if abs(dx) < 0.01 and abs(dy) < 0.01 and (local_action[4] == 1 or local_action[5] == 1):
            reward -= 0.1

        # Malus pour rester statique après avoir tué
        if killcount > self.last_killcount and (y - self.last_y) < 0.05:
            reward -= 0.1  # Encourager à bouger après avoir tué

         # ────── Vérification de fin d'épisode ──────
        done = self.game.is_episode_finished()

         # malus si mourir sans kill
        if done and killcount == 0:
            reward -= 15.0

        
        # ────── Mise à jour des variables de suivi ──────
        self.last_hitcount = hitcount
        self.last_killcount = killcount
        self.last_y = y
        self.last_x = x

        # Décrémenter le cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        

        # Observation : image de l'état actuel ou zéros si terminé
        obs = (
            self.game.get_state().screen_buffer
            if not done
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )

        terminated = done
        truncated = False
        info = {}
        
        


        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode
        
        Returns:
            observation: Image RGB de l'état initial
            info: Dictionnaire d'informations additionnelles
        """
        self.game.new_episode()
        
        # Réinitialisation des variables de suivi
        self.last_hitcount = 0
        self.last_killcount = 0
        self.last_y = 0
        self.last_x = 0
        
        obs = self.game.get_state().screen_buffer
        info = {}
        
        return obs, info

    def render(self):
        """
        Retourne l'image RGB de l'état actuel (mode rgb_array)
        """
        if self.render_mode == "rgb_array":
            state = self.game.get_state()
            if state is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return state.screen_buffer

    def close(self):
        """
        Ferme proprement l'environnement VizDoom
        """
        self.game.close()


# ================================================================================
# PARTIE 2 : WRAPPERS DE PRÉTRAITEMENT (style Universe/Atari)
# ================================================================================

def _process_frame42(frame):
    """
    Traite une frame Atari pour la réduire à 42x42 en niveaux de gris
    
    Étapes :
    1. Découpage pour retirer les parties inutiles
    2. Redimensionnement progressif (mipmapping)
    3. Conversion en niveaux de gris
    4. Normalisation entre 0 et 1
    """
    # Découpage de l'image
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(2)
    
    # Normalisation
    frame = frame.astype(np.float32) / 255.0

    return frame


class MyAtariRescale42x42(gym.ObservationWrapper):
    """
    Wrapper Gym pour redimensionner les observations Atari en 42x42
    
    Utilisé pour réduire la dimensionnalité des images tout en conservant
    les informations essentielles du jeu.
    """

    def __init__(self, env=None):
        super(MyAtariRescale42x42, self).__init__(env)
        # Nouvel espace d'observation : 1 canal, 42x42
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def observation(self, observation):
        """Applique le redimensionnement à chaque observation """
        return _process_frame42(observation)


class MyNormalizedEnv(gym.ObservationWrapper):
    """
    Wrapper Gym pour normaliser dynamiquement les observations
    
    Utilise une moyenne et un écart type glissants pour normaliser
    les observations en temps réel. Cela aide à stabiliser l'apprentissage.
    """

    def __init__(self, env=None):
        super(MyNormalizedEnv, self).__init__(env)
        self.state_mean = 0      # Moyenne glissante
        self.state_std = 0       # Écart-type glissant
        self.alpha = 0.9999      # Facteur de décroissance exponentielle
        self.num_steps = 0       # Compteur de steps

    def observation(self, observation):
        """
        Normalise l'observation avec une moyenne/std glissantes
        
        Utilise une mise à jour exponentielle avec correction de biais
        """
        self.num_steps += 1

        # Mise à jour exponentielle de la moyenne et de l'écart-type
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        # Correction du biais dû à l'initialisation à 0
        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        # Normalisation finale
        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)

        # Ajout d'une dimension canal
        return np.expand_dims(ret, axis=0)


class PreprocessImage(ObservationWrapper):
    """
    Wrapper générique pour prétraiter les images
    
    Fonctionnalités :
    - Redimensionnement à une taille personnalisée
    - Conversion en niveaux de gris (optionnel)
    - Recadrage personnalisé (optionnel)
    - Normalisation entre 0 et 1
    - Changement de format (H,W,C) -> (C,H,W)
    """

    def __init__(self, env, height=124, width=124, grayscale=True, crop=lambda img: img):
        super(PreprocessImage, self).__init__(env)

        self.img_size = (height, width)      # Taille finale
        self.grayscale = grayscale           # Conversion en gris ?
        self.crop = crop                     # Fonction de recadrage

        # Nombre de canaux (1 pour gris, 3 pour RGB)
        n_colors = 1 if self.grayscale else 3

        # Nouvel espace d'observation
        self.observation_space = Box(
            0.0, 1.0, [n_colors, height, width]
        )

    def observation(self, img):
        """
        Applique le prétraitement complet à une image
        
        Étapes :
        1. Recadrage personnalisé
        2. Redimensionnement
        3. Conversion en gris (si activée)
        4. Transposition (H,W,C) -> (C,H,W)
        5. Normalisation
        """
        # Recadrage
        img = self.crop(img)

        # Redimensionnement
        img = cv2.resize(img, self.img_size)

        # Conversion en niveaux de gris
        if self.grayscale:
            img = img.mean(-1, keepdims=True)

        # Transposition des dimensions
        img = np.transpose(img, (2, 0, 1))

        # Normalisation
        img = img.astype('float32') / 255.0

        return img


# ================================================================================
# PARTIE 3 : FONCTION DE CRÉATION D'ENVIRONNEMENT
# ================================================================================

def create_atari_env(env_id, video=False, seed=None):
    """
    Crée un environnement Atari avec prétraitement
    
    Args:
        env_id: Identifiant de l'environnement Gym (ex: \"BreakoutNoFrameskip-v4\")
        video: Si True, enregistre des vidéos périodiquement
        seed: Graine aléatoire pour la reproductibilité
    
    Returns:
        Environnement Gym enveloppé avec prétraitement
    
    Note: Pour utiliser VizDoomEnv au lieu d'Atari, remplacez :
          env = gym.make(env_id, render_mode='rgb_array')
          par :
          env = VizDoomEnv()
    """
    # Création de l'environnement de base
    env = VizDoomEnv()
    
    if seed is not None:
        state, _ = env.reset(seed=seed)

    # Enregistrement vidéo optionnel
    if video:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(
            env,
            video_folder='Video',
            episode_trigger=lambda x: 30 == 0,  # Enregistre tous les 20 épisodes
        )

    # Application des wrappers de prétraitement
    env = MyAtariRescale42x42(env)
    env = MyNormalizedEnv(env)

    return env
