# Amélioration de l’environnement Gym avec Universe (prétraitement Atari)

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.wrappers import RecordVideo
import ale_py.roms  # parfois suffisant


# Adapté depuis :
# https://github.com/openai/universe-starter-agent


def create_atari_env(env_id, video=False, seed=None):
    # Création de l’environnement Gym
    env = gym.make(env_id, render_mode='rgb_array')
    
    if seed is not None:
        state, _ = env.reset(seed=seed)

    # Enregistrement vidéo si demandé
    if video:
        env = RecordVideo(
            env,
            video_folder='Video', 
            episode_trigger=lambda x: x % 20 == 0,
            )

    # Redimensionnement et normalisation des observations
    env = MyAtariRescale42x42(env)
    env = MyNormalizedEnv(env)

    return env


def _process_frame42(frame):
    # Découpage de l’image pour retirer les parties inutiles
    frame = frame[34:34 + 160, :160]

    # Redimensionnement progressif pour éviter la perte d’information
    # (équivalent à un mipmapping)
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))

    # Conversion en niveaux de gris
    frame = frame.mean(2)

    # Conversion en float32 et normalisation entre 0 et 1
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)

    return frame


class MyAtariRescale42x42(gym.ObservationWrapper):
    """
    Wrapper Gym pour redimensionner les observations Atari
    en images 42x42 normalisées
    """

    def __init__(self, env=None):
        super(MyAtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)


class MyNormalizedEnv(gym.ObservationWrapper):
    """
    Wrapper Gym pour normaliser dynamiquement les observations
    (moyenne et écart-type glissants)
    """

    def __init__(self, env=None):
        super(MyNormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1

        # Mise à jour exponentielle de la moyenne et de l’écart-type
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        # Correction du biais dû à l’initialisation
        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        # Normalisation finale
        ret = (observation - unbiased_mean) / (unbiased_std + 1e-8)

        # Ajout d’une dimension canal
        return np.expand_dims(ret, axis=0)
