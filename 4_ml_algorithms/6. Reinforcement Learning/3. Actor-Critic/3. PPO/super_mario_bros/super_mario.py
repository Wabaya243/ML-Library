

# RÉATION DE L’ENVIRONNEMENT MARIO (GYM COMPATIBLE)
import gym_super_mario_bros

# Le "1-1" correspond au niveau 1-1 de Super Mario Bros
STAGE_NAME = 'SuperMarioBros-1-1-v0'  # Version standard
# STAGE_NAME = 'SuperMarioBros-1-1-v3'  # Version rectangulaire (optionnelle)

# Création de l’environnement Mario (API Gym classique)
env = gym_super_mario_bros.make(STAGE_NAME)


# ESPACE D’ACTIONS (BOUTONS NES)
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

print("Actions simples :", SIMPLE_MOVEMENT)
print("Actions complexes :", COMPLEX_MOVEMENT)
print("Actions droite uniquement :", RIGHT_ONLY)

 

import torch as th
from torch import nn

# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# Vectorisation (obligatoire pour Stable-Baselines3)
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO


import gym
from gym.wrappers import GrayScaleObservation

import numpy as np
import cv2
from pathlib import Path
import os






# Limite l’espace d’actions aux mouvements simples
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Conversion des observations RGB → niveaux de gris
env = GrayScaleObservation(env, keep_dim=True)

# TEST RAPIDE DE L’ENVIRONNEMENT

done = True

for step in range(5000):
    if done:
        state = env.reset()

    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()

env.close()



class SkipFrame(gym.Wrapper):
    """
    Répète une action sur plusieurs frames consécutives.
    Permet :
    - d’accélérer l’entraînement
    - de réduire la redondance temporelle
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class ResizeEnv(gym.ObservationWrapper):
    """
    Redimensionne les frames en 84x84
    Réduction massive de la complexité visuelle
    """
    def __init__(self, env, size):
        super().__init__(env)
        old_h, old_w, old_c = env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, old_c),
            dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Sécurité : s’assurer d’avoir (H, W, C)
        if frame.ndim == 2:
            frame = frame[:, :, None]

        return frame

    
class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)  # Initialise le wrapper Gym
        self.current_score = 0        # Score actuel de Mario
        self.current_x = 0            # Position horizontale actuelle (x)
        self.current_x_count = 0      # Compteur de frames sans progression
        self.max_x = 0                # Position x maximale atteinte

    def reset(self, **kwargs):
        self.current_score = 0        # Réinitialise le score
        self.current_x = 0            # Réinitialise la position x
        self.current_x_count = 0      # Réinitialise le compteur d’inactivité
        self.max_x = 0                # Réinitialise la meilleure progression
        return self.env.reset(**kwargs)  # Reset de l’environnement Mario

    def step(self, action):
        state, reward, done, info = self.env.step(action)  # Exécute une action dans l’environnement

        # Récompense la progression vers la droite (encourage l’avancement)
        reward += max(0, info['x_pos'] - self.max_x)

        # Vérifie si Mario n’a pas avancé
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1  # Incrémente le compteur si Mario stagne
        else:
            self.current_x_count = 0   # Réinitialise si Mario avance

        # Si Mario atteint le drapeau de fin
        if info["flag_get"]:
            reward += 500              # Gros bonus de récompense
            done = True                # Fin de l’épisode
            print("GOAL")              # Message de réussite

        # Si Mario perd une vie (meurt)
        if info["life"] < 2:
            reward -= 500              # Grosse pénalité
            done = True                # Fin de l’épisode

        self.current_score = info["score"]           # Mise à jour du score
        self.max_x = max(self.max_x, self.current_x) # Met à jour la meilleure progression
        self.current_x = info["x_pos"]                # Met à jour la position x actuelle

        return state, reward / 10., done, info        # Normalise la récompense

    
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
state = env.reset()
print("RGB scale : ",state.shape)

env = GrayScaleObservation(env, keep_dim=True)
state = env.reset()
print("Gray scale:",state.shape)


# Actions personnalisées : gauche, droite, saut, course
MOVEMENT = [['left', 'A'], ['right', 'B'], ['right', 'A', 'B']]

# Création de l’environnement Mario pour l’entraînement
env = gym_super_mario_bros.make(STAGE_NAME)

# Application de l’espace d’actions personnalisé
env = JoypadSpace(env, MOVEMENT)

# Wrapper de récompense personnalisé (progression, mort, victoire)
env = CustomRewardAndDoneEnv(env)

# Répète chaque action sur 4 frames (stabilité + vitesse)
env = SkipFrame(env, skip=4)

# Conversion RGB → niveaux de gris
env = GrayScaleObservation(env, keep_dim=True)

# Redimensionne les frames en 84x84 pour réduire la complexité
env = ResizeEnv(env, size=84)

# Vectorisation (format requis par Stable-Baselines3)
env = DummyVecEnv([lambda: env])

# Empilement de 4 frames pour inclure la dynamique temporelle
env = VecFrameStack(env, 4, channels_order='last')




# Model Param

# Fréquence de sauvegarde du modèle
CHECK_FREQ_NUMB = 10000

# Nombre total de pas d’entraînement
TOTAL_TIMESTEP_NUMB = 5_000_000

# Taux d’apprentissage
LEARNING_RATE = 0.0001

# Facteur GAE (bias / variance)
GAE = 1.0

# Coefficient d’entropie (exploration)
ENT_COEF = 0.01

# Nombre de pas par rollout
N_STEPS = 512

# Facteur de discount (importance du futur)
GAMMA = 0.9

# Taille du batch pour l’optimisation
BATCH_SIZE = 64

# Nombre d’epochs par update
N_EPOCHS = 10


# Test Param

# Nombre d’épisodes pour évaluer le modèle
EPISODE_NUMBERS = 20

# Nombre maximum de pas par épisode de test
MAX_TIMESTEP_TEST = 1000



class MarioNet(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super(MarioNet, self).__init__(observation_space, features_dim)  # Initialisation du BaseFeaturesExtractor

        # Nombre de canaux d’entrée (ex: 4 frames empilées → channels_first)
        n_input_channels = observation_space.shape[0]  # Doit être C, pas H

        # Définition du réseau convolutionnel
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),  # Conv 1 : réduit la résolution
            nn.ReLU(),                                                            # Activation non-linéaire
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),                # Conv 2 : extrait des patterns
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),                # Conv 3 : caractéristiques plus abstraites
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),                # Conv 4 : compression finale
            nn.ReLU(),
            nn.Flatten(),                                                         # Aplatissement pour le réseau dense
        )

        # Calcul automatique de la taille de sortie du CNN
        with th.no_grad():                                                        # Pas de calcul de gradient ici
            sample = th.as_tensor(observation_space.sample()[None]).float()       # Observation factice
            n_flatten = self.cnn(sample).shape[1]                                 # Taille du vecteur aplati

        # Couche linéaire pour produire le vecteur de features final
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),                                   # Projection vers l’espace latent
            nn.ReLU(),                                                             # Activation
            nn.Linear(features_dim , features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))                                 # Passage CNN → dense



policy_kwargs = dict(
    features_extractor_class=MarioNet,            # Utilise notre CNN personnalisé
    features_extractor_kwargs=dict(features_dim=512),  # Taille du vecteur de features
)




save_dir = Path('./model')                         # Dossier de sauvegarde des modèles
save_dir.mkdir(parents=True, exist_ok=True)        # Création si inexistant

reward_log_path = save_dir / 'reward_log.csv'      # Fichier CSV des rewards




class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)                   # Initialisation du callback SB3
        self.check_freq = check_freq                # Fréquence de sauvegarde
        self.save_path = save_path                  # Dossier de sauvegarde

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)  # Sécurité : crée le dossier

    def _on_step(self):
        # Exécuté à chaque appel de step() par SB3
        if self.n_calls % self.check_freq == 0:     # Vérifie si on doit sauvegarder

            model_path = self.save_path / f"best_model_{self.n_calls}"  # Nom du modèle
            self.model.save(model_path)             # Sauvegarde du modèle courant

            total_reward = [0] * EPISODE_NUMBERS    # Liste des rewards cumulées
            total_time = [0] * EPISODE_NUMBERS      # Liste des durées d’épisodes
            best_reward = float("-inf")              # Meilleure reward observée
            best_epoch = self.n_calls                # Epoch associé à la meilleure reward

            # Boucle d’évaluation du modèle
            for i in range(EPISODE_NUMBERS):
                state = env.reset()                  # Reset de l’environnement
                done = False                         # Indicateur de fin d’épisode

                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)  # Prédiction de l’action
                    state, reward, done, info = env.step(action)  # Exécution de l’action

                    total_reward[i] += reward[0]     # Accumule la reward
                    total_time[i] += 1               # Incrémente le temps

                if total_reward[i] > best_reward:    # Met à jour la meilleure performance
                    best_reward = total_reward[i]
                    best_epoch = self.n_calls

            # Affichage des statistiques
            print("Timesteps :", self.n_calls, "/", TOTAL_TIMESTEP_NUMB)
            print("Reward moyenne :", sum(total_reward) / EPISODE_NUMBERS)
            print("Meilleure reward :", best_reward)

            # Écriture des résultats dans un fichier CSV
            with open(reward_log_path, "a") as f:
                print(self.n_calls,
                      sum(total_reward) / EPISODE_NUMBERS,
                      best_reward,
                      sep=",",
                      file=f)

        return True                                  # Continue l’entraînement


    
    
callback = TrainAndLoggingCallback(
    check_freq=CHECK_FREQ_NUMB,                     # Fréquence de sauvegarde
    save_path=save_dir
)

model = PPO(
    'CnnPolicy',                                    # Policy CNN
    env,                                            # Environnement Mario
    verbose=1,                                      # Mode silencieux
    policy_kwargs=policy_kwargs,                    # CNN personnalisé
    learning_rate=LEARNING_RATE,                    # Taux d’apprentissage
    n_steps=N_STEPS,                                # Taille des rollouts
    batch_size=BATCH_SIZE,                          # Taille des batchs
    n_epochs=N_EPOCHS,                              # Epochs PPO
    gamma=GAMMA,                                    # Discount factor
    gae_lambda=GAE,                                 # GAE
    ent_coef=ENT_COEF                               # Exploration
)

model.learn(
    total_timesteps=TOTAL_TIMESTEP_NUMB,            # Nombre total de steps
    callback=callback,                               # Callback custom
    progress_bar=True,
    log_interval=10
)



import pandas as pd

# Lecture du fichier CSV contenant les logs de reward
reward_log = pd.read_csv(
    "reward_log_Standar.csv",
    names=["timesteps", "reward", "best_reward"]  # Noms des colonnes
)

# Utilise la colonne "timesteps" comme index
reward_log.set_index("timesteps", inplace=True)

# Affiche l’évolution de la reward dans le temps
reward_log.plot()


# Recherche du timestep avec la meilleure reward moyenne
best_epoch = reward_log["reward"].idxmax()

# Affiche le meilleur timestep
print("Best epoch :", best_epoch)

# Chemin vers le modèle sauvegardé correspondant
best_model_path = os.path.join(save_dir, f"best_model_{10000}")

# Chargement du meilleur modèle PPO
model = PPO.load(best_model_path)



state = env.reset()        # Reset initial de l’environnement
done = True               # Indique la fin d’un épisode
plays = 0                 # Nombre total d’épisodes joués
wins = 0                  # Nombre de victoires (drapeau atteint)

while plays < 100:
    if done:
        state = env.reset()      # Nouveau départ
        plays += 1               # Incrémente le compteur d’épisodes

    # Prédit l’action à partir de l’état courant
    action, _ = model.predict(state)

    # Exécute l’action dans l’environnement
    state, reward, done, info = env.step(action)

    # Vérifie si Mario a atteint le drapeau
    if done and info[0].get("flag_get", False):
        wins += 1

# Affiche le taux de victoire du modèle
print("Model win rate :", wins, "%")



state = env.reset()        # Reset de l’environnement
done = True               # Indicateur de fin d’épisode
plays = 0                 # Compteur d’épisodes

while plays < 100:
    if done:
        state = env.reset()   # Nouveau départ
        plays += 1

    action, _ = model.predict(state)        # Action prédite par le modèle
    state, reward, done, info = env.step(action)  # Application de l’action

    env.render()  # À activer uniquement en local (pas Colab)
