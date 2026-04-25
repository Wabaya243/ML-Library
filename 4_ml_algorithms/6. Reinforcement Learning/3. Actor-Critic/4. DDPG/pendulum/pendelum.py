import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "ffmpeg"

import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

# on choisis un environement avec action continu

def make_env():
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    return Monitor(env)

env = SubprocVecEnv([make_env])

env = VecVideoRecorder(
    env,
    video_folder="Video/train",
    record_video_trigger=lambda x: False,  # pas de vidéo train
    video_length=0
)

eval_env = SubprocVecEnv([lambda: gym.make("Pendulum-v1", render_mode="rgb_array")])

eval_env = VecVideoRecorder(
    eval_env,
    video_folder="Video",
    record_video_trigger=lambda x: True,
    video_length=1000,
    name_prefix="eval"
)

# l'objet bruit pour DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))


# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=-200,  # seuil de performance
    verbose=1              # affiche un message quand le seuil est atteint
)

# Policy kwargs : ajouter 2 couches MLP de 256 et 128 neurones
policy_kwargs = dict(
    net_arch=[512, 256, 128],
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    eval_freq=20_000,              # évaluation tous les n steps
    n_eval_episodes=10,            # moyenne sur n épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)


model = DDPG('MlpPolicy',
             env,
             learning_rate=0.001,
             buffer_size=100_000,
             learning_starts=300,
             batch_size=256,
             tau=1.,
             gamma=0.99,
             n_steps=20,
             action_noise=action_noise,
             policy_kwargs=policy_kwargs,
             verbose=1)

print(model.policy)


model.learn(total_timesteps=1_000_000, log_interval=10, callback=eval_callback, progress_bar=True)

model.save('Save/ddpg_pendelum')

vec_env = model.get_env()


model = DDPG.load("Save/ddpg_pendelum")

# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

