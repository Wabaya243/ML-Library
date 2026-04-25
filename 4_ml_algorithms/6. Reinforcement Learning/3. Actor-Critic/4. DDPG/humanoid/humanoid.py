import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)

import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

from torch.nn import ReLU
import gymnasium as gym


def make_env():
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    return Monitor(env)

env = SubprocVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# l'objet bruit pour DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma= 0.1 * np.ones(n_actions))


obs = env.reset()
print(obs.shape)

r'''
RÈGLE EMPIRIQUE (importante)
Cas	Meilleur choix
Linux	SubprocVecEnv
Windows	DummyVecEnv
Humanoid	DummyVecEnv
Atari pixels	SubprocVecEnv
n_envs ≤ 8	DummyVecEnv
n_envs ≥ 16	SubprocVecEnv
'''

eval_env = make_vec_env("Humanoid-v5", n_envs=1, seed=0)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

eval_env.training = False
eval_env.norm_reward = False
eval_env.obs_rms = env.obs_rms  # TRÈS IMPORTANT

eval_vec_env = VecVideoRecorder(
    eval_env,
    video_folder="Videos",
    record_video_trigger=lambda x: x % 50_000 == 0,
    video_length=1000,
    name_prefix="humanoid_eval"
)

# Humanoid = état physique continu , Pas des frames image donc on empile

# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=4000,  # seuil de performance
    verbose=1              # affiche un message quand le seuil est atteint
)

# Policy kwargs : ajouter 2 couches MLP de 256 et 128 neurones
policy_kwargs = dict(
    net_arch=[512, 256, 128],
    activation_fn=ReLU,
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_vec_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    log_path="./logs/eval",
    eval_freq=20_000,              # évaluation tous les n steps
    n_eval_episodes=10,            # moyenne sur n épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)


obs = env.reset()
print(obs.shape)

model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-3,       # raisonnable pour Humanoid
    buffer_size=1_000_000,    # grand replay buffer
    batch_size=256,            # mini-batch stable
    tau=0.005,                 # soft update du target network
    gamma=0.99,                # discount factor
    learning_starts=5000,      # démarrer l’update après collecte d’expérience
    action_noise=action_noise,
    policy_kwargs=policy_kwargs,
    verbose=1
)



#model = PPO.load("Save/best/best_model", env=vec_env)

print(model.policy)


# Exemple : 1M steps
model.learn(
    total_timesteps=10_000_000,
    progress_bar=True,
    log_interval=5,
    callback=eval_callback
    )


# Sauvegarde du modèle entraîné
model.save("Save/ddpg_humanoid")


# Suppression du modèle courant de la mémoire
del model

# Chargement du modèle sauvegardé
model = DDPG.load("Save/ddpg_humanoid", env=env)


# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def play_eval_env():
    env = gym.make("Humanoid-v5", render_mode="human")
    env = Monitor(env)

    return env

play_eval = DummyVecEnv([play_eval_env])

# Reset de l'environnement
obs = play_eval.reset()

# Boucle pour visualiser le comportement de l'agent
for i in range(1000):
    # Prédiction de l'action (policy déterministe)
    action, _states = model.predict(obs, deterministic=True)

    # Exécution de l'action
    obs, reward, terminated, truncated = play_eval.step(action)
    done = terminated or truncated  # combine les deux

    # Reset si épisode terminé
    if done:
        obs = play_eval.reset()

    # Rendu visuel
    play_eval.render()