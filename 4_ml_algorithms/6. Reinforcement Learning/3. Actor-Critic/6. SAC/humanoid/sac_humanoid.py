
import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)

import os
os.makedirs("Videos", exist_ok=True)
os.makedirs("Save/best", exist_ok=True)


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3 import SAC

from torch.nn import ReLU
import gymnasium as gym

def make_video_env():
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    env = Monitor(env)
    return env

class VideoRecorderCallback(BaseCallback):
    def __init__(self, check_freq, video_length=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.video_length = video_length

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            env = DummyVecEnv([make_video_env])
            env = VecNormalize.load("Save/vecnormalize.pkl", env)
            # env = VecNormalize(env, norm_obs=True, norm_reward=False)
            
            env.training = False
            env.norm_reward = False
            

            video_env = VecVideoRecorder(
                env,
                video_folder="Videos",
                record_video_trigger=lambda step: True,
                video_length=self.video_length,
                name_prefix=f"step_{self.num_timesteps}"
            )

            obs = video_env.reset()
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = video_env.step(action)

            video_env.close()
        return True



vec_env = make_vec_env(
    'Humanoid-v5',
    n_envs=8,
    seed=0,
    )
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

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
eval_env.obs_rms = vec_env.obs_rms  # TRÈS IMPORTANT


# Humanoid = état physique continu , Pas des frames image donc on empile

# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=15_000,  # seuil de performance
    verbose=1              # affiche un message quand le seuil est atteint
)

# Policy kwargs : ajouter 2 couches MLP de 256 et 128 neurones
policy_kwargs = dict(
    net_arch=[512, 256, 128],
    activation_fn=ReLU,
    normalize_images=True,
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    log_path="./logs/eval",
    eval_freq=2_000,              # évaluation tous les n steps
    n_eval_episodes=10,            # moyenne sur n épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)


vec_env.save("Save/vecnormalize.pkl")

obs = vec_env.reset()
print(obs.shape)

sac_kwargs = {
    "policy": "MlpPolicy",       # Réseau fully-connected
    "env": vec_env,
    "learning_rate": 3e-4,       # lr
    "buffer_size": 500_000,      # buffer_capacity
    "learning_starts": 1000,     # learning_starts
    "batch_size": 256,            # batch_size
    "tau": 0.005,                # tau pour soft-update du critic target
    "gamma": 0.99,               # facteur de discount
    "train_freq": 1,             # update critic à chaque step
    "gradient_steps": 1,         # nombre de gradient steps par update
    "ent_coef": "auto",          # "auto" est bien meilleur pour SAC que 0.2 fixe
    "verbose": 1,                # affichage logs
    "device": "cuda"              # ou "cuda" si GPU disponible
}

# Création de l'agent
model = SAC(**sac_kwargs, policy_kwargs=policy_kwargs)


model = SAC.load("Save/best/best_model", env=vec_env)

print(model.policy)

video_callback = VideoRecorderCallback(check_freq=100_000)

# Exemple : 1M steps
model.learn(
    total_timesteps=5_000_000,
    progress_bar=True,
    log_interval=30,
    callback=[eval_callback, video_callback]
    )


# SAUVEGARDE DU MODÈLE

# Sauvegarde du modèle entraîné
model.save("Save/sac_humanoid")



# RECHARGEMENT DU MODÈLE

# Suppression du modèle courant de la mémoire
del model

# Chargement du modèle sauvegardé
model = SAC.load("Save/sac_humanoid", env=vec_env)


# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10,
    deterministic=True
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def play_eval_env():
    env = gym.make("Humanoid-v5", render_mode="human")  
    env = Monitor(env)

    return env

play_eval = DummyVecEnv([play_eval_env]) 
play_eval = VecNormalize.load("Save/vecnormalize.pkl", play_eval)


play_eval.training = False
play_eval.norm_reward = False
play_eval.obs_rms = vec_env.obs_rms


# Reset de l'environnement
obs, _  = play_eval.reset()

# Boucle pour visualiser le comportement de l'agent
for i in range(1000):
    # Prédiction de l'action (policy déterministe)
    action, _states = model.predict(obs, deterministic=True)

    # Exécution de l'action
    obs, reward, terminated, truncated, info = play_eval.step(action)
    done = terminated or truncated  # combine les deux

    # Reset si épisode terminé
    if done:
        obs, _ = play_eval.reset()

    # Rendu visuel
    play_eval.render()