import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)

import numpy as np
import gymnasium as gym
import gymnasium_robotics  #  OBLIGATOIRE pour enregistrer Fetch

from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.evaluation import evaluate_policy
from torch.nn import ReLU

class VideoRecorderCallback(BaseCallback):
    def __init__(self, env, check_freq, video_length=1000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.video_length = video_length

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            video_env = VecVideoRecorder(
                self.env,
                video_folder="Videos",
                record_video_trigger=lambda episode_id: True,
                video_length=self.video_length,
                name_prefix=f"step_{self.num_timesteps}"
            )
            obs = video_env.reset()
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = video_env.step(action)  # done combine terminated+truncated
                if done:
                    obs = video_env.reset()
            video_env.close()
        return True



def make_env():
    env = gym.make("FetchReach-v4")
    return Monitor(env)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)


obs = env.reset()
print(obs.keys())        # Affiche : dict_keys(['observation', 'achieved_goal', 'desired_goal'])
print(obs['observation'].shape)  # Maintenant OK


eval_env = make_vec_env("FetchReach-v4", n_envs=1, seed=88)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

eval_env.training = False
eval_env.norm_reward = False
eval_env.obs_rms = env.obs_rms  # TRÈS IMPORTANT


policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=ReLU
)

# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=-0.05,  # proche de 0 = quasi tous les goals atteints
    verbose=1              # affiche un message quand le seuil est atteint
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


model = SAC(
    policy="MultiInputPolicy",  # Obligatoire pour HER, car l'observation est un dict
    env=env,                    # Environnement compatible HER (dict: observation, achieved_goal, desired_goal)
    replay_buffer_class=HerReplayBuffer,  # On remplace le buffer standard par HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                  # Nombre de goals "alternatifs" générés par épisode pour HER
        goal_selection_strategy="future",  # Stratégie pour choisir les goals alternatifs
        # options: "future" (le plus courant), "final", "episode", "random"
        # future = on prend un état futur du même épisode comme nouveau goal 
        # final = on prend le goal final de l'épisode
        # episode = on prend un goal aléatoire dans l'épisode 
        # random = on prend un goal aléatoire dans le buffer entier
    ),
    learning_rate=3e-4,       # Taux d'apprentissage
    batch_size=256,            # Taille des mini-batches
    tau=0.005,                 # Soft update pour le target network
    gamma=0.99,                # Discount factor
    learning_starts=1000,      # Commence l'apprentissage après x transitions
    n_steps=5,
    ent_coef='auto',
    policy_kwargs=policy_kwargs,
    verbose=1
)

video_callback = VideoRecorderCallback(eval_env, check_freq=50_000)

model.learn(
    total_timesteps=1_000_000,
    progress_bar=True,
    log_interval=10,
    callback=[eval_callback, video_callback]
)

model.save("Save/sac_fetch_her")

del model

model = SAC.load("Save/sac_fetch_her", env=env)

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True
)


print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


play_env = DummyVecEnv([
    lambda: Monitor(gym.make("FetchReach-v2", render_mode="human"))
])


obs = play_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated = play_env.step(action)

    if terminated or truncated:
        obs = play_env.reset()

    play_env.render()

