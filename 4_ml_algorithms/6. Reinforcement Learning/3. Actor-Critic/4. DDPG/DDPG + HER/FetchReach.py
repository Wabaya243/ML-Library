import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("imageio").setLevel(logging.ERROR)

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # <<< OBLIGATOIRE pour enregistrer Fetch

from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from torch.nn import ReLU

def make_env():
    env = gym.make("FetchReach-v4")
    return Monitor(env)

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)
)


policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=ReLU
)


model = DDPG(
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
    learning_rate=1e-3,       # Taux d'apprentissage
    batch_size=256,            # Taille des mini-batches
    tau=0.005,                 # Soft update pour le target network
    gamma=0.98,                # Discount factor
    learning_starts=1000,      # Commence l'apprentissage après x transitions
    action_noise=action_noise,         # Peut ajouter du bruit (ex: Ornstein-Uhlenbeck) pour exploration
    policy_kwargs=policy_kwargs,
    verbose=1
)


model.learn(
    total_timesteps=1_000_000,
    progress_bar=True,
    log_interval=10
)

model.save("Save/ddpg_fetch_her")

del model

model = DDPG.load("Save/ddpg_fetch_her", env=env)

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

