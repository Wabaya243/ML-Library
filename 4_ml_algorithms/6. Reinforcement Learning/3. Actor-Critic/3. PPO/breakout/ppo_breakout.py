
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import ale_py

from torch.nn import ReLU
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def make_eval_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = AtariWrapper(env)

    env = RecordVideo(
        env,
        video_folder="./videos",
        episode_trigger=lambda ep: ep % 50 == 0,
        name_prefix="breakout_eval"
    )
    env = Monitor(env)

    return env


env = make_atari_env(
    'ALE/Breakout-v5',
    n_envs=8,
    seed=0,
    )

vec_env = VecFrameStack(env, n_stack=8)


eval_env = DummyVecEnv([make_eval_env])
eval_vec_env = VecFrameStack(eval_env, n_stack=8)

# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=600,  # seuil de performance
    verbose=1              # affiche un message quand le seuil est atteint
)

# Policy kwargs : ajouter 2 couches MLP de 256 et 128 neurones
policy_kwargs = dict(
    activation_fn=ReLU,
    normalize_images=True,
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_vec_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    log_path="./logs/eval",
    eval_freq=50_000,              # évaluation tous les n steps
    n_eval_episodes=10,           # moyenne sur n épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)

model = PPO(
    "CnnPolicy",
    vec_env,
    learning_rate=2.5e-4,
    n_steps=128, # Nombre de steps collectés PAR ENV avant update, 128 × n_envs = taille du rollout
    batch_size=256, # Taille des mini-batchs PPO, Doit diviser (n_steps × n_envs)
    n_epochs=4,     # Nombre de passes PPO sur chaque rollout, 3–4 = standard Atari
    gamma=0.99,
    gae_lambda=0.95,  # Bias / variance trade-off pour GAE # 1.0 → fort bruit # <0.9 → biais # 0.95 = standard PPO
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=1,
)


model = PPO.load("Save/best/best_model", env=vec_env)

print(model.policy)


# Exemple : 1M steps
model.learn(
    total_timesteps=10_000_000,
    progress_bar=True,
    log_interval=5,
    callback=eval_callback
    )


# SAUVEGARDE DU MODÈLE

# Sauvegarde du modèle entraîné
model.save("Save/ppo_breakout")


# RECHARGEMENT DU MODÈLE

# Suppression du modèle courant de la mémoire
del model

# Chargement du modèle sauvegardé
model = PPO.load("Save/ppo_breakout", env=vec_env)


# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def play_eval_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = AtariWrapper(env)
    env = Monitor(env)

    return env

play_eval = DummyVecEnv([play_eval_env])
play_eval = VecFrameStack(play_eval, n_stack=8)


# Reset de l'environnement
obs, info = play_eval.reset()

# Boucle pour visualiser le comportement de l'agent
for i in range(1000):
    # Prédiction de l'action (policy déterministe)
    action, _states = model.predict(obs, deterministic=True)

    # Exécution de l'action
    obs, reward, terminated, truncated, info = play_eval.step(action)
    done = terminated or truncated  # combine les deux

    # Reset si épisode terminé
    if done:
        obs, info = play_eval.reset()

    # Rendu visuel
    play_eval.render()