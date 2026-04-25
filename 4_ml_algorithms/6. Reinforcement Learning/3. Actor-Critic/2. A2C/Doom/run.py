from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
import gymnasium as gym

from torch.nn import ReLU
import vizdoom
from envs import create_atari_env

# creation de l'environement vectorise
def make_env(video=False):
    env = create_atari_env(env_id=None, video=False)  # ton VizDoom
    if video:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(
            env,
            video_folder="Video",
            episode_trigger=lambda ep: ep % 100 == 0,
            name_prefix="A2C_doom"
        )
    env = Monitor(env)
    return env


vec_env = SubprocVecEnv([lambda: make_env(video=False) for _ in range(8)])

# Stack de 8 frames pour donner de la temporalité au réseau
vec_env = VecFrameStack(vec_env, n_stack=8)

eval_vec_env = SubprocVecEnv([lambda: make_env(video=True)])
eval_vec_env = VecFrameStack(eval_vec_env, n_stack=8)


# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=500,  # seuil de performance
    verbose=1              # affiche un message quand le seuil est atteint
)

# Policy kwargs : ajouter 2 couches MLP de 256 et 128 neurones
policy_kwargs = dict(
    net_arch=[256, 128],       # MLP après le CNN
    activation_fn = ReLU,
    normalize_images=False     # mes frames sont déjà [0,1]
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_vec_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    log_path="./logs/eval",
    eval_freq=50_000,              # évaluation tous les 30 000 steps
    n_eval_episodes=50,            # moyenne sur 50 épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)



model = A2C(
    "CnnPolicy",
    vec_env,
    learning_rate=0.0007,     # Learning rate (default: 7e-4)
    n_steps=100,               # Number of steps to run for each environment before updating (default: 5)
    gamma=0.99,              # Discount factor (default: 0.99)
    gae_lambda=1.0,          # Facteur de compromis entre biais et variance pour l'estimateur d'avantage généralis (default: 1.0)
    ent_coef=0.0,            #  Coefficient d'entropie pour le calcul des pertes (default: 0.0)
    vf_coef=0.5,             # Coefficient de fonction de valeur pour le calcul des pertes (default: 0.5)
    max_grad_norm=0.5,       # La valeur maximale pour l'écrêtage du gradient (default: 0.5)
    rms_prop_eps=1e-5,       # Epsilon value for RMSprop optimizer (default: 1e-5)
    use_rms_prop=True,       # Use RMSprop optimizer (default: True)
    use_sde=False,           # S'il faut utiliser l'exploration généralisée dépendante de l'État (gSDE) au lieu de l'exploration du bruit d'action (default: False)
    policy_kwargs=policy_kwargs, 
    verbose=1                # Verbosity level (0 = none, 1 = basic info, 2 = detailed)
)


print(model.policy)

# ENTRAÎNEMENT

# Exemple : 1M steps
model.learn(
    total_timesteps=1_000_000,
    progress_bar=True,
    log_interval=5,
    callback=eval_callback
    )


# SAUVEGARDE DU MODÈLE

# Sauvegarde du modèle entraîné
model.save("Save/A2c_doom")


# RECHARGEMENT DU MODÈLE

# Suppression du modèle courant de la mémoire
del model

# Chargement du modèle sauvegardé
model = A2C.load("Save/A2c_doom", env=vec_env)


# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# VISUALISATION (PLAY)

def make_play_env():
    env = create_atari_env(env_id=None, video=False)
    return env

play_env = SubprocVecEnv([lambda : make_play_env])
play_env = VecFrameStack(play_env, n_stack=8)
play_env = Monitor(play_env, "./logs/play")

# Reset de l'environnement
obs, info = play_env.reset()

# Boucle pour visualiser le comportement de l'agent
for i in range(1000):
    # Prédiction de l'action (policy déterministe)
    action, _states = model.predict(obs, deterministic=True)

    # Exécution de l'action
    obs, reward, terminated, truncated, info = play_env.step(action)
    done = terminated or truncated  # combine les deux

    # Reset si épisode terminé
    if done:
        obs, info = play_env.reset()

    # Rendu visuel
    play_env.render()

