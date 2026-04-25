from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import gymnasium as gym

from torch.nn import ReLU
from envs import create_atari_env

# creation de l'environement vectorise
def make_env(video=False):
    def _init():
        env = create_atari_env(env_id=None, video=False)  # ton VizDoom
        if video:
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(
                env,
                video_folder="Video",
                episode_trigger=lambda ep: ep % 50 == 0,
                name_prefix="PPO_doom"
            )
        env = Monitor(env)
        return env
    return _init


vec_env = SubprocVecEnv([make_env(video=False) for _ in range(8)])

# Stack de 8 frames pour donner de la temporalité au réseau
vec_env = VecFrameStack(vec_env, n_stack=8, channels_order="first")

eval_vec_env = DummyVecEnv([make_env(video=True)])
eval_vec_env = VecFrameStack(eval_vec_env, n_stack=8, channels_order="first")


# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=500,  # seuil de performance
    verbose=1              # affiche un message quand le seuil est atteint
)

# Policy kwargs : ajouter 2 couches MLP de 256 et 128 neurones
policy_kwargs = dict(
    activation_fn = ReLU,
    normalize_images=True     # mes frames sont déjà [0,1]
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_vec_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    log_path="./logs/eval",
    eval_freq=24000,              # évaluation tous les 30 000 steps
    n_eval_episodes=50,            # moyenne sur 50 épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)

obs = vec_env.reset()
print(obs.shape)  

model = PPO(
    "CnnPolicy",
    vec_env,
    learning_rate=0.0001,
    batch_size=512,
    n_epochs=10,
    n_steps=256,
    n_epochs=4,
    policy_kwargs=policy_kwargs,
    verbose=1
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
model.save("Save/PPO_doom")


# RECHARGEMENT DU MODÈLE

# Suppression du modèle courant de la mémoire
del model

# Chargement du modèle sauvegardé
model = PPO.load("Save/PPO_doom", env=vec_env)


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

play_env = SubprocVecEnv([lambda : make_play_env()])
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

