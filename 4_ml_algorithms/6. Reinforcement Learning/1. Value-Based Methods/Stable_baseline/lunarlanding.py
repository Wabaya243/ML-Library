
# Algorithme DQN de Stable-Baselines3
from stable_baselines3 import DQN

# Wrapper pour logger les rewards, longueurs d'épisodes, etc.
from stable_baselines3.common.monitor import Monitor

# Fonction utilitaire pour évaluer un agent entraîné
from stable_baselines3.common.evaluation import evaluate_policy

# Callbacks pour évaluer périodiquement et arrêter l'entraînement si un seuil est atteint
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# PyTorch (utilisé indirectement par SB3)
import torch

# Gymnasium : environnements de RL
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ENVIRONNEMENTS

# Environnement d'entraînement LunarLander
env = gym.make("LunarLander-v3")

# Wrapper Monitor pour enregistrer les rewards et stats d'entraînement
env = Monitor(env, "./logs/train")

# Environnement d'évaluation (séparé de celui d'entraînement)
eval_env = gym.make("LunarLander-v3", render_mode='rgb_array')

eval_env = RecordVideo(
    eval_env, 
    video_folder= "Video",
    episode_trigger= lambda ep: ep % 100 == 0, # une video tous les 100 episode
    name_prefix='dqn_lunar'
    )

# Wrapper Monitor pour logger les performances d'évaluation
eval_env = Monitor(eval_env, "./logs/eval")


# EARLY STOPPING

# Callback qui stoppe l'entraînement quand la reward moyenne atteint le seuil
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=200,  # seuil de performance (résolution de LunarLander)
    verbose=1              # affiche un message quand le seuil est atteint
)


# Callback d'évaluation périodique
eval_callback = EvalCallback(
    eval_env,                      # environnement d'évaluation
    callback_on_new_best=stop_callback,  # on vérifie le seuil seulement si nouveau record
    best_model_save_path="./Save/best",  # sauvegarde auto du best_model
    eval_freq=40_000,              # évaluation tous les 30 000 steps
    n_eval_episodes=50,            # moyenne sur 50 épisodes (plus stable)
    deterministic=True,            # pas d'exploration pendant l'évaluation
    verbose=1
)


# ARCHITECTURE DU RÉSEAU

# Architecture du réseau de neurones pour le Q-network
policy_kwargs = dict(
    net_arch=[ 256, 128, 64]
)


# PARAMÈTRES D'ENTRAÎNEMENT

# Nombre total d'interactions agent-environnement
TRAIN_STEPS = 1_000_000

# Learning rate initial (début de l'entraînement)
alpha_0 = 1e-4

# Learning rate final (fin de l'entraînement)
alpha_end = 1e-9


# Fonction de schedule du learning rate
# process_remaining ∈ [1 → 0] pendant l'entraînement
def learning_rate_f(process_remaining):
    initial = alpha_0                  # learning rate initial
    final = alpha_end                  # learning rate final
    interval = initial - final         # amplitude de décroissance
    return final + interval * process_remaining


# Paramètres DQN
params = {
    'gamma': 0.99,                     # facteur de discount (long horizon)
    'batch_size': 100,                 # taille des minibatchs du replay buffer
    # 'train_freq': 500,               # fréquence d'entraînement (optionnel)
    'target_update_interval': 10_000,  # mise à jour du réseau cible
    'learning_rate': learning_rate_f,  # learning rate dynamique
    'learning_starts': 10_000,         # steps avant de commencer l'apprentissage
    'exploration_fraction': 0.1,       # fraction du training dédiée à l'exploration
    'exploration_initial_eps': 1.0,    # epsilon initial (exploration max)
    'exploration_final_eps': 0.05,     # epsilon final
    'tau': 1,                          # soft update (1 = hard update)
    'buffer_size': 100_000,            # taille du replay buffer
    'verbose': 1,                      # niveau de logs
}


# CRÉATION DU MODÈLE


# Initialisation du modèle DQN
model = DQN(
    "MlpPolicy",           # policy basée sur un MLP
    env,                   # environnement d'entraînement
    policy_kwargs=policy_kwargs,  # architecture du réseau
    **params               # hyperparamètres DQN
)

print(model.policy)

# ENTRAÎNEMENT

# Lancement de l'entraînement
model.learn(
    total_timesteps=TRAIN_STEPS,  # nombre total de steps
    progress_bar=True,            # barre de progression
    log_interval=4,               # fréquence des logs
    callback=eval_callback        # callback d'évaluation + early stopping
)


# SAUVEGARDE DU MODÈLE

# Sauvegarde du modèle entraîné
model.save("Save/dqn_lunar")


# RECHARGEMENT DU MODÈLE

# Suppression du modèle courant de la mémoire
del model

# Chargement du modèle sauvegardé
model = DQN.load("Save/dqn_lunar", env=env)


# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# VISUALISATION (PLAY)

# Environnement pour "visualisation / play"
play_env = gym.make("LunarLander-v3", render_mode="human")
play_env = Monitor(play_env, "./logs/play")

# Reset de l'environnement
obs, info = play_env.reset()  # <-- attention au tuple

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
