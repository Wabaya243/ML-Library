import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='ipykernel')
warnings.filterwarnings("ignore", category=FutureWarning, module='stable_baselines3.common.monitor')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow.lite.python.util')
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gymnasium as gym
import numpy as np
import os
import multiprocessing

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

# Détection du nombre de cœurs CPU disponibles
cores = multiprocessing.cpu_count()
print(f"Nombre de cœurs CPU disponibles : {cores}")

# Création de l'environnement vectorisé pour évaluation
eval_env = make_vec_env("BipedalWalker-v3", n_envs=1, vec_env_cls=SubprocVecEnv) 

# Création des dossiers pour logs et modèles si ils n'existent pas
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

best_model_dir = "./model/"
os.makedirs(best_model_dir, exist_ok=True)

eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

# Callback pour évaluer et sauvegarder le meilleur modèle
# Évalue tous les eval_freq steps * nombre d'env
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=eval_log_dir,
    eval_freq=500,       # fréquence d'évaluation
    deterministic=True,  # actions déterministes lors de l'évaluation
    render=False
)

# Création de l'environnement vectorisé principal pour l'entraînement
# n_envs = cores → utilisation optimale du CPU
vec_env = make_vec_env(
    "BipedalWalker-v3",
    n_envs=cores,
    vec_env_cls=SubprocVecEnv,
    monitor_dir=log_dir
)

# Configuration du bruit pour TD3
n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
vec_action_noise = VectorizedActionNoise(action_noise, cores)

# Hyperparamètres TD3
gamma = 0.995            # Facteur de discount (réduction des récompenses futures)
tau = 0.005              # Paramètre de soft-update pour les réseaux cibles
lr = 0.001               # Taux d'apprentissage
learning_starts = 10000  # Nombre de steps avant que l'agent commence à apprendre

# Création du modèle TD3
# Remarque : Stable-Baselines3 est optimisé pour le CPU. L'utilisation du GPU n'apporte pas forcément d'amélioration ici.
model = TD3(
    "MlpPolicy",
    vec_env,
    action_noise=action_noise,
    train_freq=1,
    gradient_steps=-1,
    gamma=gamma,
    learning_rate=lr,
    tau=tau,
    learning_starts=learning_starts,
    verbose=1
)

model = TD3.load("model/best_model.zip", env=eval_env)

# Afficher la politique du modèle
print(model.policy)

# Entraînement du modèle
# 1 000 000 steps → environ 4 heures sur setup de base
model.learn(total_timesteps=1_000_000, log_interval=100, callback=eval_callback, progress_bar=True)

# Sauvegarde du modèle
model.save('model/TD3_biped_walker')

# Suppression du modèle pour libérer la mémoire
del model

# Rechargement du modèle pour évaluation ou utilisation ultérieure
model = TD3.load("model/TD3_biped_walker", env=vec_env)



# Fonction pour calculer la moyenne mobile
def moving_average(values, window):
    """
    Lisser une série de valeurs avec une moyenne mobile.

    :param values: (numpy array) tableau de valeurs à lisser
    :param window: (int) taille de la fenêtre pour la moyenne mobile
    :return: (numpy array) valeurs lissées
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


# Fonction pour tracer la courbe d'apprentissage
def plot_results(log_folder, title="Courbe d'apprentissage", window=50):
    """
    Trace la courbe d'apprentissage avec moyenne mobile et écart-type.

    :param log_folder: (str) dossier contenant les résultats SB3 à tracer
    :param title: (str) titre du graphique
    :param window: (int) taille de la fenêtre pour moyenne mobile et calcul de l'écart-type
    """
    # Charger les résultats SB3 et extraire timesteps et récompenses
    x, y = ts2xy(load_results(log_folder), "timesteps")

    # Calcul de la moyenne mobile
    y_mean = moving_average(y, window=window)
    x_mean = x[len(x) - len(y_mean):]  # Ajuster la taille pour correspondre à la moyenne mobile
    x_mean = np.array(x_mean, dtype=np.int64)

    # Calcul de l'écart-type glissant
    y_std = np.array([np.std(y[max(0, i - window + 1):i + 1]) for i in range(len(y))])
    y_std = y_std[len(y_std) - len(y_mean):]

    # Tracé de la courbe
    fig, ax = plt.subplots()
    ax.plot(x_mean, y_mean, label="Récompense Moyenne")
    ax.fill_between(x_mean, y_mean - y_std, y_mean + y_std, alpha=0.3, label="Écart-type")

    ax.set_xlabel("Nombre de Timesteps")
    ax.set_ylabel("Récompenses")
    ax.set_title(title + " (Lissé avec écart-type)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# Exemple d'utilisation après l'entraînement
plot_results(log_dir, window=20, title="Courbe d'apprentissage")



# ÉVALUATION FINALE

# Évaluation sur 10 épisodes
mean_reward, std_reward = evaluate_policy(
    model,
    model.get_env(),
    n_eval_episodes=10
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def play_eval_env():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    env = Monitor(env)

    return env

play_eval = SubprocVecEnv([play_eval_env])


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