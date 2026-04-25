# Environnement
import gymnasium as gym
import highway_env

# Agent RL
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback

# Outils
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler
import os

# Création des environnements
env = gym.make('highway-fast-v0', render_mode='rgb_array')
eval_env = gym.make('highway-fast-v0', render_mode='rgb_array')

# Configuration des observations et actions
config = {
    "action": {"type": 'ContinuousAction'},
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # pondérations pour la conversion RGB → niveaux de gris
        "scaling": 1.75,
    }
}

env.configure(config)
env.reset(seed=42)

eval_env.configure(config)
eval_env.reset(seed=40)

# Fin de l’épisode si le véhicule sort de la route
env.config["offroad_terminal"] = True

# Fonction objectif pour Optuna
def objective(trial):

    # Hyperparamètres à optimiser
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 4)
    learning_starts = trial.suggest_int("learning_starts", 0, 1000)
    tau = trial.suggest_float("tau", 0.01, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_int("buffer_size", 20000, 100000)

    # Dossier de logs pour TensorBoard et les modèles sauvegardés
    run_name = (
        f"lr_{learning_rate}_tf_{train_freq}_gs_{gradient_steps}_"
        f"ls_{learning_starts}_tau_{tau}_bs_{batch_size}_buf_{buffer_size}"
    )
    log_dir = os.path.join("highway_sac", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Possibilité d’utiliser plusieurs seeds pour plus de robustesse
    for model_seed in [9]:

        # Création du modèle SAC
        model = SAC(
            "CnnPolicy",
            env,
            buffer_size=buffer_size,
            verbose=0,
            learning_starts=learning_starts,
            device="auto",
            tensorboard_log=log_dir,
            learning_rate=learning_rate,
            tau=tau,
            batch_size=batch_size,
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            seed=model_seed
        )

        # Callback d’évaluation périodique sur l’environnement de test
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=f"{log_dir}/best_model/",
            log_path="./logs/",
            eval_freq=1000,
            deterministic=True,
            render=False,
        )

        # Entraînement du modèle
        model.learn(
            total_timesteps=int(4e4),
            callback=eval_callback
        )

        # Réinitialisation des environnements
        env.reset(seed=42)
        eval_env.reset(seed=40)

    # Critère de sélection : meilleure récompense moyenne observée en évaluation
    return eval_callback.best_mean_reward

# Création de l’étude Optuna
study = optuna.create_study(
    sampler=RandomSampler(seed=42),  # Recherche aléatoire
    pruner=MedianPruner(),
    direction="maximize",
)

# Lancement de l’optimisation
study.optimize(objective, n_trials=16)

# Affichage des meilleurs hyperparamètres trouvés
print("Meilleurs hyperparamètres :", study.best_params)

# Sauvegarde des résultats de l’étude
df = study.trials_dataframe()
df.to_csv("optuna_study_results.csv", index=False)
print("Résultats de l’étude sauvegardés dans optuna_study_results.csv")
