import time
import numpy as np

class Logger:
    """
    Classe utilitaire pour suivre :
    - le nombre de pas (steps)
    - les épisodes
    - les récompenses
    - la durée des épisodes
    - les performances (FPS, temps écoulé)
    """
    def __init__(self, total_steps: int, num_checkpoints: int):
        # Compteurs globaux
        self.current_step = 0          # Nombre total de steps effectués
        self.current_episode = 1       # Numéro de l’épisode courant
        self.current_return = 0.0      # Récompense cumulée de l’épisode courant
        self.current_length = 0        # Longueur de l’épisode courant
        
        # Historique des épisodes
        self.episode_returns = []      # Récompenses totales par épisode
        self.episode_lengths = []      # Longueurs des épisodes
        
        # Logs personnalisés (ex : losses, entropy, etc.)
        self.custom_logs = {}
        self.custom_log_keys = []
        
        # Gestion du temps
        self.start_time = time.time()
        self.total_steps = total_steps
        self.num_checkpoints = num_checkpoints
        
        # Intervalle entre checkpoints d’affichage
        self.checkpoint_interval = max(1, self.total_steps // self.num_checkpoints)
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_step = 0
        
        self.header_printed = False
        
        # Paramètres d’affichage
        self.log_interval = 100  # Afficher les logs tous les N steps
        self.window = 20         # Fenêtre glissante pour moyennes
    
    def log(self, reward: float, termination: bool, truncation: bool, **kwargs):
        """
        Met à jour les statistiques après chaque step
        """
        # Mise à jour des compteurs
        self.current_step += 1
        self.current_return += reward
        self.current_length += 1

        # Si l’épisode se termine
        if termination or truncation:
            self.episode_returns.append(self.current_return)
            self.episode_lengths.append(self.current_length)
            self.current_episode += 1
            self.current_return = 0.0
            self.current_length = 0

        # Mise à jour des logs personnalisés
        for key, value in kwargs.items():
            if key not in self.custom_log_keys:
                self.custom_log_keys.append(key)
            self.custom_logs[key] = value

    def print_logs(self):
        """
        Affiche la progression de l’entraînement
        """
        if self.current_step % self.log_interval == 0 and len(self.episode_returns) > 0:
            elapsed_time = time.time() - self.start_time
            
            # Calcul des FPS depuis le dernier checkpoint
            steps_since_checkpoint = self.current_step - self.last_checkpoint_step
            time_since_checkpoint = time.time() - self.last_checkpoint_time
            fps = (
                steps_since_checkpoint / time_since_checkpoint
                if time_since_checkpoint > 0 else 0
            )

            # Progression globale
            progress = 100 * self.current_step / self.total_steps
            
            # Moyennes glissantes
            mean_reward = np.mean(
                self.episode_returns[-self.window:]
            ) if len(self.episode_returns) >= self.window else np.mean(self.episode_returns)
            
            mean_ep_length = np.mean(
                self.episode_lengths[-self.window:]
            ) if len(self.episode_lengths) >= self.window else np.mean(self.episode_lengths)
            
            # Formatage du temps (hh:mm:ss)
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

            # Impression de l’en-tête (une seule fois)
            if not self.header_printed:
                log_header = (
                    f"{'Progress':>8}  |  "
                    f"{'Step':>8}  |  "
                    f"{'Episode':>8}  |  "
                    f"{'Mean Rew':>8}  |  "
                    f"{'Mean Len':<7}  |  "
                    f"{'FPS':>6}  |  "
                    f"{'Time':>8}"
                )
                for key in self.custom_log_keys:
                    log_header += f"  |  {key:>{len(key)}}"
                print(log_header)
                self.header_printed = True

            # Construction de la ligne de log
            log_string = (
                f"{progress:>7.1f}%  |  "
                f"{self.current_step:>8,}  |  "
                f"{self.current_episode:>8,}  |  "
                f"{mean_reward:>8.2f}  |  "
                f"{mean_ep_length:>8.1f}  |  "
                f"{fps:>6,.0f}  |  "
                f"{formatted_time:>8}"
            )

            # Ajout des logs personnalisés
            for key in self.custom_log_keys:
                value = self.custom_logs.get(key, 0)
                if isinstance(value, float):
                    log_string += f"  |  {value:>{len(key)}.2f}"
                elif isinstance(value, int):
                    log_string += f"  |  {value:>{len(key)}d}"
                else:
                    log_string += f"  |  {str(value):>{len(key)}}"

            # Impression sur une seule ligne (overwrite)
            print(f"\r{log_string}", end='')
        
        # Gestion des checkpoints
        if self.current_step % self.checkpoint_interval == 0:
            print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step

    @property
    def logs(self):
        """
        Retourne un résumé complet de l’entraînement
        """
        return {
            'total_steps': self.current_step,
            'total_episodes': self.current_episode - 1,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'best_reward': (
                np.max(self.episode_returns)
                if len(self.episode_returns) > 0 else None
            ),
            'total_duration': time.time() - self.start_time,
            'mean_fps': self.current_step / (time.time() - self.start_time + 1e-6),
            'custom_logs': self.custom_logs
        }
