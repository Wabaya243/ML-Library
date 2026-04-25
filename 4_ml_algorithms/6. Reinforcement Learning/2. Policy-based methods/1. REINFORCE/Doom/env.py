from vizdoom import DoomGame, ScreenFormat, ScreenResolution, scenarios_path, GameVariable, Button
import gymnasium as gym  
from gymnasium import spaces 
import numpy as np 
import os 


class VizdoomEnv(gym.Env): 
    metadata = {"render_modes": ["rgb_array"], "render_fps": 35} 

    def __init__(self, cfg_file="deadly_corridor.cfg", render_mode="rgb_array", difficulty=3):
        super().__init__()  
        self.render_mode = render_mode 

        cfg_path = os.path.join(scenarios_path, cfg_file)  
        self.game = DoomGame()  # Création d'une instance Doom
        self.game.load_config(cfg_path)  # Chargement du scénario
        self.game.set_screen_format(ScreenFormat.RGB24)  # Format de l'image (RGB)
        
        # Ici on augmente la taille des images
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)  # Résolution plus grande // RES_640X480 // RES_320X240 // RES_160X120
        
        
        # Changer la difficulté directement
        self.game.set_doom_skill(difficulty)  # 1 à 5
        
        self.game.init()  # Initialisation du jeu
        
        self.last_hitcount = 0
        self.last_killcount = 0
        
        self.last_y = 0
        self.last_x = 0 

        self.attack_cooldown = 0
        self.attack_cooldown_max = 5  # nombre de steps avant tir "gratuit"
        
        # Boutons que l’agent est autorisé à utiliser
        self.buttons = [
            Button.MOVE_FORWARD,
            Button.MOVE_BACKWARD,
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.ATTACK,
        ]

        # indices réels dans VizDoom
        self.button_indices = [
            self.game.get_available_buttons().index(b)
            for b in self.buttons
        ]

        # index du bouton ATTACK dans ton espace réduit (0–7)
        self.attack_local_idx = self.buttons.index(Button.ATTACK)

        # index réel du bouton ATTACK dans VizDoom
        self.attack_vizdoom_idx = self.button_indices[self.attack_local_idx]

        # taille réelle attendue par VizDoom
        self.full_button_count = self.game.get_available_buttons_size()

        self.actions_list = [
        # ────── BASIQUES ──────
        [0, 0, 0, 0, 0, 0, 0, 0],  # 0 : ne rien faire (utile pour stabilité)
        
        # ────── DÉPLACEMENTS ──────
        [1, 0, 0, 0, 0, 0, 0, 0],  # 1 : avancer
        [0, 1, 0, 0, 0, 0, 0, 0],  # 2 : reculer
        [0, 0, 1, 0, 0, 0, 0, 0],  # 3 : strafe gauche
        [0, 0, 0, 1, 0, 0, 0, 0],  # 4 : strafe droite
        
        # ────── ROTATIONS ──────
        [0, 0, 0, 0, 1, 0, 0, 0],  # 5 : tourner gauche
        [0, 0, 0, 0, 0, 1, 0, 0],  # 6 : tourner droite
        
        # ────── COMBATS ──────
        [0, 0, 0, 0, 0, 0, 1, 0],  # 7 : tirer
        [1, 0, 0, 0, 0, 0, 1, 0],  # 8 : avancer + tirer
        [0, 0, 0, 0, 1, 0, 1, 0],  # 9 : tourner gauche + tirer
        [0, 0, 0, 0, 0, 1, 1, 0],  # 10: tourner droite + tirer
    ]

        self.action_space = spaces.Discrete(len(self.actions_list))
       
        # Ici on met à jour l'espace d'observation
        # pour correspondre à la nouvelle résolution
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )  # Image RGB de 640x480
        
        

    def step(self, action):
        # action réduite (8 boutons)
        local_action = self.actions_list[action]

        # action VizDoom complète
        doom_action = [0] * self.full_button_count
        # mapper les boutons autorisés
        for local_idx, value in enumerate(local_action):
            if value == 1:
                doom_idx = self.button_indices[local_idx]
                doom_action[doom_idx] = 1

        reward = self.game.make_action(doom_action)  # Appliquer l'action et récupérer la 
        # Récupérer les variables de jeu

        hitcount = self.game.get_game_variable(GameVariable.HITCOUNT)
        killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
        
        
        ammo = self.game.get_game_variable(GameVariable.AMMO1)
        
        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        
        
        # Pénaliser le tir automatique
        # Pénaliser le tir automatique si pas de munitions
        if local_action[self.attack_local_idx] == 1:
            if ammo > 0:
                reward -= 0.02  # petit malus pour le tir
            else:
                reward -= 0.5   # gros malus pour tirer sans munitions
            
             # Malus pour tirer sans ennemis
            if (hitcount - self.last_hitcount) == 0:
                reward -= 0.2

            # cooldown actif
            if self.attack_cooldown > 0:
                reward -= 0.05 * self.attack_cooldown  # penalité selon les temps pris
            else :
                # tir autorisé reset cooldown
                self.attack_cooldown = self.attack_cooldown_max
            
        # Récompenser les hits et kills
        reward += 2.0 * (hitcount - self.last_hitcount)
        reward += 8.0 * (killcount - self.last_killcount)
        
        
        #reward shaping pour avancer vers l'objectif
        reward += 0.20 * (y - self.last_y)

        # Malus pour rester statique après avoir tué
        if killcount > self.last_killcount and (y - self.last_y) < 0.05:
            reward -= 0.1  # Encourager à bouger après avoir tué
        
                
        # Sauvegarder pour le prochain step
        self.last_hitcount = hitcount
        self.last_killcount = killcount
        self.last_y = y
        self.last_x = x

        # decrementer le cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        
                
        done = self.game.is_episode_finished()  # Vérifier si l'épisode est terminé

        # Observation : si l'épisode est terminé, on retourne un tableau de zéros
        obs = (
            self.game.get_state().screen_buffer
            if not done
            else np.zeros((480, 640, 3), dtype=np.uint8)  # Taille correspondante à la résolution
        )

        terminated = done
        truncated = False
        info = {}  # Gymnasium exige un dictionnaire info
        

        return obs, reward, terminated, truncated, info  # Retourne observation et infos

    def reset(self, *, seed=None, options=None):  
        self.game.new_episode()  # Commence un nouvel épisode
        self.last_hitcount = 0 
        self.last_killcount = 0
        self.last_y = 0
        self.last_x = 0
        obs = self.game.get_state().screen_buffer  # Récupère l'image initiale
        info = {}  # Gymnasium exige un dictionnaire info
        return obs, info  # Retourne l'observation et info


    def render(self): 
        if self.render_mode == "rgb_array":  # Si mode rgb_array
            state = self.game.get_state()
            if state is None:  # Si l'épisode est terminé ou état non disponible
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return state.screen_buffer
            

    def close(self):
        self.game.close()  # Ferme le jeu
