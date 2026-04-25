import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

# ============================
#  ÉCRITURE DES FONCTIONS PRINCIPALES ET DE LA CLASSE
# ============================
class QLearningAgent():
    def __init__(self, obs_n, act_n, lr, gamma, e_greed):
        # obs_n : nombre d'états observables dans l'environnement
        self.act_n = act_n      # nombre d'actions possibles dans l'espace d'actions
                               # les actions sont représentées par des entiers de 0 à act_n-1
        self.lr = lr            # taux d'apprentissage (learning rate)
        self.gamma = gamma      # facteur de discount (réduction des récompenses futures)
        self.epsilon = e_greed  # taux d'exploration (epsilon-greedy)
        
        # Table Q : stocke les valeurs Q(s, a)
        self.Q = np.zeros((obs_n, act_n))  
    
    # ==========================================
    # Choix de l'action selon la politique ε-greedy
    # Behaviour Policy
    # ==========================================
    def sample(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            # avec une probabilité epsilon, choisir une action aléatoire (exploration)
            action = np.random.choice(self.act_n)
        else:
            # sinon, choisir la meilleure action selon la table Q actuelle (exploitation)
            action = self.predict(obs)
            
        return action
    
    # ==========================================
    # Politique cible : choisir la meilleure action
    # Target Policy
    # ==========================================
    def predict(self, obs):
        # récupérer les valeurs Q de toutes les actions dans l'état obs
        obs_list = self.Q[obs, :]
        
        # valeur Q maximale pour cet état
        rw_max = np.max(obs_list)
        
        # plusieurs actions peuvent avoir la même valeur maximale
        action_list = np.where(obs_list == rw_max)[0]
        
        # choisir aléatoirement parmi les meilleures actions
        action = np.random.choice(action_list)
        
        return action
    
    # ==========================================
    # Mise à jour de la table Q (apprentissage)
    # Apprentissage sur un pas de temps (1-step)
    # ==========================================
    def learn(self, obs, action, reward, next_obs, done):
        # valeur Q actuelle pour (état, action)
        q = self.Q[obs, action]
        
        # y = cible TD (Temporal Difference target)
        if done:
            # si l'épisode est terminé (état terminal)
            y = reward
        else:
            # sinon, récompense + meilleure valeur Q de l'état suivant
            y = reward + self.gamma * self.Q[next_obs, self.predict(next_obs)]
        
        # mise à jour de Q(s,a) selon l'équation de Bellman
        self.Q[obs, action] = self.Q[obs, action] + self.lr * (y - q)


# ==============================================
# Écrire la classe SarsaAgent en s’inspirant de QLearningAgent
# ==============================================
class SarsaAgent():
    def __init__(self, obs_n, act_n, lr, gamma, e_greed):
        self.act_n = act_n      # nombre d’actions possibles
        self.lr = lr            # taux d’apprentissage (learning rate)
        self.gamma = gamma      # facteur de discount (réduction des récompenses futures)
        self.epsilon = e_greed  # taux d’exploration (epsilon-greedy)
        
        # Table Q : stockage des valeurs Q(s, a)
        self.Q = np.zeros((obs_n, act_n))  
    
    # ======================================================
    # Choisir une action selon la stratégie epsilon-greedy
    # Behaviour Policy & Target Policy (SARSA = on-policy)
    # ======================================================
    def sample(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            # avec une probabilité epsilon, choisir une action aléatoire
            action = np.random.choice(self.act_n)
        else:
            # sinon, choisir la meilleure action selon la table Q
            action = self.predict(obs)
            
        return action
    
    # ======================================================
    # À partir de l’état observé, retourner l’action optimale
    # selon la table Q actuelle
    # ======================================================
    def predict(self, obs):
        # valeurs Q pour toutes les actions dans l’état obs
        obs_list = self.Q[obs, :]
        
        # valeur Q maximale dans cet état
        rw_max = np.max(obs_list)
        
        # plusieurs actions peuvent partager la même valeur maximale
        action_list = np.where(obs_list == rw_max)[0]
        
        # sélection aléatoire parmi les meilleures actions
        action = np.random.choice(action_list)
        
        return action
    
    # ======================================================
    # Mise à jour de la table Q (apprentissage SARSA)
    # ======================================================
    def learn(self, obs, action, reward, next_obs, done):
        # valeur Q actuelle pour l’état et l’action effectuée
        q = self.Q[obs, action]
        
        if done:
            # si l’épisode est terminé (état terminal)
            y = reward
        else:
            # SARSA : utiliser l’action réellement choisie
            # selon la politique epsilon-greedy
            y = reward + self.gamma * self.Q[next_obs, self.sample(next_obs)]
        
        # mise à jour de Q(s,a) avec la règle de Bellman
        self.Q[obs, action] = self.Q[obs, action] + self.lr * (y - q)


# =========================
# Fonction d'entraînement
# =========================
def train_episode(env, agent):
    total_reward = 0      # enregistre le score total obtenu pendant cet épisode
    action_list = []      # liste des actions effectuées
    obs = env.reset()[0]  # réinitialiser l’environnement et obtenir l’état initial
    
    while 1:
        # Behaviour Policy (politique de comportement)
        action = agent.sample(obs)   # choisir une action
        action_list.append(action)
        
        # interaction avec l’environnement : nouvel état, récompense, fin ou non
        next_obs, reward, done, info, _ = env.step(action)
        
        # mise à jour de la table Q
        agent.learn(obs, action, reward, next_obs, done)
        
        obs = next_obs                # mise à jour de l’état courant
        total_reward += reward        # accumulation des récompenses
        
        if done:                      # si l’épisode est terminé
            break
            
    return total_reward, action_list.copy()


# =========================
# Fonction de test
# =========================
def test_episode(env, agent):
    total_reward = 0      # enregistre le score total du test
    env.reset()
    obs = env.reset()[0]
    
    img = plt.imshow(env.render())    # affichage initial de l’environnement
    action_list = []                  # actions effectuées
    reward_list = []                  # récompenses reçues
    
    while 1:
        # sélection de l’action optimale (politique greedy)
        action = agent.predict(obs)
        action_list.append(action)
        
        # interaction avec l’environnement
        obs, reward, done, info, _ = env.step(action)
        
        # accumulation de la récompense
        total_reward += reward
        reward_list.append(reward)
        
        time.sleep(0.2)               # pause de 0.2 seconde entre chaque frame
        data = env.render()           # rendu graphique
        img.set_data(data)            # mise à jour de l’image
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
        if done:                      # si l’épisode est terminé
            break
            
    return total_reward, action_list, reward_list


# implementation de l'env
env = gym.make('CliffWalking-v0', render_mode='rgb_array')

lr = 0.1
gamma = 0.9
e_greedy = 0.1
episode_max = 1000
reward_Q = []

agent_q = QLearningAgent(env.observation_space.n, env.action_space.n, lr, gamma, e_greedy)

from tqdm import tqdm

#entrainement 
for episode in tqdm(range(episode_max)):
    reward, _ = train_episode(env, agent_q)
    reward_Q.append(reward)


#test 
test_reward_Q, action_list_Q, reward_list_Q = test_episode(env, agent_q)
print("Test reward Q:", test_reward_Q)

## Sarsa TRAIN e TEST

# 参数设置
lr = 0.1
gamma = 0.9
e_greed = 0.1
episode_max = 1000
reward_S = []  
agent_S = SarsaAgent(env.observation_space.n, env.action_space.n, lr, gamma, e_greed)
# train
for episode in tqdm(range(episode_max)):  
    reward,_ = train_episode(env, agent_S)
    reward_S.append(reward)


#test
test_reward_S,action_list_S,reward_list_S = test_episode(env, agent_S)
print("Test reward S:", test_reward_S)


# 输出最终奖励
print('————————train————————')
## 将最后-100:-1的均值作为收敛的结果
print('Q-learning total reward:',np.mean(reward_Q[-100:-1]))
print('SARSA total reward:',np.mean(reward_S[-100:-1]))

print('————————test————————')
print('Q-learning total reward:',test_reward_Q)
print('SARSA total reward:',test_reward_S)


# Afficher le chemin planifié
print("Chemin planifié par Q-learning :")
for i in range(len(action_list_Q)):
    print(
        f"À l’étape {i+1}, l’action est {action_dic[action_list_Q[i]]}, "
        f"la récompense est {reward_list_Q[i]}, "
        f"la récompense cumulée est {np.sum(reward_list_Q[:i+1])}"
    )
print(f"Récompense totale finale : {np.sum(reward_list_Q)}")

print("——————————————————————————————————————————")

print("Chemin planifié par SARSA :")
for i in range(len(action_list_S)):
    print(
        f"À l’étape {i+1}, l’action est {action_dic[action_list_S[i]]}, "
        f"la récompense est {reward_list_S[i]}, "
        f"la récompense cumulée est {np.sum(reward_list_S[:i+1])}"
    )
print(f"Récompense totale finale : {np.sum(reward_list_S)}")
