# Importation de la bibliothèque pandas pour la manipulation de données sous forme de DataFrame
import pandas as pd

# Importation de numpy pour les opérations numériques avancées
import numpy as np

# Importation de matplotlib pour la visualisation
import matplotlib

# Importation du module pyplot pour tracer des graphiques
import matplotlib.pyplot as plt

# Définir le backend matplotlib à 'Agg' pour générer des graphiques sans interface graphique
matplotlib.use('Agg')

# Importation de yfinance pour récupérer les données financières
import yfinance as yf

# Importation de modules et fonctions de FinRL pour la gestion des données financières, l'environnement et l'agent DRL
from finrl import config
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent

from gymnasium.utils import seeding
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv



# Liste des actions du STI (Straits Times Index) à analyser
STI_stock_list = ["A17U.SI","9CI.SI","C38U.SI","C09.SI","C52.SI","D01.SI","D05.SI","BUOU.SI","G13.SI","H78.SI","C07.SI","J36.SI","BN4.SI","AJBU.SI",
                  "N2IU.SI","ME8U.SI","M44U.SI","O39.SI","S58.SI","U96.SI","S68.SI","C6L.SI","Z74.SI","S63.SI","Y92.SI","U11.SI","U14.SI","V03.SI",
                  "F34.SI","BS6.SI"]


# Définition des dates pour l'analyse : début, fin de l'entraînement et fin générale
start_date = '2012-01-01'
train_end_date = '2020-07-01'
end_date = '2022-01-31'


# Création d'une liste pour stocker les tickers réellement actifs
active_tickers = []

# Vérification pour chaque ticker si des données existent sur yfinance
for tic in STI_stock_list:
    df = yf.download(tic, start='2012-01-01', end='2022-01-01', progress=False)  # Téléchargement des données historiques
    if not df.empty:  # Si des données existent pour ce ticker
        active_tickers.append(tic)  # Ajouter le ticker à la liste des actifs

# Affichage des tickers actifs
print("Tickers actifs :", active_tickers)


# Liste pour stocker les DataFrames de chaque ticker actif
data_list = []

for tic in active_tickers:
    df = yf.download(tic, start=start_date, end=end_date, interval='1d', progress=False)
    
    if df.empty:
        continue
    
    # --- ÉTAPE CRUCIALE : Aplatir le MultiIndex de yfinance ---
    # Si les colonnes sont un MultiIndex (Price, Ticker), on ne garde que le premier niveau (Price)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()  # Met la date en colonne
    
    # Renommer pour correspondre aux attentes de FinRL
    df.rename(columns={
        'Date': 'date', 
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    df['tic'] = tic
    
    # Garder uniquement les colonnes nécessaires (et s'assurer qu'elles sont simples)
    df = df[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
    data_list.append(df)

# Combiner en un DataFrame plat
raw_df = pd.concat(data_list, ignore_index=True)

# 1. S'assurer que la date est au format datetime
raw_df['date'] = pd.to_datetime(raw_df['date'])
# 2. Trier par date et ticker (indispensable pour FinRL)
raw_df = raw_df.sort_values(['date', 'tic']).reset_index(drop=True)


# Vérification
print(raw_df.head())
print(raw_df.columns)


# Vérification spécifique : ligne correspondant à l'action 'C38U.SI' à la date '2009-02-09'
raw_df.loc[(raw_df["tic"]=="C38U.SI") & (raw_df["date"]=="2019-02-09")]

# Vérification des valeurs manquantes pour chaque colonne
raw_df.isnull().sum()

# Nombre d'actions uniques dans le DataFrame
raw_df['tic'].nunique()

# Compter le nombre de lignes pour chaque ticker
raw_df['tic'].value_counts()

# Afficher les dimensions du DataFrame (lignes, colonnes)
raw_df.shape


# Initialisation de l'ingénieur de caractéristiques pour calculer les indicateurs techniques
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'], # Spécifiez les indicateurs si nécessaire
    use_turbulence=False,
    user_defined_feature=False
)


# Prétraitement des données pour ajouter les indicateurs techniques
df = fe.preprocess_data(raw_df)
#df = fe.clean_data(raw_df)  # Optionnel : nettoyage des données


# Vérification de la nouvelle forme du DataFrame après prétraitement
df.shape

# Vérification du nombre d'actions uniques après prétraitement
df['tic'].nunique()


# Tri du DataFrame par date et ticker
df = df.sort_values(['date','tic'],ignore_index=True)

# Création d'un index basé sur la factorisation des dates
df.index = df.date.factorize()[0]

# Listes pour stocker les matrices de covariance et les retours

cov_list = []
return_list = []

lookback = 252
unique_dates = df.date.unique()

# Boucle pour calculer les matrices de covariance et les rendements historiques
for i in range(lookback, len(unique_dates)):
    # Extraire les données de la fenêtre
    data_lookback = df.loc[i-lookback:i, :]
    
    # Créer la table pivot (Prix de clôture)
    price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
    
    # Calculer les rendements
    return_lookback = price_lookback.pct_change().dropna()
    
    # Calculer la matrice de covariance
    covs = return_lookback.cov().values
    
    # Ajouter aux listes
    return_list.append(return_lookback)
    cov_list.append(covs)

# --- VÉRIFICATION DES LONGUEURS ---
# Les trois chiffres DOIVENT être identiques (2281)
print(f"Dates: {len(unique_dates[lookback:])}")
print(f"Covariances: {len(cov_list)}")
print(f"Returns: {len(return_list)}")

# Création du DataFrame
df_cov = pd.DataFrame({
    'date': unique_dates[lookback:], 
    'cov_list': cov_list, 
    'return_list': return_list
})

# Fusionner avec les données originales
df = df.merge(df_cov, on='date', how='left')
df = df.sort_values(['date','tic']).reset_index(drop=True)

print("DataFrame final prêt !")
print(df.head())

# Vérification de la forme finale du DataFrame
df.shape

# Nombre d'actions uniques
df['tic'].nunique()

# Comptage du nombre de lignes pour chaque ticker
df['tic'].value_counts()

# Séparation des données pour l'entraînement
train = data_split(df, start_date, train_end_date)
trade = data_split(df, '2020-01-01', end_date)  # Optionnel : séparation pour le trading

# Affichage des premières lignes des données d'entraînement
train.head()


class StockPortfolioEnv(gym.Env):
    """
    Environnement de trading multi-actions pour un portefeuille boursier, compatible avec OpenAI Gym.

    Attributs
    ----------
        df : DataFrame
            Données historiques des actions et indicateurs techniques.
        stock_dim : int
            Nombre d'actions uniques dans le portefeuille.
        hmax : int
            Nombre maximum d'actions pouvant être achetées ou vendues en une seule transaction.
        initial_amount : int
            Capital initial du portefeuille.
        transaction_cost_pct : float
            Pourcentage de frais de transaction par trade.
        reward_scaling : float
            Facteur de mise à l’échelle de la récompense, utile pour l’entraînement DRL.
        state_space : int
            Dimension de l’espace des états (features).
        action_space : int
            Dimension de l’espace des actions (nombre d’actions du portefeuille).
        tech_indicator_list : list
            Liste des noms des indicateurs techniques utilisés.
        turbulence_threshold : float
            Seuil de turbulence pour ajuster le risque.
        day : int
            Index du jour courant dans les données.
        lookback : int
            Fenêtre de calcul pour les matrices de covariance.

    Méthodes
    -------
        step(actions)
            Effectue une action, calcule la récompense et renvoie le nouvel état.
        reset()
            Réinitialise l’environnement au jour initial.
        render()
            Retourne l’état actuel.
        softmax_normalization(actions)
            Normalise les poids du portefeuille avec la fonction softmax.
        save_asset_memory()
            Sauvegarde l’évolution du portefeuille.
        save_action_memory()
            Sauvegarde les actions/positions prises à chaque pas.
        get_sb_env()
            Retourne un environnement compatible avec Stable-Baselines3.
    """
    metadata = {'render.modes': ['human']}  # Mode de rendu humain

    def __init__(self, 
                 df,
                 stock_dim,
                 hmax,
                 initial_amount,
                 transaction_cost_pct,
                 reward_scaling,
                 state_space,
                 action_space,
                 tech_indicator_list,
                 turbulence_threshold=None,
                 lookback=252,
                 day=0):
        # Initialisation des paramètres de l'environnement
        self.day = day  # Jour courant
        self.lookback = lookback  # Fenêtre de lookback pour la covariance
        self.df = df  # DataFrame des données
        self.stock_dim = stock_dim  # Nombre d’actions
        self.hmax = hmax  # Max actions par trade
        self.initial_amount = initial_amount  # Capital initial
        self.transaction_cost_pct = transaction_cost_pct  # Frais par trade
        self.reward_scaling = reward_scaling  # Facteur de récompense
        self.state_space = state_space  # Dimension de l’état
        self.action_space = action_space  # Dimension des actions
        self.tech_indicator_list = tech_indicator_list  # Liste des indicateurs techniques

        # Définition de l’espace des actions : vecteur continu [0,1] pour chaque action
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        
        # Définition de l’espace des états : covariance + indicateurs techniques
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stock_dim + len(self.tech_indicator_list), self.stock_dim),
            dtype=np.float32  # Crucial
        )


        # Charger les données du premier jour
        self.data = self.df.loc[self.day, :]
        self.covs = self.data['cov_list'].values[0]  # Matrice de covariance initiale
        
                
        # S'assurer que c'est 2D (matrice de covariance)
        if self.covs.ndim == 1:
            self.covs = self.covs.reshape(-1, 1)
            
        # Etat initial : covariance + indicateurs techniques
        
        # Covariance : (27,27)
        cov_matrix = self.covs  # ne pas flatten
        
        # Indicateurs techniques : stack verticalement (4,27)
        tech_matrix = np.vstack([self.data[tech].values for tech in self.tech_indicator_list])
        
        # Concaténation verticale : (27+4, 27) = (31,27)
        self.state = np.vstack([cov_matrix, tech_matrix])
        self.terminal = False  # Flag pour savoir si la simulation est terminée
        self.turbulence_threshold = turbulence_threshold  # Seuil de turbulence
        self.portfolio_value = self.initial_amount  # Valeur initiale du portefeuille

        # Mémoire pour suivre la valeur du portefeuille et les retours
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]  # Poids initiaux égaux
        self.date_memory = [self.data.date.unique()[0]]  # Dates pour chaque étape

    def step(self, actions):
        """Effectue une étape dans l'environnement"""
        self.terminal = self.day >= len(self.df.index.unique())-1  # Vérifie si fin de données

        if self.terminal:
            # Optionnel : commenter ces lignes pour gagner en vitesse
            # df = pd.DataFrame(self.portfolio_return_memory, columns=['daily_return'])
            # plt.plot(df.daily_return.cumsum(), 'r')
            # plt.savefig('results/cumulative_reward.png') # C'est ici que ça plante
            # plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory, columns=['daily_return'])
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252**0.5) * df_daily_return['daily_return'].mean() / df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal, False, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data
    
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # Récupération correcte de la covariance et des indicateurs
            self.covs = np.array(self.data['cov_list'].values[0], dtype=np.float32)
            tech_matrix = np.array([self.data[tech].values for tech in self.tech_indicator_list], dtype=np.float32)
            self.state = np.vstack([self.covs, tech_matrix])
            
            # Calcul du rendement (inchangé)
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            self.portfolio_value = self.portfolio_value * (1 + portfolio_return)
            
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(self.portfolio_value)
            self.reward = self.portfolio_value
    
        return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.asset_memory = [self.initial_amount]
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        
        self.data = self.df.loc[self.day, :]
        
        # --- CORRECTION ICI ---
        # On prend la matrice de covariance (27x27)
        self.covs = np.array(self.data['cov_list'].values[0], dtype=np.float32)
        
        # On crée la matrice technique (4x27)
        tech_matrix = np.array([self.data[tech].values for tech in self.tech_indicator_list], dtype=np.float32)
        
        # Assemblage (27 + 4 = 31, 27)
        self.state = np.vstack([self.covs, tech_matrix])
        # ----------------------
    
        self.actions_memory = [[1/self.stock_dim]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        
        return self.state, {}

    def render(self, mode='human'):
        """Rendu de l'environnement"""
        return self.state

    def softmax_normalization(self, actions):
        """Normalise les poids avec softmax pour que la somme = 1"""
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        return numerator / denominator

    def save_asset_memory(self):
        """Retourne un DataFrame des valeurs du portefeuille et des retours"""
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """Retourne un DataFrame des actions prises à chaque étape"""
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list, columns=['date'])
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values  # Colonnes = tickers
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        """Initialise la seed pour la reproductibilité"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Retourne un environnement compatible avec Stable-Baselines3"""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    


# Détermination des dimensions du portefeuille
stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

TECHNICAL_INDICATORS_LIST = ['macd', 'rsi_30', 'cci_30', 'dx_30']


# Paramètres pour l'environnement
env_kwargs = {
    "hmax": 100,  # Max actions par trade
    "initial_amount": 1000000,  # Capital initial
    "transaction_cost_pct": 0.001,  # Frais par trade
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-2,
    "tech_indicator_list": TECHNICAL_INDICATORS_LIST 
}

# s'assurer que chaque cov_list est un np.ndarray 2D avec la bonne taille
def fix_cov(x):
    try:
        arr = np.array(x)
        return arr.reshape(stock_dimension, stock_dimension)
    except:
        # Si impossible (valeur manquante ou mauvaise taille), mettre une matrice nulle
        return np.zeros((stock_dimension, stock_dimension))

train['cov_list'] = train['cov_list'].apply(fix_cov)


# Création de l'environnement pour l'entraînement
e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))  # Affichage du type d'environnement SB3



agent = DRLAgent(env = env_train)

# agent = DRLAgent(env = env_train)

# A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
# model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)
# trained_a2c = agent.train_model(model=model_a2c, 
#                                 tb_log_name='a2c',
#                                 total_timesteps=50000)
# trained_a2c.save('/content/trained_models/trained_a2c.zip')


# agent = DRLAgent(env = env_train)
# PPO_PARAMS = {
#     "n_steps": 2048,
#     "ent_coef": 0.005,
#     "learning_rate": 0.0001,
#     "batch_size": 128,
# }
# model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)
# trained_ppo = agent.train_model(model=model_ppo, 
#                              tb_log_name='ppo',
#                              total_timesteps=80000)
# trained_ppo.save('/content/trained_models/trained_ppo.zip')


# agent = DRLAgent(env = env_train)
# DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}


# model_ddpg = agent.get_model("ddpg",model_kwargs = DDPG_PARAMS)
# trained_ddpg = agent.train_model(model=model_ddpg, 
#                              tb_log_name='ddpg',
#                              total_timesteps=50000)
# trained_ddpg.save('/content/trained_models/trained_ddpg.zip')


# agent = DRLAgent(env = env_train)
# SAC_PARAMS = {
#     "batch_size": 128,
#     "buffer_size": 500000,
#     "learning_rate": 0.0003,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.2",
#     "gamma": 1
# }

# model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
# trained = agent.train_model(model=model_sac, 
#                              tb_log_name='sac',
#                              total_timesteps=500000)
# trained.save('/content/trained_models/trained_sac.zip')

import os

# Créer le dossier 'results' s'il n'existe pas
if not os.path.exists("./results"):
    os.makedirs("./results")
    
agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 100, 
              "buffer_size": 10000,
              "gamma": 1,
              "learning_rate": 0.0002}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
trained = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=150000)
#trained.save('/content/trained_models/trained_td3.zip')