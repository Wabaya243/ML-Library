


# a lancer dans la Ypython %matplotlib auto

import pandas as pd
# Importation du module pyplot pour créer des graphiques
import matplotlib.pyplot as plt
import yfinance as yf

import torch

# Importation de l’outil FinRL pour télécharger les données Yahoo Finance
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# Importation des outils de prétraitement des données (indicateurs techniques, split, etc.)
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# Importation de l’environnement de trading d’actions (gym RL)
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
# Importation de l’agent de Deep Reinforcement Learning
from finrl.agents.stablebaselines3.models import DRLAgent
# Importation de l’outil de configuration des logs
from stable_baselines3.common.logger import configure
# Importation du processeur de données FinRL
from finrl.meta.data_processor import DataProcessor
# Importation des fonctions de backtesting et d’évaluation
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline


# . TÉLÉCHARGEMENT ET NETTOYAGE (Sans fonction bloquante)
s = "2020-11-28"
e = "2023-11-28"
ticker = "ETH-USD"

print(f"Tentative de téléchargement pour {ticker}...")
df_raw = yf.download(ticker, start=s, end=e, progress=False)

if df_raw.empty:
    raise ValueError("Le téléchargement a échoué. Vérifiez votre connexion.")

print(f"Succès ! {len(df_raw)} lignes téléchargées.")

# Création d'une copie de travail
df = df_raw.copy()

# Gestion du MultiIndex (problème fréquent yfinance récent)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Reset index pour avoir la Date en colonne
df = df.reset_index()

# Standardisation des noms de colonnes en minuscules
df.columns = df.columns.str.lower()

# Renommer la colonne date si nécessaire
if 'datetime' in df.columns:
    df = df.rename(columns={'datetime': 'date'})

# Conversion explicite en datetime
df['date'] = pd.to_datetime(df['date'])

# Ajout des colonnes obligatoires pour FinRL
df['tic'] = ticker
df['day'] = df['date'].dt.dayofweek

# Vérification visuelle
print("Aperçu des données brutes :")
print(df.head())

# Affichage du premier graphique
plt.figure(figsize=(15,8))
plt.plot(df['date'], df['close'])
plt.title(f"Cours de {ticker}")
plt.xlabel("Date")
plt.ylabel("Prix ($)")
plt.show()


# Importation de la liste des indicateurs techniques prédéfinis dans FinRL
from finrl.config import INDICATORS

# Initialisation de l’ingénieur de features (feature engineering)
fe = FeatureEngineer(
    use_technical_indicator=True,     # Ajout d’indicateurs techniques
    tech_indicator_list=INDICATORS,   # Liste des indicateurs utilisés
    use_vix=False,                     # Utilisation de l’indice de volatilité (VIX)
    use_turbulence=True,              # Utilisation de l’indicateur de turbulence
    user_defined_feature=False        # Pas de features personnalisées
)

# Prétraitement des données (calcul des indicateurs, nettoyage, etc.)
processed = fe.preprocess_data(df)

# Remplacement des valeurs manquantes par 0
processed.fillna(0, inplace=True)

# Suppression de la colonne volume et affichage des données transformées
processed.drop(columns=['volume']).plot(figsize=(16,8))

# Séparation des données d’entraînement
train = data_split(processed, "2020-11-28", "2022-08-31")
# Séparation des données de trading (test)
trade = data_split(processed, "2022-09-01", "2023-11-28")

print(train.columns)

# Affichage du nombre de lignes du dataset d’entraînement
print(len(train))

# Affichage du nombre de lignes du dataset de trading
print(len(trade))

# Nombre d’actifs uniques (ici ETH seulement)
stock_dimension = len(train.tic.unique())

# Calcul de la taille de l’espace d’état pour le modèle RL
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

# Affichage des dimensions calculées
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# Définition des coûts d’achat (0.1%)
buy_cost_list = sell_cost_list = [0.001] * stock_dimension

# Nombre initial d’actions détenues (0 au départ)
num_stock_shares = [0] * stock_dimension


# Dictionnaire des paramètres de l’environnement de trading
env_kwargs = {
    "hmax": 50,                       # Quantité max achetable/vendable par step
    "initial_amount": 10000,           # Capital initial
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-1             # Mise à l’échelle de la récompense
}

# Création de l’environnement gym pour l’entraînement
e_train_gym = StockTradingEnv(df=train,  **env_kwargs)

# Conversion vers un environnement compatible Stable-Baselines3
env_train, _ = e_train_gym.get_sb_env()

# Affichage du type de l’environnement
print(type(env_train))

# Initialisation de l’agent DRL
agent = DRLAgent(env=env_train)

from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3 import DDPG

n_actions = env_train.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))


policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 128],   # Actor (policy)
        qf=[256, 128]    # Critic (Q-function)
    ),
    activation_fn=torch.nn.ReLU
)

# Création du modèle DDPG
model_ddpg = DDPG(
    policy="MlpPolicy",
    env=env_train,
    action_noise=action_noise,
    learning_rate=0.001,
    buffer_size=50000,
    batch_size=128,
    policy_kwargs=policy_kwargs,
    verbose=1
)

# Entraînement du modèle DDPG

from stable_baselines3.common.callbacks import BaseCallback

class RewardDebugCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            print("Last reward:", self.locals["rewards"])
        return True

trained_ddpg = model_ddpg.learn(
    total_timesteps=50_000,
    tb_log_name="ddpg",
    log_interval=5,
    progress_bar=True,
    callback=RewardDebugCallback()
)


# Création de l’environnement de trading avec seuil de turbulence
e_trade_gym = StockTradingEnv(
    df=trade,
    turbulence_threshold=70,
    risk_indicator_col='turbulence',
    **env_kwargs
)

# Prédiction du modèle entraîné sur la période de trading
df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment=e_trade_gym
)

# Affichage des dernières valeurs du portefeuille
df_account_value.tail()

# Tracé de l’évolution de la valeur du portefeuille
plt.plot(df_account_value.account_value)
plt.show()

# Affichage des premières actions prises par l’agent
df_actions.head()

# Affichage des actions (buy/sell/hold) effectuées
df_actions.actions
