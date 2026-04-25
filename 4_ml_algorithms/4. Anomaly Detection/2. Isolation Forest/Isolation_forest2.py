# =========================
# 1. Import des librairies
# =========================
import numpy as np                  # Pour les calculs numériques
import pandas as pd                 # Pour la manipulation de données
import matplotlib.dates as mdates   # Pour gérer les dates sur les graphiques
import matplotlib.pyplot as plt     # Pour les graphiques
from sklearn.ensemble import IsolationForest   # Pour la détection d’anomalies
from sklearn import tree            # Pour visualiser l’arbre de décision
import warnings                     # Pour ignorer les warnings

# On ignore les avertissements pour ne pas "polluer" la sortie
warnings.filterwarnings("ignore")


# =========================
# 2. Chargement des données
# =========================
# Lecture du fichier CSV et conversion de la colonne "timestamp" en datetime
data = pd.read_csv("Data/nyc_taxi.csv", parse_dates=['timestamp'])

# Quelques infos sur les données
print(data.describe())          # Statistiques descriptives
print(data.isnull().sum())      # Vérification des valeurs manquantes

print('Temps de départ : ', data['timestamp'].min())   # Date minimale
print('Temps de fin    : ', data['timestamp'].max())   # Date maximale
print('Durée totale    : ', data['timestamp'].max() - data['timestamp'].min())


# =========================
# 3. Visualisations temporelles (Horaire, Journalier, Hebdomadaire)
# =========================
# On met la colonne "timestamp" comme index (utile pour resampler par date)
df = data.set_index('timestamp')

# --- Horaire ---
df_hourly = df.resample('H').mean()
plt.figure(figsize=(12,4))
plt.plot(df_hourly.index, df_hourly['value'], color="blue")
plt.title("NYC Taxi - Horaire")
plt.ylabel("Nombre de passagers")
plt.xlabel("Date")
plt.grid(True)
plt.show()

# --- Journalier ---
df_daily = df.resample('D').mean()
plt.figure(figsize=(12,4))
plt.plot(df_daily.index, df_daily['value'], color="green")
plt.title("NYC Taxi - Journalier")
plt.ylabel("Nombre de passagers")
plt.xlabel("Date")
plt.grid(True)
plt.show()

# --- Hebdomadaire ---
df_weekly = df.resample('W').mean()
plt.figure(figsize=(12,4))
plt.plot(df_weekly.index, df_weekly['value'], color="orange")
plt.title("NYC Taxi - Hebdomadaire")
plt.ylabel("Nombre de passagers")
plt.xlabel("Date")
plt.grid(True)
plt.show()


# =========================
# 4. Split en train/test
# =========================
# On prend la moyenne journalière pour simplifier l'analyse
df_model1 = df.resample('D').mean().reset_index()

# Split : 65% pour train, 35% pour test
df_train_split1, df_test_split1 = np.split(df_model1, [int(0.65 * len(df_model1))])


# =========================
# 5. Distribution des valeurs (train)
# =========================
df_visualize = df.resample('D').mean()
df_visualize_train, _ = np.split(df_visualize, [int(0.65 * len(df_visualize))])

plt.figure(figsize=(8,4))
plt.hist(df_visualize_train['value'], bins=50, density=True, alpha=0.7, color='blue')
plt.title("Distribution des valeurs (Train)")
plt.xlabel("Valeur")
plt.ylabel("Densité")
plt.grid(True)
plt.show()


# =========================
# 6. Isolation Forest (détection anomalies)
# =========================
# On définit la colonne "value" comme seule variable d'entrée
features = ['value']
df_train1 = df_train_split1[features]
df_test1 = df_test_split1[features]

# Modèle Isolation Forest
model = IsolationForest(random_state=0, contamination=0.03)
model.fit(df_train1)

# Prédiction sur train
outliers_train = pd.Series(model.predict(df_train1)).apply(lambda x: 1 if x == -1 else 0).to_numpy()
anomaly_score_train = model.decision_function(df_train1)

# Ajout des résultats dans le dataframe train
df_train_split1 = df_train_split1.assign(outliers=outliers_train,
                                         anomaly_score=anomaly_score_train)

print("Nombre d'outliers détectés (train) :", df_train_split1['outliers'].sum())


# =========================
# 7. Visualisation anomalies sur train
# =========================
plt.figure(figsize=(12,5))
# Courbe normale
plt.plot(df_train_split1['timestamp'], df_train_split1['value'], 
         label="Valeurs normales", color='green')
# Points d’anomalie en rouge
plt.scatter(df_train_split1.loc[df_train_split1['outliers']==1, 'timestamp'],
            df_train_split1.loc[df_train_split1['outliers']==1, 'value'],
            color='red', label="Anomalies")
plt.title("NYC Taxi - Anomalies détectées (Train)")
plt.xlabel("Date")
plt.ylabel("Nombre de passagers")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 8. Histogramme des scores d’anomalie
# =========================
plt.figure(figsize=(8,4))
plt.hist(anomaly_score_train, bins=50, color='purple', alpha=0.7)
plt.title("Distribution des scores d'anomalie (Train)")
plt.xlabel("Anomaly Score")
plt.ylabel("Fréquence")
plt.grid(True)
plt.show()


# =========================
# 9. Anomalies sur le set de test
# =========================
outliers_test = pd.Series(model.predict(df_test1)).apply(lambda x: 1 if x == -1 else 0).to_numpy()
anomaly_score_test = model.decision_function(df_test1)

# Ajout au dataframe test
df_test_split1 = df_test_split1.assign(outliers=outliers_test,
                                       anomaly_score=anomaly_score_test)

# Visualisation anomalies
plt.figure(figsize=(12,5))
plt.plot(df_test_split1['timestamp'], df_test_split1['value'], 
         label="Valeurs normales", color='green')
plt.scatter(df_test_split1.loc[df_test_split1['outliers']==1, 'timestamp'],
            df_test_split1.loc[df_test_split1['outliers']==1, 'value'],
            color='red', label="Anomalies")
plt.title("NYC Taxi - Anomalies détectées (Test)")
plt.xlabel("Date")
plt.ylabel("Nombre de passagers")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 10. Visualisation d’un arbre de décision interne du modèle
# =========================
fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
tree.plot_tree(model.estimators_[0],       # Premier arbre de la forêt
               feature_names=df_train1.columns, 
               max_depth=2,                # On limite la profondeur pour lisibilité
               filled=True,                # Coloration des noeuds
               ax=ax)
plt.show()
