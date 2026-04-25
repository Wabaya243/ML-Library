import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


# ===========================
# 🔹 Paramètres globaux
# ===========================
RANDOM_STATE = 42          # Graine aléatoire pour reproductibilité
n_samples = 10000          # Nombre d'échantillons dans le dataset
contamination = 0.02       # Proportion d'anomalies (2%) -> aide le modèle à fixer son seuil


# ===========================
# 🔹 Génération d’un dataset synthétique
# ===========================
X, y_binary = make_classification(
    n_samples=n_samples,
    n_features=8,           # 8 features -> assez varié pour tester l'algo
    n_informative=2,        # 2 variables réellement utiles à la séparation
    n_redundant=0,          # pas de features corrélées artificiellement
    n_clusters_per_class=1, # une seule grappe par classe pour simplifier
    weights=[1 - contamination], # 98% normaux, 2% anomalies
    flip_y=0,               # pas de bruit artificiel
    class_sep=2.0,          # séparation entre classes bien marquée
    random_state=RANDOM_STATE
)

# Convention IsolationForest : -1 = anomalie, 1 = normal
y_true = np.where(y_binary == 1, -1, 1)

# Mettre sous forme de DataFrame pour visualiser
cols = [f"feat_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=cols)
df["y_true"] = y_true
print(df.head())  # aperçu des données


# ===========================
# 🔹 Découpage train/test
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true,
    test_size=0.3,          # 30% test
    random_state=RANDOM_STATE,
    stratify=y_true         # conserve la proportion anomalies/normaux
)


# ===========================
# 🔹 Modèle Isolation Forest
# ===========================
model = IsolationForest(
    n_estimators=200,       # Nombre d’arbres (plus = stabilité ↑, coût ↑ ; 200 = bon compromis)
    max_samples='auto',     # Chaque arbre est entraîné sur 256 échantillons aléatoires
                            # (idéal pour isoler vite les anomalies)
    contamination=contamination, # On indique au modèle qu’on s’attend à 2% d’anomalies
    random_state=RANDOM_STATE
)

# Apprentissage
model.fit(X_train)


# ===========================
# 🔹 Prédictions
# ===========================
y_pred = model.predict(X_test)                # -1 = anomalie, 1 = normal
decision_scores = model.decision_function(X_test)  
# Les scores indiquent à quel point un point est "normal" (plus haut = plus normal)


# ===========================
# 🔹 Évaluation
# ===========================
# On évalue uniquement sur les anomalies (y == -1)
prec = precision_score(y_test == -1, y_pred == -1)
rec = recall_score(y_test == -1, y_pred == -1)
f1 = f1_score(y_test == -1, y_pred == -1)

# ROC AUC : mesure la capacité du modèle à bien séparer normal/anomalie
roc_auc = roc_auc_score((y_test == -1).astype(int), -decision_scores)

print("Precision (anomalies):", prec)
print("Recall (anomalies):", rec)
print("F1-score (anomalies):", f1)
print("ROC AUC:", roc_auc)


# ===========================
# 🔹 Matrice de confusion
# ===========================
cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
print("\nMatrice de confusion:")
print(pd.DataFrame(
    cm,
    index=["Vrai normal (1)", "Vrai anomalie (-1)"],
    columns=["Prédit normal (1)", "Prédit anomalie (-1)"]
))


# ===========================
# 🔹 Visualisation 2D
# ===========================
plt.figure(figsize=(8,6))
x_vis = X_test[:, 0]
y_vis = X_test[:, 1]

mask_anom = (y_pred == -1)  # anomalies détectées
mask_norm = (y_pred == 1)   # normaux détectés

plt.scatter(x_vis[mask_norm], y_vis[mask_norm], s=8, label='Normal')
plt.scatter(x_vis[mask_anom], y_vis[mask_anom], s=12, label='Anomalie')
plt.xlabel("feat_1")
plt.ylabel("feat_2")
plt.title("Projection 2D : normal vs anomalie (prédiction)")
plt.legend()
plt.grid(True)
plt.show()


# ===========================
# 🔹 Courbe ROC
# ===========================
fpr, tpr, thresholds = roc_curve((y_test == -1).astype(int), -decision_scores)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label="ROC curve")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (Isolation Forest)")
plt.legend()
plt.grid(True)
plt.show()


# ===========================
# 🔹 Inspection des anomalies les plus "suspectes"
# ===========================
test_df = pd.DataFrame(X_test, columns=cols)
test_df['y_true'] = y_test
test_df['y_pred'] = y_pred
test_df['decision_score'] = decision_scores

# Trier par score croissant → les plus "anormaux" en haut
top_anomalies = test_df.sort_values(by="decision_score").head(20)
print(top_anomalies)
