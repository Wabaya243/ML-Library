import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


# ===========================
# 🔹 Paramètres globaux
# ===========================
RANDOM_STATE = 42
n_samples = 5000         # dataset plus petit que précédemment
contamination = 0.02     # 2% anomalies attendues
n_neighbors = 20         # nb de voisins pour mesurer la densité locale


# ===========================
# 🔹 Génération du dataset synthétique
# ===========================
X, y_binary = make_classification(
    n_samples=n_samples,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[1 - contamination],
    flip_y=0,
    class_sep=2.0,
    random_state=RANDOM_STATE
)

# Convention LOF (comme IsolationForest) : -1 = anomalie, 1 = normal
y_true = np.where(y_binary == 1, -1, 1)

# Découpage train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_true
)


# ===========================
# 🔹 Modèle Local Outlier Factor
# ===========================
# LOF fonctionne en "fit_predict" (pas de predict séparé comme IF)
lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,     # nb de voisins utilisés pour estimer la densité locale
    contamination=contamination, # % anomalies estimé
    novelty=True                 # permet d’utiliser .fit() + .predict() (mode détection "nouveauté")
)

# Entraînement sur le train set
lof.fit(X_train)

# Prédiction sur le test set
y_pred = lof.predict(X_test)  # -1 = anomalie, 1 = normal
decision_scores = -lof.decision_function(X_test)
# ⚠️ ici les scores sont inversés : plus grand = plus anormal


# ===========================
# 🔹 Évaluation
# ===========================
prec = precision_score(y_test == -1, y_pred == -1)
rec = recall_score(y_test == -1, y_pred == -1)
f1 = f1_score(y_test == -1, y_pred == -1)
roc_auc = roc_auc_score((y_test == -1).astype(int), decision_scores)

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

mask_anom = (y_pred == -1)
mask_norm = (y_pred == 1)

plt.scatter(x_vis[mask_norm], y_vis[mask_norm], s=8, label='Normal')
plt.scatter(x_vis[mask_anom], y_vis[mask_anom], s=12, label='Anomalie', color='red')
plt.xlabel("feat_1")
plt.ylabel("feat_2")
plt.title("Projection 2D : normal vs anomalie (LOF)")
plt.legend()
plt.grid(True)
plt.show()


# ===========================
# 🔹 Courbe ROC
# ===========================
fpr, tpr, thresholds = roc_curve((y_test == -1).astype(int), decision_scores)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label="ROC curve")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (LOF)")
plt.legend()
plt.grid(True)
plt.show()


# ===========================
# 🔹 Inspection des anomalies les plus suspectes
# ===========================
test_df = pd.DataFrame(X_test, columns=[f"feat_{i+1}" for i in range(X.shape[1])])
test_df['y_true'] = y_test
test_df['y_pred'] = y_pred
test_df['decision_score'] = decision_scores

# Trier par score décroissant (les plus anormaux en haut)
top_anomalies = test_df.sort_values(by="decision_score", ascending=False).head(20)
print(top_anomalies)
