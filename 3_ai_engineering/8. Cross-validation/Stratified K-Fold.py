# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

# Étape 1 : Chargement du jeu de données
print("📥 Chargement des données...")
dataset = pd.read_csv("creditcard.csv")

# Étape 2 : Séparation des variables explicatives (X) et de la cible (y)
X = dataset.drop("Class", axis=1)
y = dataset["Class"]

# Étape 3 : Normalisation de la colonne 'Amount' (montant de la transaction)
# Pourquoi ? Pour éviter que l’échelle de cette variable n’influence le modèle
X["Amount"] = StandardScaler().fit_transform(X["Amount"].values.reshape(-1, 1))

# Étape 4 : Division du jeu de données en jeu d'entraînement et de test
# 80% pour l'entraînement (utilisé avec la validation croisée), 20% pour le test final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Étape 5 : Définition du modèle de classification
model_RF = RandomForestClassifier(n_estimators=100, random_state=42)

# Étape 6 : Validation croisée avec stratification (5 folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracies = []
cv_aucs = []

print("\n🔄 Démarrage de la validation croisée...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n📁 Fold {fold + 1} / 5")

    # Création des sous-ensembles pour ce fold
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Entraînement du modèle sur les données de ce fold
    model_RF.fit(X_tr, y_tr)

    # Prédiction
    y_val_pred = model_RF.predict(X_val)
    y_val_proba = model_RF.predict_proba(X_val)[:, 1]  # Probabilité pour la classe 1 (fraude)

    # Évaluation
    acc = accuracy_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)

    print(f"✅ Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    cv_accuracies.append(acc)
    cv_aucs.append(auc)

# Étape 7 : Résumé des scores de la validation croisée
print("\n📊 Résumé de la validation croisée :")
print(f"🔹 Moyenne Accuracy : {np.mean(cv_accuracies):.4f}")
print(f"🔹 Moyenne ROC AUC  : {np.mean(cv_aucs):.4f}")

# Étape 8 : Réentraînement sur l’ensemble du jeu d'entraînement
model_RF.fit(X_train, y_train)

# Étape 9 : Évaluation finale sur le jeu de test
print("\n📈 Évaluation finale sur le jeu de test :")
y_test_pred = model_RF.predict(X_test)
y_test_proba = model_RF.predict_proba(X_test)[:, 1]

print("📋 Rapport de classification :")
print(classification_report(y_test, y_test_pred))

print("📌 Matrice de confusion :")
print(confusion_matrix(y_test, y_test_pred))

print(f"🏁 ROC-AUC Score : {roc_auc_score(y_test, y_test_proba):.4f}")

# Étape 10 : Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_test_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.title("🎯 Courbe ROC - Random Forest sur le jeu de test")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.legend()
plt.grid(True)
plt.show()
