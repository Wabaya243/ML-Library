# Bibliothèques de base
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.semi_supervised import SelfTrainingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print("Features:", feature_names)
print("Classes:", target_names)


## Création d’un scénario semi-supervisé

rng = np.random.RandomState(42)
y_semi = y.copy()
mask = rng.rand(len(y)) < 0.7
y_semi[mask] = -1

print("Labels connus:", np.sum(y_semi != -1))
print("Labels masqués:", np.sum(y_semi == -1))

## Division train/test et standardisation

X_train, X_test, y_train, y_test = train_test_split(
    X, y_semi, test_size=0.3, random_state=42, stratify=y_semi
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Self-Training avec RandomForest

from sklearn.ensemble import RandomForestClassifier

base_clf = RandomForestClassifier(n_estimators=100, random_state=42)

self_training_model = SelfTrainingClassifier(
    base_clf,
    criterion='threshold',  # ici on ajoute labels dont la confiance > threshold
    threshold=0.9,          # confiance minimum de 90%
    max_iter=20
)

self_training_model.fit(X_train, y_train)

#Visualisation de l’évolution des labels ajoutés

added_counts = self_training_model.transduction_  # labels finaux
print("Nombre final de labels connus :", np.sum(added_counts != -1))


# Indices des données qui ont été ajoutées
added_indices = np.where((y_train == -1) & (self_training_model.transduction_ != -1))[0]

print("Indices des labels ajoutés :", added_indices)
print("Labels ajoutés :", self_training_model.transduction_[added_indices])

## evaluation avancé

from sklearn.metrics import accuracy_score, f1_score

y_pred = self_training_model.predict(X_test[y_test != -1])

print("Accuracy:", accuracy_score(y_test[y_test != -1], y_pred))
print("F1-score:", f1_score(y_test[y_test != -1], y_pred, average='macro'))
print(classification_report(y_test[y_test != -1], y_pred, target_names=target_names))

cm = confusion_matrix(y_test[y_test != -1], y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.title("Matrice de confusion - Self-Training Wine dataset")
plt.show()


### Test sur nouvelles données

# Exemples fictifs
new_data = np.array([
    [13.5, 2.5, 2.5, 20.0, 100.0, 2.0, 2.0, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0],
    [12.0, 3.0, 2.0, 18.0, 110.0, 2.5, 2.0, 0.4, 1.2, 4.0, 1.2, 3.2, 800.0]
])

new_data_scaled = scaler.transform(new_data)
predictions = self_training_model.predict(new_data_scaled)

for i, pred in enumerate(predictions):
    print(f"Exemple {i+1}: classe prédite = {target_names[pred]}")




