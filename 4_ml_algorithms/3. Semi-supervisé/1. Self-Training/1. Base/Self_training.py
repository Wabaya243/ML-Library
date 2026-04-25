# Bibliothèques de base
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.semi_supervised import SelfTrainingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Features:", feature_names)
print("Classes:", target_names)


# Fraction de labels connus
rng = np.random.RandomState(42)
y_semi = y.copy()


# Masquer 70% des labels
mask = rng.rand(len(y)) < 0.7
y_semi[mask] = -1  # -1 indique non étiqueté pour Self-Training

print("Nombre de labels connus:", np.sum(y_semi != -1))
print("Nombre de labels masqués:", np.sum(y_semi == -1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y_semi, test_size=0.3, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## Creation du model self training
base_clf = LogisticRegression(max_iter=1000)


self_training_model = SelfTrainingClassifier(
    base_clf,
    criterion='k_best',  # choisir les k meilleures prédictions à ajouter
    k_best=50,           # ici 50 échantillons à chaque itération
    max_iter=10          # nombre maximum d’itérations
)

self_training_model.fit(X_train, y_train)


y_pred = self_training_model.predict(X_test)

# Matrice de confusion
cm = confusion_matrix(y_test[y_test != -1], y_pred[y_test != -1])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.title("Matrice de confusion")
plt.show()

#Rapport de classification
print(classification_report(y_test[y_test != -1], y_pred[y_test != -1], target_names=target_names))


new_data = np.array([
    [5.1, 3.3, 1.7, 0.5],  # ressemble à setosa
    [6.0, 2.9, 4.5, 1.5],  # ressemble à versicolor
    [6.5, 3.0, 5.5, 2.0]   # ressemble à virginica
])

new_data_scaled = scaler.transform(new_data)
predictions = self_training_model.predict(new_data_scaled)

for i, pred in enumerate(predictions):
    print(f"Exemple {i+1}: classe prédite = {target_names[pred]}")



