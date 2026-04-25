# 1. Imports
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2 on Normalise les données

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creer les 3 Model

model_lr = LinearRegression()
model_tree = DecisionTreeRegressor(random_state=42)
model_knn = KNeighborsRegressor(n_neighbors=5)

# 5 Entrainez tous les models

model_lr.fit(X_train, y_train)
model_tree.fit(X_train, y_train)
model_knn.fit(X_train, y_train)

# 6. Prediction indivuduelles
pred_lr = model_lr.predict(X_test)
pred_tree = model_tree.predict(X_test)
pred_knn = model_knn.predict(X_test)

# 7. ensemble simple : Moyenne de prediction 
ensemble_pred = (pred_lr + pred_tree + pred_knn) / 3

# 8. Evaluation

print("Scores individuels:")
print(f"Régression Linéaire - R2: {r2_score(y_test, pred_lr):.4f}")
print(f"Arbre de Décision     - R2: {r2_score(y_test, pred_tree):.4f}")
print(f"KNN                   - R2: {r2_score(y_test, pred_knn):.4f}")

print("\nModèle d'Ensemble (moyenne):")
print(f"R2 Score: {r2_score(y_test, ensemble_pred):.4f}")
print(f"MSE     : {mean_squared_error(y_test, ensemble_pred):.4f}")





# partie 2 La meme chose mais en plus propre

from sklearn.ensemble import VotingRegressor

# 1. Charger les données
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Séparer en train / test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Définir les modèles de base
lr = LinearRegression()
tree = DecisionTreeRegressor(random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)

# 5. Créer le VotingRegressor
voting_reg = VotingRegressor(estimators=[
    ('lr', lr),
    ('tree', tree),
    ('knn', knn)
],
    )

# 6. Entraîner
voting_reg.fit(X_train, y_train)

# 7. Prédictions
y_pred = voting_reg.predict(X_test)

# 8. Évaluer
print(f"VotingRegressor - R2 Score : {r2_score(y_test, y_pred):.4f}")
print(f"VotingRegressor - MSE      : {mean_squared_error(y_test, y_pred):.4f}")
