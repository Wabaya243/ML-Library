# Imports
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Charger les données California Housing
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Définir les modèles de base
lr = LinearRegression()
tree = DecisionTreeRegressor(random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)

# 5. Voting Regressor (moyenne)
voting_reg = VotingRegressor(
    estimators=[('lr', lr), ('tree', tree), ('knn', knn)]
)

# 6. Stacking Regressor (méta-modèle = régression linéaire)
stacking_reg = StackingRegressor(
    estimators=[('lr', lr), ('tree', tree), ('knn', knn)],
    final_estimator=LinearRegression(),
    cv=5
)

# 7. Mettre tous les modèles dans un dictionnaire
models = {
    "LinearRegression": lr,
    "DecisionTree": tree,
    "KNN": knn,
    "VotingRegressor": voting_reg,
    "StackingRegressor": stacking_reg
}

# 8. Entraînement et évaluation
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Modèle": name,
        "R2 Score": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred)
    })

# 9. Résultats dans un DataFrame
df_results = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print("\n Résultats comparatifs :")
print(df_results)
