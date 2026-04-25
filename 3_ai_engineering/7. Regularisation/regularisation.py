# On Importe les librairies 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


# Chargement du dataset California Housing
dataset = fetch_california_housing()
x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

# Normalisation
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Séparation en train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.model_selection import GridSearchCV

# Grilles d'hyperparamètres
alphas = np.logspace(-4, 1, 20)

# Ridge
ridge_params = {'alpha': alphas}
grid_ridge = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
grid_ridge.fit(x_train, y_train)
best_ridge = grid_ridge.best_estimator_

# Lasso
lasso_params = {'alpha': alphas}
grid_lasso = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=5, scoring='r2')
grid_lasso.fit(x_train, y_train)
best_lasso = grid_lasso.best_estimator_

# ElasticNet
elastic_params = {
    'alpha': alphas,
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
grid_elastic = GridSearchCV(ElasticNet(max_iter=10000), elastic_params, cv=5, scoring='r2')
grid_elastic.fit(x_train, y_train)
best_elastic = grid_elastic.best_estimator_

model_ridge = best_ridge
model_lasso = best_lasso
model_elastic = best_elastic

print("Best Ridge alpha:", grid_ridge.best_params_)
print("Best Lasso alpha:", grid_lasso.best_params_)
print("Best ElasticNet params:", grid_elastic.best_params_)



#Creation des model
model_lr = LinearRegression()
model_ridge = Ridge(alpha=0.1) # L2 Le modèle garde toutes les variables, mais diminue leur influence 
model_lasso = Lasso(alpha=0.1) # L1 Fait automatiquement de la sélection de variables en mettant les inutiles au poids 0
model_elastic = ElasticNet(alpha=0.1, l1_ratio=0.5) #Melange L1 et L2 ElasticNet gère la sélection de variables ET la colinéarité


#Entrainement
model_lr.fit(x_train, y_train)
model_ridge.fit(x_train, y_train)
model_lasso.fit(x_train, y_train)
model_elastic.fit(x_train, y_train)

# Evalutaion des perfs

models = {
    'LinearReg': model_lr,
    'RIDGE': model_ridge,
    'LASSO': model_lasso,
    'ELASTICNET':model_elastic
    }

for name, model in models.items():
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:\n  MSE = {mse:.4f}, R² = {r2:.4f}\n")
    
    
#Visualisation du resulat    
coef_dataset = pd.DataFrame({
    'Feature': dataset.feature_names,
    'LinearReg': model_lr.coef_,
    'RIDGE': model_ridge.coef_,
    'LASSO': model_lasso.coef_,
    'ELASTICNET': model_elastic.coef_,
    })


coef_dataset.set_index('Feature').plot(kind='bar', figsize=(12,6))
plt.title('Comparaison des coefficients des modèles')
plt.ylabel('Valeur du coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()