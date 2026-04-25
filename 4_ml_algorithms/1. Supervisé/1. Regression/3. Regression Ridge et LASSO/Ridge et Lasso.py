import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


#importer le dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Gerer les variables categoriel
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
x = x[:, 1:] # on del la premiere colone pour eviter la colineratité des dummy variables

# Train set et Test Set
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.30, random_state=42)


#construction du model 
model_ridge = Ridge(alpha=1.0) # Alpha controle la force de la regularisation
model_lasso = Lasso(alpha=0.1)

model_lasso.fit(x_train, y_train)
model_ridge.fit(x_train, y_train)

#predire les resultats des 2 
y_pred_lasso = model_lasso.predict(x_test)
model_lasso.predict(np.array([[1, 0, 130000, 140000, 300000]]))

y_pred_ridge = model_ridge.predict(x_test)
model_lasso.predict(np.array([[1, 0, 130000, 140000, 300000]]))

#afficher l'erreur quadratique
print(np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))


#Evaluation du model
r_squared_lasso = r2_score(y_test, y_pred_lasso)
print("R² :", r_squared_lasso)

r_squared_ridge = r2_score(y_test, y_pred_ridge)
print("R² :", r_squared_ridge)


# Calcul de R² ajusté
### Lasso ###

n = len(y_test)                 # nombre d'observations
p = x_test.shape[1]            # nombre de variables explicatives (colonnes)
adjusted_r_squared_lasso = 1 - (1 - r_squared_lasso) * (n - 1) / (n - p - 1)
print("R² ajusté de lasso :", adjusted_r_squared_lasso)


#Ridge 
adjusted_r_squared_ridge = 1 - (1 - r_squared_ridge) * (n - 1) / (n - p - 1)
print("R² ajusté de ridge :", adjusted_r_squared_ridge)


