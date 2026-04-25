#regression lineaire multiple
import numpy as np	
import matplotlib.pyplot as plt
import pandas as pd


#importer le dataset

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#gerer les variables catégoriques
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
x = x[:, 1:] #on supprime la première colonne pour éviter la colinéarité de dummy variable

#diviser le dataset en training set et test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)

#construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#prédire les résultats du test set
y_pred = regressor.predict(x_test)
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))

from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Evaluation du modèle
from sklearn.metrics import r2_score

# Calcul de R²
r_squared = r2_score(y_test, y_pred)
print("R² :", r_squared)

# Calcul de R² ajusté
n = len(y_test)                 # nombre d'observations
p = x_test.shape[1]            # nombre de variables explicatives (colonnes)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print("R² ajusté :", adjusted_r_squared)
