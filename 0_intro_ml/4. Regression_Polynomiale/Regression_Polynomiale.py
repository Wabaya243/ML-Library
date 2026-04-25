#regression polynomial
import numpy as np	
import matplotlib.pyplot as plt
import pandas as pd


#importer le dataset

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values



#construction du modèle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly,y)

#prédire les résultats du test set
y_pred = regressor.predict(x) #prédire les résultats du test set
regressor.predict([[15]]) #prédire le salaire pour 15 ans d'expérience

#visualiser les résultats
plt.scatter(x, y, color = 'red')  #points de test
plt.plot(x, regressor.predict(x_poly), color = 'blue') #ligne de régression
plt.title('Salaire vs Expérience (Test set)')
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.show()


