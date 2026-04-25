import numpy as np	
import matplotlib.pyplot as plt
import pandas as pd


#importer le dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#diviser le dataset en training set et test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 1.0/3, random_state = 0)

#construction du modèle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#prédire les résultats du test set
y_pred = regressor.predict(x_test) #prédire les résultats du test set
regressor.predict([[15]]) #prédire le salaire pour 15 ans d'expérience

#visualiser les résultats
plt.scatter(x_test, y_test, color = 'red')  #points de test
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #ligne de régression
plt.title('Salaire vs Expérience (Test set)')
plt.xlabel('Expérience')
plt.ylabel('Salaire')
plt.show()


