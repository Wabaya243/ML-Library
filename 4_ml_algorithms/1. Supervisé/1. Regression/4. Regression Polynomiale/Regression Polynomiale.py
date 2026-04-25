import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('Ice_cream selling data.csv')
x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, -1].values

# test set e train set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


#Transforme les features en carac polynomiale

poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

#On init un model
reg_model = LinearRegression()
reg_model.fit(x_train_poly, y_train)

y_pred = reg_model.predict(x_test_poly)


new_value = np.array([[-4.0]])  # valeur de x
new_value_poly = poly.transform(new_value)  # transforme en [1, -2, 4]
prediction = reg_model.predict(new_value_poly)
print(f"Prédiction pour x = {new_value} :", prediction)


#l'erreur
mse = mean_squared_error(y_test, y_pred)
print("l'erreur quadratique est : ", np.sqrt(mse))



# Visualiser les résultats
plt.scatter(x, y, color='red', label="Données réelles")

# Générer une grille régulière
x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
x_grid_poly = poly.transform(x_grid)
y_grid_pred = reg_model.predict(x_grid_poly)

# Tracer la courbe polynomiale
plt.plot(x_grid, y_grid_pred, color='blue', label="Régression polynomiale")

plt.title('Ventes de glace vs Température')
plt.xlabel('Température')
plt.ylabel('Ventes de glace')
plt.legend()
plt.show()

