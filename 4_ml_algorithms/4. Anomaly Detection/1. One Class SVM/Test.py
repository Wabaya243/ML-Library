# Importation des librairies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from matplotlib.lines import Line2D  # Pour créer une légende custom

# Générer quelques points des données
X = 0.3 * np.random.rand(100, 2)
X_train = np.r_[X + 2, X - 2]

# Générer les outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20,2))

# Entraîner et créer le modèle
classifier = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
classifier.fit(X_train)

# Créer un mesh grid pour le plot
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# Obtenir les valeurs de la fonction de décision
Z = Z.reshape(xx.shape)

# Dessiner la frontière des décisions
plt.figure(figsize=(12,8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

# Dessiner les points des données
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors='k', marker="o")
b2 = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors='k', marker="o")

plt.axis("tight")

# Créer une légende propre
frontier_line = Line2D([0], [0], color='darkred', linewidth=2)
plt.legend([frontier_line, b1, b2],
           ['frontière apprise', "observations d\'entrainement", "outliers"],
           fontsize=11)

plt.title("One-Class SVM Anomaly detection")
plt.show()
