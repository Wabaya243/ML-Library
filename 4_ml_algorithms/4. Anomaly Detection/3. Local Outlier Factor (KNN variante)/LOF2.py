import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Fixer la graine aléatoire pour avoir des résultats reproductibles
np.random.seed(42)

# Création d'une grille de points (xx, yy) pour tracer la frontière de décision
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# Générer des données normales (non anormales) pour l'entraînement
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]  # on déplace les points pour former deux "nuages"

# Générer de nouvelles données normales (non anormales) pour le test
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# Générer des données anormales (outliers) uniformément réparties
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# Créer et entraîner le modèle LOF en mode "novelty detection"
# - n_neighbors = 20 : nombre de voisins considérés
# - novelty=True : on active la détection de nouveautés (pas seulement fit_predict)
# - contamination=0.1 : proportion attendue d’anomalies
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)

# ⚠️ Ne pas utiliser predict/decision_function sur X_train car il a servi à l'entraînement
# On applique uniquement sur de nouvelles données
y_pred_test = clf.predict(X_test)           # prédiction sur les nouvelles données normales
y_pred_outliers = clf.predict(X_outliers)   # prédiction sur les anomalies générées
n_error_test = y_pred_test[y_pred_test == -1].size   # erreurs sur données normales
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size  # erreurs sur anomalies

# Calcul de la fonction de décision sur toute la grille (xx, yy)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# === TRACÉ GRAPHIQUE ===
plt.title("Détection de nouveautés avec LOF")

# Affichage de la frontière de décision (zone normale = bleu clair, zone anormale = rouge clair)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
frontiere = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

# Taille des points
s = 40

# Nuage d'apprentissage (blanc avec bord noir)
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')

# Nouvelles données normales (violet)
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')

# Données anormales (jaune)
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')

# Réglage des axes
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

# Légende : on prend le premier élément de la frontière pour représenter la courbe
plt.legend(
    [frontiere.collections[0], b1, b2, c],
    ["Frontière apprise", "Observations d'entraînement",
     "Nouvelles normales", "Nouvelles anormales"],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11)
)

# Texte sous le graphique indiquant les erreurs détectées
plt.xlabel(
    "Erreurs sur normales: %d/40 ; Erreurs sur anormales: %d/40"
    % (n_error_test, n_error_outliers)
)

# Affichage du graphe
plt.show()
