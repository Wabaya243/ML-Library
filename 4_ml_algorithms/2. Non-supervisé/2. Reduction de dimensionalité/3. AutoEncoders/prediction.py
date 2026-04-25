'''
Encodeur comme étape de préparation des données pour un modèle prédictif.

Idée : utiliser l’encodeur déjà entraîné (issu d’un autoencodeur) pour compresser les données d’entrée,
puis entraîner un modèle de classification (ici une régression logistique) sur ces données compressées.

Mais d’abord → établir une BASELINE (point de référence).
Pourquoi ? → Si la régression logistique ne fait pas mieux avec les données compressées qu’avec les données brutes,
alors la compression ne sert à rien.
'''

# --- Baseline avec régression logistique directement sur les données brutes ---

from sklearn.datasets import make_classification   # Génération d’un dataset artificiel
from sklearn.preprocessing import MinMaxScaler     # Mise à l’échelle (0 → 1)
from sklearn.preprocessing import LabelEncoder     # Encodage éventuel de labels (pas nécessaire ici car y est déjà numérique)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Définir un dataset artificiel (1000 exemples, 100 features, dont 10 informatifs et 90 redondants)
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)

# Diviser en jeu d’entraînement (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normaliser les données (important pour les modèles linéaires et les réseaux)
t = MinMaxScaler()
t.fit(X_train)                  # Ajuster sur train
X_train = t.transform(X_train)  # Transformer train
X_test = t.transform(X_test)    # Transformer test

# Définir le modèle de base : régression logistique
model = LogisticRegression()

# Entraîner sur les données brutes
model.fit(X_train, y_train)

# Prédire sur le jeu de test
yhat = model.predict(X_test)

# Calculer la précision
acc = accuracy_score(y_test, yhat)
print(acc)  # Résultat ≈ 0.893 (89.3%)


'''
Résultat : la régression logistique atteint environ 89.3 % de précision.

Hypothèse → En utilisant les données compressées par l’encodeur,
la précision devrait s’améliorer (meilleure extraction de caractéristiques).
'''


# --- Régression logistique avec données encodées (via un autoencodeur) ---

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model  # Pour recharger l’encodeur sauvegardé


# Générer à nouveau le dataset artificiel (mêmes paramètres)
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normalisation
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

# Charger l’encodeur entraîné précédemment (sauvegardé dans un fichier)
encoder = load_model('Save/encoder.keras')


# --- Utilisation de l’encodeur pour transformer les données ---
# L’encodeur réduit 100 colonnes → ex. 50 colonnes (compression, espace latent)
X_train_encode = encoder.predict(X_train)  # Encoder le jeu d’entraînement
X_test_encode = encoder.predict(X_test)    # Encoder le jeu de test

# C’est donc une forme de réduction de dimension / extraction de caractéristiques


# Définir la régression logistique sur les données encodées
model = LogisticRegression(max_iter=200)  
# Pourquoi max_iter=200 ? → Pour éviter les avertissements de convergence.
# En pratique, le modèle converge déjà avant, mais on s’assure qu’il ait assez d’itérations.

# Entraîner sur les données encodées
model.fit(X_train_encode, y_train)

# Prédire sur le jeu de test encodé
yhat = model.predict(X_test_encode)

# Calculer la précision
acc = accuracy_score(y_test, yhat)
print(acc)  # Résultat ≈ 0.93–0.94 (93–94 %)


'''
Résultat : la régression logistique atteint environ 93–94 % de précision 
avec les données encodées (vs 89 % sur données brutes).

Conclusion → L’encodeur extrait des caractéristiques utiles (feature extraction),
et la compression améliore la performance du modèle prédictif.
'''
