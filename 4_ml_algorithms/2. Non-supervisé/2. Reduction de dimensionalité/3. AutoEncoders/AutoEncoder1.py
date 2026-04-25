# Importation des librairies nécessaires
import numpy as np                  # Pourquoi ? → Pour la manipulation de tableaux numériques et calculs
import pandas as pd                 # Pourquoi ? → Pour la manipulation de données tabulaires
import matplotlib.pyplot as plt     # Pourquoi ? → Pour visualiser les résultats (courbes, images, etc.)
import seaborn as sns               # Pourquoi ? → Pour des graphiques plus esthétiques et avancés
from sklearn.datasets import make_classification  # Pourquoi ? → Pour générer un dataset artificiel supervisé


# Générer un dataset artificiel de classification
X, y = make_classification(
    n_samples=1000,     # Pourquoi ? → 1000 échantillons
    n_features=100,     # Pourquoi ? → Chaque échantillon a 100 variables (features)
    n_informative=10,   # Pourquoi ? → Seulement 10 variables contiennent une vraie information discriminante
    n_redundant=90,     # Pourquoi ? → 90 variables sont redondantes (combinées des variables utiles)
    random_state=1      # Pourquoi ? → Pour avoir un résultat reproductible
)

# Afficher les dimensions des données (1000, 100) et des labels (1000,)
print(X.shape, y.shape)

# Enregistrer le nombre de colonnes (features), utile pour définir l’architecture du réseau
n_inputs = X.shape[1]

# Afficher les 5 premières étiquettes (classes) pour vérifier les données
print(y[:5])

## Avant de définir et entraîner le modèle,
## on divise les données en ensembles d’entraînement et de test,
## puis on normalise les valeurs entre 0 et 1.
## Pourquoi ? → La normalisation améliore l’entraînement des réseaux de neurones (MLP).

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Division du dataset en apprentissage (67%) et test (33%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Mise à l’échelle (scaling) des données entre 0 et 1
t = MinMaxScaler()
t.fit(X_train)               # Pourquoi ? → Apprendre les min/max sur l’ensemble d’entraînement
X_train = t.transform(X_train)  # Normaliser les données d’entraînement
X_test = t.transform(X_test)    # Normaliser les données de test avec les mêmes paramètres


'''
Nous allons définir l’encodeur (encoder) :
- 1ère couche cachée : deux fois plus grande que les entrées (200 neurones)
- 2ème couche cachée : même taille que les entrées (100 neurones)
- Couche "bottleneck" : même taille que les entrées (100), espace latent réduit
On utilise Batch Normalization + LeakyReLU pour stabiliser et améliorer l’apprentissage.
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization

# Définir l’entrée du modèle (nombre de colonnes = n_inputs)
visible = Input(shape=(n_inputs,))

# --- ENCODEUR ---

# Couche cachée 1 (2 * n_inputs)
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)  # Pourquoi ? → Normalise les activations pour stabiliser l’apprentissage
e = LeakyReLU()(e)           # Pourquoi ? → Fonction d’activation qui évite le problème du "ReLU mort"

# Couche cachée 2 (n_inputs)
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

# Couche "goulot d’étranglement" (bottleneck, espace latent)
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)


'''
Nous allons définir le décodeur (decoder), structure inversée :
- 1ère couche cachée : même taille que les entrées (100 neurones)
- 2ème couche cachée : deux fois la taille des entrées (200 neurones)
- Couche de sortie : même taille que les entrées (100) avec activation linéaire
Pourquoi ? → Pour reconstruire fidèlement les données originales.
'''

# --- DECODEUR ---

# Couche cachée 1 (n_inputs)
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# Couche cachée 2 (2 * n_inputs)
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# Couche de sortie (reconstruction des features)
output = Dense(n_inputs, activation='linear')(d)

# Définition du modèle autoencodeur complet
model = Model(inputs=visible, outputs=output)


'''
Compilation du modèle :
- Optimiseur : Adam (version améliorée du SGD, plus rapide et efficace)
- Perte : MSE (erreur quadratique moyenne), adaptée car la reconstruction est une régression multi-sorties
'''
model.compile(optimizer="adam", loss='mse')


# Visualiser l’architecture de l’autoencodeur
from tensorflow.keras.utils import plot_model
plot_model(model, 'images/autoencoder_no_compress.png', show_shapes=True)


'''
Entraînement de l’autoencodeur :
- Objectif : reproduire les données d’entrée (X_train → X_train)
- Nombre d’époques : 200
- Taille du batch : 16
- Validation : on suit la performance sur l’ensemble de test
'''
history = model.fit(X_train, X_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_test, X_test))


'''
Après entraînement, on visualise la courbe de perte (loss) :
- courbe d’apprentissage (train)
- courbe de validation (test)
Pourquoi ? → Pour vérifier si le modèle apprend correctement et s’il y a du surapprentissage.
'''
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


'''
Enfin, on sauvegarde uniquement l’encodeur (encoder) :
Pourquoi ? → Dans la vraie vie, on n’a besoin que de l’encodeur pour réduire la dimension des données,
pas du décodeur.
'''
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'images/encoder_no_compress.png', show_shapes=True)

# Sauvegarde de l’encodeur sur disque
encoder.save('Save/encoder.keras')
