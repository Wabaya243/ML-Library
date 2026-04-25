# --- Importation des librairies ---
import os
import numpy as np  # Pourquoi ? → Manipulation de vecteurs et matrices numériques
import pandas as pd  # Pourquoi ? → Manipulation des datasets sous forme de tableaux (CSV)

from numpy.random import seed  # Pourquoi ? → Fixer la graine aléatoire pour reproductibilité
from sklearn.preprocessing import minmax_scale  # Pourquoi ? → Mise à l’échelle des données entre 0 et 1
from sklearn.model_selection import train_test_split  # Pourquoi ? → Diviser le dataset en apprentissage/test

from keras.layers import Input, Dense  # Pourquoi ? → Définir les couches du réseau de neurones
from keras.models import Model         # Pourquoi ? → Construire et entraîner le modèle
from tensorflow.keras.utils import plot_model  # Pourquoi ? → Visualiser l’architecture du modèle


# --- Chargement des données ---
train = pd.read_csv('Data/train.csv')  # Dataset d’apprentissage (avec features + target)
test = pd.read_csv('Data/test.csv')    # Dataset de test (features uniquement)


# --- Séparation des colonnes inutiles ---
target = train['target']  # Variable cible (classification)
train_id = train['ID']    # Identifiant unique (inutile pour l’apprentissage)
test_id = test['ID']

train.drop(['target'], axis=1, inplace=True)  # On retire la target du train (sera ajoutée plus tard)
train.drop(['ID'], axis=1, inplace=True)      # On supprime l’ID car non pertinent
test.drop(['ID'], axis=1, inplace=True)


# --- Normalisation des données ---
train_scaled = minmax_scale(train, axis=0)  # Mise à l’échelle du train entre 0 et 1
test_scaled = minmax_scale(test, axis=0)    # Mise à l’échelle du test entre 0 et 1


# --- Définir le nombre de features ---
ncol = train_scaled.shape[1]  # Nombre de colonnes/features du dataset


# --- Division en apprentissage/validation ---
X_train, X_test, Y_train, Y_test = train_test_split(
    train_scaled, target, train_size=0.9, random_state=seed(2017)
)
# 90% → entraînement, 10% → validation


# --- Taille de la couche "bottleneck" (compression forte) ---
encoding_dim = 200  # On réduit toutes les features à 200 dimensions

input_dim = Input(shape=(ncol,))  # Entrée = toutes les colonnes/features


# --- ENCODEUR (réduction progressive des dimensions) ---
encoded1 = Dense(3000, activation='relu')(input_dim)
encoded2 = Dense(2750, activation='relu')(encoded1)
encoded3 = Dense(2500, activation='relu')(encoded2)
encoded4 = Dense(2250, activation='relu')(encoded3)
encoded5 = Dense(2000, activation='relu')(encoded4)
encoded6 = Dense(1750, activation='relu')(encoded5)
encoded7 = Dense(1500, activation='relu')(encoded6)
encoded8 = Dense(1250, activation='relu')(encoded7)
encoded9 = Dense(1000, activation='relu')(encoded8)
encoded10 = Dense(750, activation='relu')(encoded9)
encoded11 = Dense(500, activation='relu')(encoded10)
encoded12 = Dense(250, activation='relu')(encoded11)
encoded13 = Dense(encoding_dim, activation='relu')(encoded12)
# Ici, on arrive au "bottleneck" → 200 dimensions


# --- DECODEUR (reconstruction progressive) ---
decoded1 = Dense(250, activation='relu')(encoded13)
decoded2 = Dense(500, activation='relu')(decoded1)
decoded3 = Dense(750, activation='relu')(decoded2)
decoded4 = Dense(1000, activation='relu')(decoded3)
decoded5 = Dense(1250, activation='relu')(decoded4)
decoded6 = Dense(1500, activation='relu')(decoded5)
decoded7 = Dense(1750, activation='relu')(decoded6)
decoded8 = Dense(2000, activation='relu')(decoded7)
decoded9 = Dense(2250, activation='relu')(decoded8)
decoded10 = Dense(2500, activation='relu')(decoded9)
decoded11 = Dense(2750, activation='relu')(decoded10)
decoded12 = Dense(3000, activation='relu')(decoded11)
decoded13 = Dense(ncol, activation='sigmoid')(decoded12)
# Sortie finale = reconstruction des données originales


# --- Définition de l’autoencodeur complet ---
autoencoder = Model(inputs=input_dim, outputs=decoded13)


# --- Compilation du modèle ---
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# Optimiseur = Adadelta (stable)
# Perte = Binary Crossentropy (car données normalisées entre 0 et 1)


# --- Résumé du modèle ---
autoencoder.summary()


# --- Entraînement de l’autoencodeur ---
autoencoder.fit(
    X_train, X_train,      # Objectif = reconstruire X_train
    epochs=10,             # Seulement 10 époques (rapide pour test)
    batch_size=32,
    shuffle=False,
    validation_data=(X_test, X_test)
)


# --- Sauvegarde de l’encodeur uniquement ---
encoder = Model(inputs=input_dim, outputs=encoded13)
plot_model(encoder, 'images/encoder_no_compress.png', show_shapes=True)
encoder.save('Save/encoder.keras')


# --- Transformation des données par l’encodeur ---
encoded_train = pd.DataFrame(encoder.predict(train_scaled))
encoded_train = encoded_train.add_prefix('feature_')  # Renommer les colonnes → feature_0, feature_1, ...

encoded_test = encoder.predict(test_scaled, batch_size=256)
encoded_test = encoded_test.add_prefix('feature_')

encoded_train['target'] = target  # Réajout de la target


# --- Vérification ---
print(encoded_train.shape)  # → (nb_lignes, 200 features + 1 target)
print(encoded_train.head())


# --- Sauvegarde des datasets encodés ---
encoded_train.to_csv('train_encoded.csv', index=False)
encoded_test.to_csv('test_encoded.csv', index=False)
