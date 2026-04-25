# =============================
# AUTOENCODER pour détection d'anomalies
# Dataset : MNIST
# Normalité = chiffre "1"
# Anomalies = autres chiffres
# =============================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# -----------------------
# 1. Chargement du dataset
# -----------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalisation des pixels entre 0 et 1
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

# Aplatir les images (28x28 → vecteur de taille 784)
X_train = X_train.reshape((len(X_train), -1))
X_test = X_test.reshape((len(X_test), -1))

# -----------------------
# 2. Séparation normal / anomalie
# -----------------------
# On garde uniquement les "1" pour l'entraînement
X_train_norm = X_train[y_train == 1]

# Dans le test, on conserve toutes les classes (1 = normal, autres = anomalies)
X_test_norm = X_test[y_test == 1]
X_test_anom = X_test[y_test != 1]

print("Taille train normal :", X_train_norm.shape)
print("Taille test normal :", X_test_norm.shape)
print("Taille test anomalie :", X_test_anom.shape)

# -----------------------
# 3. Construction de l'autoencoder
# -----------------------
input_dim = X_train.shape[1]  # 784

# Couche d'entrée
input_layer = Input(shape=(input_dim,))

# ENCODER (réduit la dimension)
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # couche latente

# DECODER (reconstruit l'entrée à partir de la couche latente)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # sortie reconstruite

# Modèle autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compilation (MSE = mesure la différence reconstruction vs. original)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

autoencoder.summary()

# -----------------------
# 4. Entraînement
# -----------------------
history = autoencoder.fit(
    X_train_norm, X_train_norm,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test_norm, X_test_norm),
    verbose=1
)

# -----------------------
# 5. Reconstruction et seuil
# -----------------------
# On calcule l'erreur de reconstruction sur les normales
reconstructions = autoencoder.predict(X_test_norm)
mse = np.mean(np.power(X_test_norm - reconstructions, 2), axis=1)

# On fixe un seuil basé sur la moyenne + 3*écart-type
threshold = np.mean(mse) + 3*np.std(mse)
print("Seuil de détection fixé à :", threshold)

# -----------------------
# 6. Détection sur données normales et anomalies
# -----------------------
def detect_anomalies(X, seuil):
    recon = autoencoder.predict(X)
    mse = np.mean(np.power(X - recon, 2), axis=1)
    return mse, mse > seuil  # booléen True = anomalie

# Test sur normales
mse_norm, pred_norm = detect_anomalies(X_test_norm, threshold)
print("Taux de fausses anomalies détectées (sur normales) :",
      np.mean(pred_norm) * 100, "%")

# Test sur anomalies
mse_anom, pred_anom = detect_anomalies(X_test_anom, threshold)
print("Taux de vraies anomalies détectées (sur anormales) :",
      np.mean(pred_anom) * 100, "%")

# -----------------------
# 7. Visualisation
# -----------------------
# On affiche quelques exemples
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    # image normale
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test_norm[i].reshape(28, 28), cmap="gray")
    plt.title("Normal")
    plt.axis("off")
    
    # image anormale
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_test_anom[i].reshape(28, 28), cmap="gray")
    plt.title("Anomalie")
    plt.axis("off")
plt.show()
