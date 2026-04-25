# =============================
# AUTOENCODER pour détection de fraudes
# Dataset : Credit Card Fraud (Kaggle)
# Normalité = transactions légitimes
# Anomalies = transactions frauduleuses
# =============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------
# 1. Chargement du dataset
# -----------------------
# ⚠️ Dataset disponible sur Kaggle : "Credit Card Fraud Detection"
# https://www.kaggle.com/mlg-ulb/creditcardfraud
data = pd.read_csv("creditcard.csv")

print(data.head())

# Variable cible : "Class" (0 = normal, 1 = fraude)
X = data.drop("Class", axis=1).values
y = data["Class"].values

# Normalisation (important pour autoencoder)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------
# 2. Séparation normal / fraude
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# On garde uniquement les normales pour l'entraînement
X_train_norm = X_train[y_train == 0]
X_test_norm = X_test[y_test == 0]
X_test_fraud = X_test[y_test == 1]

print("Train normal :", X_train_norm.shape)
print("Test normal :", X_test_norm.shape)
print("Test fraude :", X_test_fraud.shape)

# -----------------------
# 3. Construction de l’autoencoder
# -----------------------
input_dim = X_train.shape[1]  # nombre de features (~30 dans ce dataset)

# Entrée
input_layer = Input(shape=(input_dim,))

# ENCODEUR
encoded = Dense(20, activation='relu')(input_layer)
encoded = Dense(10, activation='relu')(encoded)
latent = Dense(5, activation='relu')(encoded)  # espace latent

# DECODEUR
decoded = Dense(10, activation='relu')(latent)
decoded = Dense(20, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)  # reconstruction

# Modèle autoencoder
autoencoder = Model(inputs=input_layer, outputs=decoded)
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
# 5. Calcul des erreurs de reconstruction
# -----------------------
reconstructions = autoencoder.predict(X_test_norm)
mse_norm = np.mean(np.power(X_test_norm - reconstructions, 2), axis=1)

# Seuil basé sur la distribution des normales
threshold = np.mean(mse_norm) + 3*np.std(mse_norm)
print("Seuil fixé à :", threshold)

# -----------------------
# 6. Détection des fraudes
# -----------------------
# Normales
recon_norm = autoencoder.predict(X_test_norm)
mse_norm = np.mean(np.power(X_test_norm - recon_norm, 2), axis=1)
pred_norm = mse_norm > threshold
print("Faux positifs sur normales :", np.mean(pred_norm) * 100, "%")

# Fraudes
recon_fraud = autoencoder.predict(X_test_fraud)
mse_fraud = np.mean(np.power(X_test_fraud - recon_fraud, 2), axis=1)
pred_fraud = mse_fraud > threshold
print("Fraudes détectées :", np.mean(pred_fraud) * 100, "%")

# -----------------------
# 7. Visualisation
# -----------------------
plt.hist(mse_norm, bins=50, alpha=0.6, label="Normales")
plt.hist(mse_fraud, bins=50, alpha=0.6, label="Fraudes")
plt.axvline(threshold, color="red", linestyle="--", label="Seuil")
plt.legend()
plt.xlabel("Erreur de reconstruction")
plt.ylabel("Fréquence")
plt.title("Distribution des erreurs - Détection de fraudes")
plt.show()
