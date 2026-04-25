"""
Laboratoire : MLP (Perceptron Multi-couches) pour une tâche de régression
Framework choisi : TensorFlow / Keras
Dataset choisi : diabetes (sklearn.datasets.load_diabetes)
But : construire, entraîner et évaluer un MLP pour prédire une valeur continue (régression).

Contenu :
1) Préparation des données (chargement, normalisation, split)
2) Construction du modèle avec tf.keras.Sequential
3) Compilation (loss = MSE, optimizer = Adam)
4) Entraînement et affichage de la courbe de perte
5) Évaluation finale (MSE, R²)
"""

# ---------------------------
# 1) Imports et configuration
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Fixer la graine pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 2) Chargement et préparation des données
# ---------------------------
# Charger le dataset diabetes (régression)
X, y = load_diabetes(return_X_y=True)

# Reshape y pour correspondre au format attendu par Keras
y = y.reshape(-1, 1)

'''
👉 Donc en résumé :

(442,) = vecteur 1D → valeurs de sortie alignées.

(442, 1) = matrice colonne → chaque sortie est bien identifiée comme une cible.

C’est purement une question de forme des données, pas de contenu.
'''

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation (standardisation)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 3) Construction du modèle MLP avec Keras
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation= 'relu'),
    tf.keras.layers.Dense(1)  # sortie unique pour la régression
])

# ---------------------------
# 4) Compilation du modèle
# ---------------------------
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

# ---------------------------
# 5) Entraînement
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    verbose=2
)

# ---------------------------
# 6) Évaluation finale
# ---------------------------
# Prédictions
preds = model.predict(X_test)

# Calcul métriques
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nRésultats sur le jeu de test :")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ---------------------------
# 7) Visualisation des courbes d'apprentissage
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='train loss (MSE)')
plt.plot(history.history['val_loss'], label='val loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Courbe d\'apprentissage du MLP (régression)')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 7) Prédictions sur de nouvelles données fournies par l’utilisateur
# ---------------------------
print("\n--- Prédictions sur nouvelles données ---")


# Exemple : l’utilisateur entre 10 valeurs correspondant aux features du dataset
# Ici on simule avec un input factice, mais on peut utiliser input() pour saisie réelle
nouvelle_donnee = np.array([[0.05, -0.02, 0.03, -0.01, 0.04, 0.02, -0.03, 0.01, -0.02, 0.03]])


# Normaliser avec le scaler appris sur X_train
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)


# Prédire avec le modèle
prediction = model.predict(nouvelle_donnee_scaled)
print("Prédiction pour la nouvelle donnée :", prediction.flatten()[0])


# Exemple de nouvelle donnée (30 valeurs)
# ⚠️ Ici je mets des valeurs fictives, l'utilisateur doit entrer de vraies valeurs
nouvelle_donnee = np.array([14.5, 20.0, 95.0, 550.0, 0.10, 0.15, 0.05, 0.06, 0.18, 0.06,
                            0.25, 1.2, 2.0, 20.0, 0.005, 0.02, 0.02, 0.01, 0.02, 0.003,
                            16.0, 25.0, 105.0, 800.0, 0.12, 0.22, 0.07, 0.08, 0.30, 0.08])

# Reshape en (1, 30)
nouvelle_donnee = nouvelle_donnee.reshape(1, -1)

# Normaliser avec le même scaler appris avant
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)

# Prédire
proba = model.predict(nouvelle_donnee_scaled)[0][0]
classe = int(proba > 0.5)

print(f"Probabilité d'être malin : {proba:.4f}")
print("Classe prédite :", classe)



### pour demander a l'utilisateur de saisir 

valeurs = []
for i in range(X_train.shape[1]):
    val = float(input(f"Entrer la valeur de la feature {i+1}: "))
    valeurs.append(val)

nouvelle_donnee = np.array(valeurs).reshape(1, -1)
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)

proba = model.predict(nouvelle_donnee_scaled)[0][0]
classe = int(proba > 0.5)

print(f"Probabilité d'être malin : {proba:.4f}")
print("Classe prédite :", "Malin (1)" if classe==1 else "Bénin (0)")










