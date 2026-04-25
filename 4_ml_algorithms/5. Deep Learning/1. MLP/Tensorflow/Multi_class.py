import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# 1) Charger le dataset
data = load_iris()
X, y = data.data, data.target

print("Shape X :", X.shape)   # (150,4)
print("Shape y :", y.shape)   # (150,)
print("Classes :", np.unique(y))  # [0,1,2]

# 2) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4) One-hot encoding des labels
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# 5) Construire le modèle
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(48, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')  # multiclasses
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6) Entraînement
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=8,
    verbose=1
)

# 7) Évaluation
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report :\n", classification_report(y_test, y_pred, target_names=data.target_names))

cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :\n", cm)

# 8) Courbes d’apprentissage
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Categorical Crossentropy Loss")
plt.legend()
plt.title("Courbe de perte")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Courbe de précision")

plt.show()




# Exemple de nouvelles données (valeurs arbitraires)
nouvelle_donnee = np.array([[5.1, 3.5, 1.4, 0.2]])

# Normalisation
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)


# Prédiction : probabilités pour chaque classe
pred_probs = model.predict(nouvelle_donnee_scaled)

# Classe prédite
classe_predite = np.argmax(pred_probs, axis=1)[0]

# Nom de la classe
nom_classe = data.target_names[classe_predite]

print("Probabilités :", pred_probs)
print("Classe prédite :", classe_predite, "-", nom_classe)


# Demander à l'utilisateur
long_sep = float(input("Longueur sépale : "))
lar_sep = float(input("Largeur sépale : "))
long_pet = float(input("Longueur pétale : "))
lar_pet = float(input("Largeur pétale : "))

nouvelle_donnee = np.array([[long_sep, lar_sep, long_pet, lar_pet]])
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)
pred_probs = model.predict(nouvelle_donnee_scaled)
classe_predite = np.argmax(pred_probs, axis=1)[0]
nom_classe = data.target_names[classe_predite]

print("Classe prédite :", nom_classe)

