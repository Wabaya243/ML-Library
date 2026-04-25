import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

# ---------------------------
# 1️⃣ Charger MNIST
# ---------------------------
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype('float32') / 255.0  # Normalisation entre 0 et 1
X_test = X_test.astype('float32') / 255.0

# Ajouter un canal pour CNN
X_train_full = np.expand_dims(X_train_full, -1)  # Transformer (n,28,28) → (n,28,28,1) pour CNN
X_test = np.expand_dims(X_test, -1)

# ---------------------------
# 2️⃣ Créer pool étiqueté / non-étiqueté
# ---------------------------
labeled_size = int(0.1 * X_train_full.shape[0])  # 10% des données étiquetées pour simuler semi-supervisé
X_labeled = X_train_full[:labeled_size]          # pool initial étiqueté
y_labeled = y_train_full[:labeled_size]
X_unlabeled = X_train_full[labeled_size:]        # pool non-étiqueté
y_unlabeled = y_train_full[labeled_size:]        # utilisé uniquement pour suivi des performances

# One-hot pour Keras
y_labeled_cat = to_categorical(y_labeled, 10)   # Convertir les labels en vecteurs one-hot
y_test_cat = to_categorical(y_test, 10)

# ---------------------------
# 3️⃣ Définir un CNN simple
# ---------------------------
def create_cnn():
    # CNN pour extraire automatiquement les features spatiales locales (coins, lignes, textures)
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),  # Premier filtre 3x3
        layers.Conv2D(64, 3, activation='relu'),                          # Deuxième filtre 3x3
        layers.MaxPooling2D(),                                            # Réduction dimension + invariance translation
        layers.Flatten(),                                                 # Aplatir en vecteur
        layers.Dense(128, activation='relu'),                             # Couche dense pour combiner features
        layers.Dropout(0.25),                                             # Dropout pour éviter overfitting
        layers.Dense(10, activation='softmax')                            # 10 classes de sortie
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn()

# ---------------------------
# 4️⃣ Boucle de Self-training
# ---------------------------
num_iterations = 5
num_add = 500  # nombre d'exemples les plus confiants à ajouter à chaque itération

for it in range(num_iterations):
    # 4a) Entraîner sur les données étiquetées
    # Le modèle apprend d'abord avec le petit pool étiqueté
    model.fit(X_labeled, y_labeled_cat, epochs=3, batch_size=64, verbose=0)
    
    # 4b) Prédire sur les non-étiquetés
    # Générer les pseudo-labels et la confiance associée
    probs = model.predict(X_unlabeled, verbose=0)       # probabilités pour chaque classe
    conf = np.max(probs, axis=1)                        # confiance max pour chaque exemple
    preds = np.argmax(probs, axis=1)                    # label prédit (pseudo-label)
    
    # 4c) Sélectionner les exemples les plus confiants
    top_idx = conf.argsort()[-num_add:]                 # indices des exemples les plus sûrs
    
    # 4d) Ajouter au pool étiqueté
    # Ces exemples deviennent "étiquetés" avec le pseudo-label pour le prochain entraînement
    X_labeled = np.vstack([X_labeled, X_unlabeled[top_idx]])
    y_labeled = np.hstack([y_labeled, preds[top_idx]])
    y_labeled_cat = to_categorical(y_labeled, 10)
    
    # 4e) Retirer du pool non-étiqueté
    # Pour ne pas réutiliser ces exemples comme non-étiquetés
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[top_idx] = False
    X_unlabeled = X_unlabeled[mask]
    
    # 4f) Afficher évolution
    # Permet de suivre la taille du pool étiqueté et la performance sur le jeu test
    test_acc = model.evaluate(X_test, y_test_cat, verbose=0)[1]
    print(f"It {it+1}: labeled_size={len(X_labeled)} | test_acc={test_acc:.3f}")

# ---------------------------
# 5️⃣ Évaluation finale
# ---------------------------
final_acc = model.evaluate(X_test, y_test_cat, verbose=0)[1]
print(f"\nAccuracy finale du Self-training CNN : {final_acc:.3f}")
