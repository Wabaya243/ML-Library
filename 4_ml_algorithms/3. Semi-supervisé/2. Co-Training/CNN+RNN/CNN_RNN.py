import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

# ==========================
# 1️⃣ Charger MNIST
# ==========================
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype('float32') / 255.0  # Normalisation entre 0 et 1
X_test = X_test.astype('float32') / 255.0

# --------------------------
# Préparer les inputs pour CNN et RNN
# --------------------------
X_train_cnn = np.expand_dims(X_train_full, -1)  # CNN attend (n_samples, 28,28,1)
X_test_cnn = np.expand_dims(X_test, -1)

# Pour RNN, chaque image devient une séquence de lignes (28 timesteps, 28 features)
X_train_rnn = X_train_full.copy()  # (n_samples, 28,28)
X_test_rnn = X_test.copy()

# ==========================
# 2️⃣ Créer pool étiqueté / non-étiqueté
# ==========================
labeled_size = int(0.1 * X_train_full.shape[0])  # 10% étiqueté pour simuler semi-supervisé

# Pool étiqueté
X_labeled_cnn = X_train_cnn[:labeled_size]
X_labeled_rnn = X_train_rnn[:labeled_size]
y_labeled = y_train_full[:labeled_size]

# Pool non-étiqueté
X_unlabeled_cnn = X_train_cnn[labeled_size:]
X_unlabeled_rnn = X_train_rnn[labeled_size:]
y_unlabeled = y_train_full[labeled_size:]  # utilisé seulement pour suivi des performances

y_labeled_cat = to_categorical(y_labeled, 10)  # One-hot pour Keras
y_test_cat = to_categorical(y_test, 10)

# ==========================
# 3️⃣ Définir CNN
# ==========================
def create_cnn():
    # CNN capte les **patterns spatiaux locaux** (ex. lignes, contours, coins)
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),  # filtre 3x3
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),  # réduit dimension + invariance translation
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')  # sortie 10 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================
# 4️⃣ Définir RNN
# ==========================
def create_rnn():
    # RNN capte les **séquences**, ici chaque image devient une séquence de lignes
    # Il peut détecter des patterns “verticaux” ou progression de traits ligne par ligne
    model = models.Sequential([
        layers.SimpleRNN(128, activation='tanh', input_shape=(28,28)),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn()
rnn_model = create_rnn()

# ==========================
# 5️⃣ Boucle de Co-training
# ==========================
num_iterations = 5
num_add = 500  # nombre d'exemples les plus confiants à ajouter à chaque itération

for it in range(num_iterations):
    # ---------------------------------------------------
    # 5a) Entraîner chaque modèle sur ses données étiquetées
    # CNN → patterns spatiaux locaux
    # RNN → patterns ligne par ligne (séquence)
    # Les deux voient des aspects différents des images
    # ---------------------------------------------------
    cnn_model.fit(X_labeled_cnn, y_labeled_cat, epochs=3, batch_size=64, verbose=0)
    rnn_model.fit(X_labeled_rnn, y_labeled_cat, epochs=3, batch_size=64, verbose=0)
    
    # ---------------------------------------------------
    # 5b) Prédire sur le pool non-étiqueté
    # Chaque modèle génère ses pseudo-labels et son niveau de confiance
    # ---------------------------------------------------
    cnn_probs = cnn_model.predict(X_unlabeled_cnn, verbose=0)
    rnn_probs = rnn_model.predict(X_unlabeled_rnn, verbose=0)
    
    cnn_conf = np.max(cnn_probs, axis=1)  # confiance max pour chaque exemple
    rnn_conf = np.max(rnn_probs, axis=1)
    
    cnn_preds = np.argmax(cnn_probs, axis=1)
    rnn_preds = np.argmax(rnn_probs, axis=1)
    
    # ---------------------------------------------------
    # 5c) Sélectionner les plus confiants
    # Ces exemples seront ajoutés à l'autre modèle pour améliorer sa diversité
    # ---------------------------------------------------
    idx_cnn = cnn_conf.argsort()[-num_add:]
    idx_rnn = rnn_conf.argsort()[-num_add:]
    
    # ---------------------------------------------------
    # 5d) Échanger les pseudo-labels
    # CNN reçoit les exemples les plus confiants du RNN
    # → CNN apprend des séquences verticales que RNN a bien comprises
    # RNN reçoit les exemples les plus confiants du CNN
    # → RNN apprend des patterns spatiaux locaux que CNN a bien détectés
    # Cette **coopération** améliore les deux modèles
    # ---------------------------------------------------
    X_labeled_cnn = np.vstack([X_labeled_cnn, X_unlabeled_cnn[idx_rnn]])
    X_labeled_rnn = np.vstack([X_labeled_rnn, X_unlabeled_rnn[idx_cnn]])
    
    y_labeled = np.hstack([y_labeled, rnn_preds[idx_rnn]])  # labels de RNN ajoutés
    y_labeled = np.hstack([y_labeled, cnn_preds[idx_cnn]])  # labels de CNN ajoutés
    y_labeled_cat = to_categorical(y_labeled, 10)
    
    # ---------------------------------------------------
    # 5e) Retirer ces exemples du pool non-étiqueté
    # Pour éviter de les réutiliser à l'avenir
    # ---------------------------------------------------
    mask = np.ones(len(X_unlabeled_cnn), dtype=bool)
    mask[idx_cnn] = False
    mask[idx_rnn] = False
    X_unlabeled_cnn = X_unlabeled_cnn[mask]
    X_unlabeled_rnn = X_unlabeled_rnn[mask]
    y_unlabeled = y_unlabeled[mask]
    
    # ---------------------------------------------------
    # 5f) Affichage de suivi
    # Permet de voir l'évolution de la taille du pool étiqueté et les performances
    # ---------------------------------------------------
    cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
    rnn_acc = rnn_model.evaluate(X_test_rnn, y_test_cat, verbose=0)[1]
    print(f"It {it+1}: labeled_size={len(X_labeled_cnn)} | CNN_acc={cnn_acc:.3f} | RNN_acc={rnn_acc:.3f}")

# ==========================
# 6️⃣ Évaluation finale
# ==========================
final_cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
final_rnn_acc = rnn_model.evaluate(X_test_rnn, y_test_cat, verbose=0)[1]
print(f"\nAccuracy finale CNN : {final_cnn_acc:.3f}")
print(f"Accuracy finale RNN : {final_rnn_acc:.3f}")
