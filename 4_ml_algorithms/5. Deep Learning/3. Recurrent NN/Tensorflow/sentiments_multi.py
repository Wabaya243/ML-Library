# ------------------------------
# IMPORTATIONS
# ------------------------------
import pandas as pd  # Pour manipuler les données sous forme de DataFrame
import numpy as np   # Pour les opérations mathématiques sur tableaux
import tensorflow as tf  # Pour construire et entraîner les modèles de deep learning

# Prétraitement de texte : tokenisation et padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Modules pour créer et charger les modèles Keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, GlobalMaxPooling1D, Bidirectional

# Callbacks pour contrôler l'entraînement (sauvegarde et early stopping)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Séparation des données en train/test
from sklearn.model_selection import train_test_split

import json, os  # Pour gérer les fichiers JSON et les chemins système


# ------------------------------
# CONFIGURATION DES HYPERPARAMÈTRES
# ------------------------------
vocab_size = 15000       # Taille maximale du vocabulaire à utiliser
max_len = 300            # Longueur maximale des séquences après padding
embedding_dim = 128      # Dimension des vecteurs d'embedding
epochs = 15              # Nombre d'époques d'entraînement
batch_size = 64          # Taille des mini-batchs pour l'entraînement


# ------------------------------
# CHARGEMENT DES DONNÉES
# ------------------------------
files = [
    "Data/goemotions_1.csv",
    "Data/goemotions_2.csv",
    "Data/goemotions_3.csv",
]  # Liste des fichiers CSV contenant les données

# Colonnes représentant les émotions à prédire
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]

# Lire tous les CSV et concaténer dans un seul DataFrame
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs)

# Convertir toutes les colonnes d'émotions en float32 et remplir les valeurs manquantes par 0
for col in emotion_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)

# Extraire les textes et les labels
texts = df['text'].tolist()
labels = (df[emotion_columns].values > 0).astype(np.float32)  # Binarisation des émotions (>0 devient 1)


# ------------------------------
# TOKENISATION ET PADDING
# ------------------------------
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")  # Tokenizer limité à vocab_size, remplace les mots inconnus par <UNK>
tokenizer.fit_on_texts(texts)  # Construire le vocabulaire à partir des textes

# Convertir les textes en séquences d'entiers puis appliquer le padding pour uniformiser la longueur
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding='post')


# ------------------------------
# SPLIT TRAIN / TEST
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)  # 80% train, 20% test, random_state pour reproductibilité


# ------------------------------
# FONCTION DE CRÉATION DE MODÈLE
# ------------------------------
def create_model(rnn_type="LSTM", bidirectional=False):
    """
    Crée un modèle RNN pour la classification multi-label des émotions.
    rnn_type : "LSTM" ou "GRU"
    bidirectional : True pour un RNN bidirectionnel
    """
    rnn_layer = LSTM if rnn_type == "LSTM" else GRU  # Choix du type de RNN

    # Si bidirectionnel, envelopper la couche RNN dans Bidirectional
    if bidirectional:
        rnn = Bidirectional(rnn_layer(128, return_sequences=True))
    else:
        rnn = rnn_layer(128, return_sequences=True)

    # Construction du modèle séquentiel
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),  # Couche d'embedding
        rnn,                                                         # Couche RNN
        GlobalMaxPooling1D(),                                        # Réduction des dimensions par max pooling
        Dense(len(emotion_columns), activation='sigmoid')            # Couche de sortie multi-label
    ])

    # Compilation du modèle avec pertes et métriques adaptées à multi-label
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Multi-label classification
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


# ------------------------------
# ENTRAÎNEMENT MULTI-MODELES
# ------------------------------
os.makedirs("Save", exist_ok=True)       # Crée le dossier Save si inexistant
os.makedirs("Exportation", exist_ok=True)  # Crée le dossier Exportation si inexistant

# Configurations des modèles à entraîner
models_config = [
    ("LSTM", False),
    ("GRU", False),
    ("LSTM", True),
    ("GRU", True)
]

trained_models = {}  # Dictionnaire pour stocker les modèles entraînés

# Boucle sur chaque configuration
for rnn_type, bidirectional in models_config:
    name = f"{'Bi' if bidirectional else ''}{rnn_type}"  # Nom du modèle
    print(f"\n===== Entraînement {name} =====")

    model = create_model(rnn_type, bidirectional)  # Création du modèle
    checkpoint_path = f"Save/{name}_checkpoint.keras"

    # Callback pour sauvegarder le meilleur modèle selon la précision
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="precision"
    )

    # Callback pour arrêter l'entraînement si auc ne s'améliore pas pendant 3 époques
    early_stop = EarlyStopping(
        monitor= 'auc',
        patience=3,
        restore_best_weights=True
    )

    # Entraînement du modèle
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # Sauvegarde finale du modèle entraîné
    keras_path = f"Save/{name}.keras"
    model.save(keras_path)
    trained_models[name] = model

    # Export du modèle en TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,   # Opérations standard TFLite
        tf.lite.OpsSet.SELECT_TF_OPS      # Autoriser certaines opérations TensorFlow non natives
    ]
    
    tflite_model = converter.convert()  # Conversion
    with open(f"Exportation/{name}.tflite", "wb") as f:
        f.write(tflite_model)  # Sauvegarde du fichier .tflite

    print(f"✅ Modèle {name} entraîné et exporté.")


# ------------------------------
# ÉVALUATION SUR LE JEU DE TEST
# ------------------------------
print("\n=== Évaluation complète sur le jeu de test ===")
results_summary = []

for name, model in trained_models.items():
    preds = model.predict(X_test, verbose=0)          # Prédictions sur le jeu de test
    mse = np.mean((preds - y_test) ** 2)             # Calcul du MSE moyen

    # Calcul du Top1 match rate (la meilleure prédiction correspond-elle à une émotion réelle ?)
    correct_top1 = 0
    for i in range(len(y_test)):
        top1_pred = np.argmax(preds[i])
        true_labels = np.where(y_test[i] > 0)[0]  # indices des vraies émotions
        if top1_pred in true_labels:
            correct_top1 += 1
    top1_rate = correct_top1 / len(y_test)

    results_summary.append((name, mse, top1_rate))

# Tri par MSE ascendant puis Top1 descendant
results_summary.sort(key=lambda x: (x[1], -x[2]))
for name, mse, top1_rate in results_summary:
    print(f"{name}: MSE={mse:.4f} | Top1 match={top1_rate:.2%}")

# Meilleur modèle global
best_model_name, best_mse, best_top1 = results_summary[0]
print(f"\n🏆 Meilleur modèle: {best_model_name} (MSE={best_mse:.4f}, Top1={best_top1:.2%})")


# ------------------------------
# EXPORT DU VOCABULAIRE
# ------------------------------
word_index = tokenizer.word_index
limited_word_index = {k: v for k, v in word_index.items() if v <= vocab_size}  # Limite vocabulaire
with open('Exportation/vocab_multiple.json', 'w', encoding='utf-8') as f:
    json.dump(limited_word_index, f, ensure_ascii=False, indent=2)  # Export JSON


# ------------------------------
# PRÉDICTION ET COMPARAISON SUR EXEMPLES
# ------------------------------
# Fonction pour prédire le vecteur d'émotions d'un texte
def predict_vector(text, model):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return model.predict(padded, verbose=0)[0]

# Comparaison MSE pour chaque modèle
print("\n===== COMPARAISON MSE =====")
for text, truth in examples:
    truth = np.array(truth, dtype=np.float32)
    print(f"\nTexte: {text}")
    best_model = None
    best_mse = float("inf")
    for name, model in trained_models.items():
        mse = np.mean((predict_vector(text, model) - truth) ** 2)
        print(f"{name} MSE: {mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_model = name
    print(f"→ Plus proche: {best_model}")

# Fonction pour récupérer les 3 émotions les plus probables
def predict_top3_emotions(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    preds = model.predict(padded, verbose=0)[0]
    top_indices = preds.argsort()[-3:][::-1]  # Indices des 3 plus grandes valeurs
    results = {emotion_columns[i]: float(preds[i]) for i in top_indices}
    return results

# Exemple unique
examples = "I am feel nothing"
print("\n=== TOP-3 pour l'exemple:", examples, "===")
for name, model in trained_models.items():
    print(f"{name} -> {predict_top3_emotions(examples, model, tokenizer)}")

# TOP-3 pour tous les exemples
print("\n\n=== TOP-3 pour tous les exemples ===")
for text, truth in examples:
    print(f"\nTexte: {text}")
    for name, model in trained_models.items():
        preds_top3 = predict_top3_emotions(text, model, tokenizer)
        print(f" {name} top3: {preds_top3}")
    truth_idx = np.where(np.array(truth) > 0)[0]
    truth_names = [emotion_columns[i] for i in truth_idx]
    print(" Truth:", truth_names)

# Petit metric : top1 correct
top1_counts = {name: 0 for name in trained_models.keys()}
for text, truth in examples:
    truth_names = [emotion_columns[i] for i, v in enumerate(truth) if v > 0]
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    for name, model in trained_models.items():
        preds = model.predict(padded, verbose=0)[0]
        top1 = emotion_columns[np.argmax(preds)]
        if top1 in truth_names:
            top1_counts[name] += 1

print("\n\n=== Top1 matches (combien de fois le top1 est dans la vérité) ===")
for name, cnt in top1_counts.items():
    print(f"{name}: {cnt}/{len(examples)}")


# ------------------------------
# CONVERSION SELECT_TF_OPS POUR TFLITE
# ------------------------------
model = tf.keras.models.load_model("Save/BiLSTM.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Autoriser les opérations TensorFlow complètes (non natives TFLite)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Désactiver le lowering automatique des TensorList ops (préserve certaines opérations)
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
open("Exportation/BiLSTM_select_tf_ops.tflite", "wb").write(tflite_model)
print("Conversion terminée avec SELECT_TF_OPS !")
