import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import json, os


# CONFIG

vocab_size = 15000
max_len = 300
embedding_dim = 128
epochs = 15
batch_size = 64

# CHARGEMENT DES DONNÉES

files = [
    "Data/goemotions_1.csv",
    "Data/goemotions_2.csv",
    "Data/goemotions_3.csv",
]

emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise'
]

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs)

for col in emotion_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)

texts = df['text'].tolist()
labels = (df[emotion_columns].values > 0).astype(np.float32)

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


# FONCTION DE CREATION MODELE

def create_model(rnn_type="LSTM", bidirectional=False):
    rnn_layer = LSTM if rnn_type == "LSTM" else GRU
    if bidirectional:
        rnn = Bidirectional(rnn_layer(128, return_sequences=True))
    else:
        rnn = rnn_layer(128, return_sequences=True)

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        rnn,
        GlobalMaxPooling1D(),
        Dense(len(emotion_columns), activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# ENTRAINEMENT MULTI-MODELES

os.makedirs("Save", exist_ok=True)
os.makedirs("Exportation", exist_ok=True)

models_config = [
    ("LSTM", False),
    ("GRU", False),
    ("LSTM", True),
    ("GRU", True)
]

trained_models = {}

for rnn_type, bidirectional in models_config:
    name = f"{'Bi' if bidirectional else ''}{rnn_type}"
    print(f"\n===== Entraînement {name} =====")

    model = create_model(rnn_type, bidirectional)
    checkpoint_path = f"Save/{name}_checkpoint.keras"

    checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    keras_path = f"Save/{name}.keras"
    model.save(keras_path)
    trained_models[name] = model

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    with open(f"Exportation/{name}.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"✅ Modèle {name} entraîné et exporté.")

# EXPORT VOCABULAIRE
word_index = tokenizer.word_index
limited_word_index = {k: v for k, v in word_index.items() if v <= vocab_size}
with open('Exportation/vocab_multiple.json', 'w', encoding='utf-8') as f:
    json.dump(limited_word_index, f, ensure_ascii=False, indent=2)

# COMPARAISON MSE SUR EXEMPLES
examples = [
    ("I am feel nothing",                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
    ("I am so happy today!",               [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
    ("I can't believe you lied to me",     [0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I love spending time with you",      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
    ("That was so disappointing",          [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I am really scared right now",       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I feel so proud of my work",         [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
    ("This is absolutely disgusting",      [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I forgive you",                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
    ("I can't stop laughing",              [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I'm very grateful for your help",    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
    ("I can't wait to see you again",      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
    ("I am deeply moved by your words",    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I am worried about the results",     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("This makes me so angry",             [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("I miss you so much",                 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
    ("What a surprise!",                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
    ("I'm so confused right now",          [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    ("You really cared about me",          [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
]

def predict_vector(text, model):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return model.predict(padded, verbose=0)[0]

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


#  Fonction prédiction Top 3 émotions (à ajouter après l'entraînement) ---
def predict_top3_emotions(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    preds = model.predict(padded, verbose=0)[0]
    top_indices = preds.argsort()[-3:][::-1]
    results = {emotion_columns[i]: float(preds[i]) for i in top_indices}
    return results

# Exemple unique
exemple = "I am feel nothing"
print("\n=== TOP-3 pour l'exemple:", exemple, "===\n")
for name, model in trained_models.items():
    print(f"{name} -> {predict_top3_emotions(exemple, model, tokenizer)}")

# TOP-3 pour tous les exemples (ta liste 'examples' déjà définie)
print("\n\n=== TOP-3 pour tous les exemples ===")
for text, truth in examples:
    print(f"\nTexte: {text}")
    for name, model in trained_models.items():
        preds_top3 = predict_top3_emotions(text, model, tokenizer)
        print(f" {name} top3: {preds_top3}")
    # affiche la vérité terrain (noms des émotions)
    truth_idx = np.where(np.array(truth) > 0)[0]
    truth_names = [emotion_columns[i] for i in truth_idx]
    print(" Truth:", truth_names)

# Petit metric utile : combien de fois le top1 (prédiction la plus forte) est dans la vérité
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

