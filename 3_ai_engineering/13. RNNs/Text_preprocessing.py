import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
import numpy as np
import matplotlib.pyplot as plt


# ------------------------
# Paramètres principaux
# ------------------------
vocab_size = 15000  # Nombre maximal de mots à considérer dans le vocabulaire
max_len = 200       # Longueur maximale des séquences (reviews), padding ajouté si plus court

# ------------------------
# Chargement des données IMDB (critique film)
# ------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# ------------------------
# Décodage (optionnel) des premières critiques pour visualiser le texte (pas utilisé dans la suite)
# ------------------------
word_index = imdb.get_word_index()  # Dictionnaire mot → indice
reverse_word_index = {value: key for key, value in word_index.items()}
decoded_reviews = ["".join([reverse_word_index.get(i - 3, '?') for i in review]) for review in x_train[:5]]

# ------------------------
# Padding des séquences : on complète les séquences plus courtes avec des 0 à la fin
# ------------------------
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')  # padding après la séquence
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

print(f"La forme de données d'entrain : {x_train.shape}, {y_train.shape}")
print(f"La forme de données de test : {x_test.shape}, {y_test.shape}")

# ------------------------
# Chargement des embeddings GloVe pré-entraînés (100 dimensions)
# ------------------------
embedding_index = {}
glove_file = 'glove.6B/glove.6B.100d.txt'
with open(glove_file, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embedding_index[word] = coefs
        
print(f"Chargé {len(embedding_index)} vecteurs de mots depuis GloVe")

# ------------------------
# Construction de la matrice d'embedding adaptée à notre vocabulaire
# Chaque ligne correspond au vecteur GloVe d'un mot du vocabulaire
# ------------------------
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))  # matrice initialisée à zéro

for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)  # récupérer vecteur GloVe
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # insérer vecteur dans la matrice

# ------------------------
# Modèle LSTM avec embeddings GloVe (non entraînables)
# ------------------------
lstm_model = Sequential({
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),  # embeddings fixes
    LSTM(128, activation='tanh', return_sequences=False),  # couche LSTM classique
    Dense(1, activation='sigmoid')  # sortie binaire (sentiment positif/négatif)
})

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics={'accuracy'})

lstm_model.summary()

# Entraînement du modèle avec validation (20% des données d'entraînement)
history = lstm_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# Évaluation sur les données test
lstm_loss, lstm_accuracy = lstm_model.evaluate(x_test, y_test)

print(f"La perte de LSTM avec GloVe : {lstm_loss:.4f}")
print(f"La précision du LSTM avec GloVe : {lstm_accuracy:.4f}")

# ------------------------
# Modèle GRU avec embeddings GloVe (non entraînables)
# ------------------------
gru_model = Sequential({
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
})

gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics={'accuracy'})

gru_model.summary()

history = gru_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

gru_loss, gru_accuracy = gru_model.evaluate(x_test, y_test)

print(f"La perte de GRU avec GloVe : {gru_loss:.4f}")
print(f"La précision de GRU avec GloVe : {gru_accuracy:.4f}")

# ------------------------
# Modèle LSTM sans embeddings pré-entraînés (embeddings entraînés à partir de zéro)
# ------------------------
wh_lstm_model = Sequential({
    Embedding(input_dim=vocab_size, output_dim=128),  # embeddings apprises pendant l'entraînement
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
})

wh_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics={'accuracy'})

wh_lstm_model.summary()

history = wh_lstm_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

wh_lstm_loss, wh_lstm_accuracy = wh_lstm_model.evaluate(x_test, y_test)

print(f"La perte de LSTM sans GloVe : {wh_lstm_loss:.4f}")
print(f"La précision du LSTM sans GloVe : {wh_lstm_accuracy:.4f}")

# ------------------------
# Modèle GRU sans embeddings pré-entraînés (embeddings entraînés à partir de zéro)
# ------------------------
wh_gru_model = Sequential({
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
})

wh_gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics={'accuracy'})

wh_gru_model.summary()

history = wh_gru_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

wh_gru_loss, wh_gru_accuracy = wh_gru_model.evaluate(x_test, y_test)

print(f"La perte de GRU sans GloVe : {wh_gru_loss:.4f}")
print(f"La précision de GRU sans GloVe : {wh_gru_accuracy:.4f}")

# ------------------------
# Visualisation des performances (accuracy) des modèles LSTM avec/sans GloVe
# ------------------------
models = ['LSTM Sans GloVe', 'LSTM Avec GloVe']
accuracies = [wh_lstm_accuracy, lstm_accuracy]

plt.bar(models, accuracies, color=["blue", 'yellow'])
plt.title('Comparaison des précisions LSTM Avec et Sans GloVe')
plt.ylabel('Accuracy')
plt.show()
