import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU


vocab_size = 15000 # on veut juste 15000 mots les plus courant
max_len = 300 

#On charge les données
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len, padding ='post') # post pour faire les remplissage  a la fin
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

#Imprimer les jeux des donnes
print(f"Donnes d'entrainement : {x_train.shape}")
print(f"Donnes de test : {x_test.shape}")


#On creer les models
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128 ),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
    ])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rnn_model.summary()

history = rnn_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)



# Save the model
rnn_model.save('Save/RNN_normal.keras')


# Load the model
from tensorflow.keras.models import load_model
rnn_model = load_model('Save/RNN_normal.keras')


rnn_loss, rnn_accuracy = rnn_model.evaluate(x_test, y_test)

print(f"La perte de RNN : {rnn_loss:.4f}")
print(f"La precision du RNN : {rnn_accuracy:.4f}")



### LSTM ###
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

history_lstm = lstm_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)



# Save the model
lstm_model.save('Save/LSTM_normal.keras')


lstm_loss, lstm_accuracy = lstm_model.evaluate(x_test, y_test)
print(f"La perte de LSTM : {lstm_loss:.4f}")
print(f"La précision du LSTM : {lstm_accuracy:.4f}")



### GRU ###
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.summary()

history_gru = gru_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)


# Save the model
gru_model.save('Save/GRU_normal.keras')




gru_loss, gru_accuracy = gru_model.evaluate(x_test, y_test)
print(f"La perte de GRU : {gru_loss:.4f}")
print(f"La précision de GRU : {gru_accuracy:.4f}")


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Precision Entrainement RNN ')
plt.plot(history_lstm.history['accuracy'], label='Precision Entrainement LSTM ')
plt.plot(history_gru.history['accuracy'], label='Precision Entrainement GRU ')
plt.title('Entrainement')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()




# ... Tout ton code d'entraînement et de sauvegarde ...

# Chargement du modèle entraîné
from tensorflow.keras.models import load_model
rnn_model = load_model('Save/RNN_normal.keras')
gru_model = load_model('Save/GRU_normal.keras')
lstm_model = load_model('Save/LSTM_normal.keras')


# Ensuite tu ajoutes ce code de test personnalisé ici
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
index_from = 3
word_to_id = {k:(v + index_from) for k,v in word_index.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3

max_len = 300

vocab_size = 15000  # doit être la même valeur que pour le modèle

def preprocess_sentence(sentence):
    words = text_to_word_sequence(sentence)
    seq = [word_to_id.get(word, 2) for word in words]  # 2 = <UNK>
    seq = [1] + seq  # ajouter <START>
    # remplacer les indices trop grands par 2 (<UNK>)
    seq = [w if w < vocab_size else 2 for w in seq]
    seq = pad_sequences([seq], maxlen=max_len, padding='post')
    return seq


# Phrase à tester
my_sentence = "i'am feel nothing"
processed = preprocess_sentence(my_sentence)

# Prédiction RNN simple
prediction_rnn = rnn_model.predict(processed)
print(f"Score de sentiment (0 négatif - 1 positif) : {prediction_rnn[0][0]:.4f}")

if prediction_rnn[0][0] > 0.5:
    print("Sentiment prédit : POSITIF")
else:
    print("Sentiment prédit : NÉGATIF")

# Prédiction LSTM
prediction_lstm = lstm_model.predict(processed)
print(f"Score de sentiment (0 négatif - 1 positif) : {prediction_lstm[0][0]:.4f}")

if prediction_lstm[0][0] > 0.5:
    print("Sentiment prédit : POSITIF")
else:
    print("Sentiment prédit : NÉGATIF")




# Prédiction GRU
prediction_gru = gru_model.predict(processed)
print(f"Score de sentiment (0 négatif - 1 positif) : {prediction_gru[0][0]:.4f}")

if prediction_gru[0][0] > 0.5:
    print("Sentiment prédit : POSITIF")
else:
    print("Sentiment prédit : NÉGATIF")



import json

# Supposons que word_to_id est ton dictionnaire mot->indice
with open('Exportation/vocab.json', 'w') as f:
    json.dump(word_to_id, f)



import tensorflow as tf

model = tf.keras.models.load_model("Save/GRU_normal.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_enable_resource_variables = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   # Opérations natives TFLite
    tf.lite.OpsSet.SELECT_TF_OPS      # Autoriser certaines opérations TF
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("Exportation/sentiment_model.tflite", "wb") as f:
    f.write(tflite_model)






