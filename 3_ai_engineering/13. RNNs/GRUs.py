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

history = lstm_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)



# Save the model
lstm_model.save('Save/LSTM_normal.keras')


# Load the model
from tensorflow.keras.models import load_model
lstm_model = load_model('Save/LSTM_normal.keras')


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

history = gru_model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.2)


# Save the model
gru_model.save('Save/GRU_normal.keras')


# Load the model
from tensorflow.keras.models import load_model
gru_model = load_model('Save/GRU_normal.keras')




gru_loss, gru_accuracy = gru_model.evaluate(x_test, y_test)
print(f"La perte de GRU : {gru_loss:.4f}")
print(f"La précision de GRU : {gru_accuracy:.4f}")

