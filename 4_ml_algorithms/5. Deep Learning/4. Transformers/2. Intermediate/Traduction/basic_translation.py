import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- 1. Jeu de données minimal (paires français -> anglais) ---
french_sentences = [
    "chat",
    "chien",
    "pomme",
    "maison",
    "fromage",
    "bonjour",
    "comment ça va",
    "merci",
    "bonne nuit",
    "je suis fatigué",
    "où est la gare",
    "je t'aime",
    "j'aime le fromage",
    "il fait chaud",
    "je mange une pomme",
    "elle est gentille",
    "nous allons au parc",
    "il pleut aujourd'hui",
    "je bois de l'eau",
    "tu es mon ami",
    "je vais à l'école",
    "il est mon frère",
    "elle est ma sœur",
    "nous aimons le chocolat",
    "veux-tu un café ?",
    "je n'aime pas ça",
    "je parle français",
    "pouvez-vous m'aider ?",
    "quelle heure est-il ?",
    "où est la bibliothèque ?",
    "je voudrais un gâteau",
    "c'est une belle journée",
    "je travaille demain",
    "je vais courir",
    "il est très grand",
    "j'aime lire des livres",
    "nous habitons à Paris",
    "avez-vous un stylo ?",
    "je suis content",
    "elle chante bien",
]

english_sentences = [
    "cat",
    "dog",
    "apple",
    "house",
    "cheese",
    "hello",
    "how are you",
    "thank you",
    "good night",
    "i am tired",
    "where is the station",
    "i love you",
    "i like cheese",
    "it is hot",
    "i am eating an apple",
    "she is kind",
    "we are going to the park",
    "it is raining today",
    "i drink water",
    "you are my friend",
    "i go to school",
    "he is my brother",
    "she is my sister",
    "we like chocolate",
    "do you want a coffee?",
    "i don't like that",
    "i speak french",
    "can you help me?",
    "what time is it?",
    "where is the library?",
    "i would like a cake",
    "it is a beautiful day",
    "i work tomorrow",
    "i am going running",
    "he is very tall",
    "i like to read books",
    "we live in paris",
    "do you have a pen?",
    "i am happy",
    "she sings well",
]


# --- 2. Tokenization ---
tokenizer_fr = Tokenizer(filters='')
tokenizer_fr.fit_on_texts(french_sentences)
input_sequences = tokenizer_fr.texts_to_sequences(french_sentences)

tokenizer_en = Tokenizer(filters='')
tokenizer_en.fit_on_texts(english_sentences)
target_sequences = tokenizer_en.texts_to_sequences(english_sentences)

print("Vocabulaire français :", tokenizer_fr.word_index)
print("Séquences françaises :", input_sequences)
print("Vocabulaire anglais :", tokenizer_en.word_index)
print("Séquences anglaises :", target_sequences)

# --- 3. Padding ---
max_input_len = max(len(seq) for seq in input_sequences)

# On définit les index spéciaux <start> et <end>
start_token_index = len(tokenizer_en.word_index) + 1
end_token_index = start_token_index + 1

# Ajouter <start> et <end> aux séquences cibles
target_sequences_with_start_end = []
for seq in target_sequences:
    new_seq = [start_token_index] + seq + [end_token_index]
    target_sequences_with_start_end.append(new_seq)

max_target_len = max(len(seq) for seq in target_sequences_with_start_end)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
decoder_input_data = pad_sequences(target_sequences_with_start_end, maxlen=max_target_len, padding='post')

# Construire la cible décalée
decoder_target_data = []
for seq in target_sequences_with_start_end:
    shifted = seq[1:] + [0]*(max_target_len - len(seq))
    decoder_target_data.append(shifted)

decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_target_len, padding='post')

print("decoder_target_data :")
print(decoder_target_data)
print(decoder_target_data.dtype)
print("Max value:", decoder_target_data.max())
print("Min value:", decoder_target_data.min())

# --- 4. Paramètres ---
num_decoder_tokens = end_token_index + 1  # Inclut <start> et <end>
embedding_dim = 64
latent_dim = 128

# --- 5. Définir l'encodeur ---
encoder_inputs = tf.keras.Input(shape=(None,), name="encoder_inputs")
enc_emb = tf.keras.layers.Embedding(input_dim=len(tokenizer_fr.word_index) + 1,
                                    output_dim=embedding_dim,
                                    mask_zero=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# --- 6. Définir le décodeur ---
decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = tf.keras.layers.Embedding(input_dim=num_decoder_tokens,
                                          output_dim=embedding_dim,
                                          mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# --- 7. Créer et compiler le modèle complet ---
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 8. Préparer les entrées du décodeur pour l'entraînement ---
decoder_input_data_with_start = np.zeros_like(decoder_input_data)
decoder_input_data_with_start[:, 1:] = decoder_input_data[:, :-1]
# Le premier token est bien <start> (start_token_index)
decoder_input_data_with_start[:, 0] = start_token_index

# --- 9. Entraînement ---
model.fit([encoder_input_data, decoder_input_data_with_start], decoder_target_data,
          batch_size=2,
          epochs=100)

# --- 10. Modèle encodeur pour l'inférence ---
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

# --- 11. Modèle décodeur pour l'inférence ---
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# --- 12. Fonction de traduction mot par mot ---
def translate_batch(input_seqs):
    index_to_word = {index: word for word, index in tokenizer_en.word_index.items()}
    index_to_word[start_token_index] = "<start>"
    index_to_word[end_token_index] = "<end>"

    translations = []
    for input_seq in input_seqs:
        input_seq = input_seq.reshape(1, -1)
        states_value = encoder_model.predict(input_seq)

        target_seq = np.array([[start_token_index]])
        decoded_sentence = []
        stop_condition = False

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if sampled_token_index == end_token_index or len(decoded_sentence) > max_target_len:
                stop_condition = True
            else:
                sampled_word = index_to_word.get(sampled_token_index, '')
                decoded_sentence.append(sampled_word)

            target_seq = np.array([[sampled_token_index]])
            states_value = [h, c]

        translations.append(' '.join(decoded_sentence))
    return translations



# --- 13. Test de traduction ---
test_sentences = ["chat", "chien", "je suis fatigué"]

test_seqs = tokenizer_fr.texts_to_sequences(test_sentences)
test_seqs = pad_sequences(test_seqs, maxlen=max_input_len, padding='post')

results = translate_batch(test_seqs)
for sent, trans in zip(test_sentences, results):
    print(f"{sent} --> {trans}")




# Sauvegarder les modèles
model.save('Save/seq2seq_model_full.keras')
encoder_model.save('Save/seq2seq_encoder.keras')
decoder_model.save('Save/seq2seq_decoder.keras')

# Sauvegarder les tokenizers
import pickle
with open('Save/tokenizer_fr.pkl', 'wb') as f:
    pickle.dump(tokenizer_fr, f)
with open('Save/tokenizer_en.pkl', 'wb') as f:
    pickle.dump(tokenizer_en, f)
