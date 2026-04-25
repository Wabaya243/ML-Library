import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Liste des chiffres de 0 à 9
numbers = ["0","1","2","3","4","5","6","7","8","9"]

# Correspondance texte
numbers_in_words = ["zéro","un","deux","trois","quatre","cinq","six","sept","huit","neuf"]

# Tokenisation des chiffres (entrée)
tokenizer_in = Tokenizer(char_level=True, filters='')
tokenizer_in.fit_on_texts(numbers)
input_sequences = tokenizer_in.texts_to_sequences(numbers)

# Tokenisation des mots (sortie)
tokenizer_out = Tokenizer(char_level=True, filters='')
tokenizer_out.fit_on_texts(numbers_in_words)
target_sequences = tokenizer_out.texts_to_sequences(numbers_in_words)

'''
char_level=True → chaque caractère devient un token.
Exemple : "zéro" → [z, é, r, o].
texts_to_sequences transforme les mots/chaînes en séquences d’indices numériques.
Notre modèle ne comprend que des nombres  c’est obligatoire pour le réseau.
'''

# Définir les tokens spéciaux
start_token_index = len(tokenizer_out.word_index) + 1
end_token_index = start_token_index + 1

# Ajouter <start> et <end> aux séquences cibles
target_sequences_with_start_end = []
for seq in target_sequences:
    new_seq = [start_token_index] + seq + [end_token_index]
    target_sequences_with_start_end.append(new_seq)
    
'''
<start> = indique au décodeur où commencer la séquence.
<end> = indique au décodeur quand arrêter.
Exemple : "zéro"  [<start>, z, é, r, o, <end>].
'''

# Déterminer la longueur maximale
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences_with_start_end)

# Padding (ajout de 0 pour uniformiser les longueurs)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
decoder_input_data = pad_sequences(target_sequences_with_start_end, maxlen=max_target_len, padding='post')

'''
Tous les séquences doivent avoir la même longueur pour le batch processing.
padding='post' → ajoute des zéros à la fin de chaque séquence.
'''
# Décaler de 1 pour créer la cible du décodeur
decoder_target_data = []
for seq in target_sequences_with_start_end:
    shifted = seq[1:] + [0]*(max_target_len - len(seq))
    decoder_target_data.append(shifted)

decoder_target_data = np.array(decoder_target_data)

'''
Décodeur prédit le token suivant à chaque étape.
Exemple : <start> z é r o <end>
decoder_input = <start> z é r o
decoder_target = z é r o <end>
Ce décalage permet l’entraînement étape par étape.
'''

num_decoder_tokens = end_token_index + 1
embedding_dim = 32
latent_dim = 64

'''
num_decoder_tokens = taille du vocabulaire de sortie (inclut <start> et <end>).
embedding_dim = dimension des vecteurs de représentation pour chaque caractère.
latent_dim = taille de l’état caché du LSTM (plus grand → modèle plus puissant).
'''

encoder_inputs = tf.keras.Input(shape=(None,), name="encoder_inputs")
enc_emb = tf.keras.layers.Embedding(input_dim=len(tokenizer_in.word_index)+1,
                                    output_dim=embedding_dim,
                                    mask_zero=True)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

'''
Embedding → transforme chaque token en vecteur dense.
LSTM → encode la séquence d’entrée en état caché (state_h, state_c).
encoder_states seront utilisés pour initialiser le décodeur.
'''
decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = tf.keras.layers.Embedding(input_dim=num_decoder_tokens,
                                          output_dim=embedding_dim,
                                          mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

'''
Décodeur prend le token <start> comme entrée initiale.
LSTM prédit le token suivant à chaque étape.
Dense + softmax → transforme l’état LSTM en probabilités sur tout le vocabulaire.
'''

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

decoder_input_data_with_start = np.zeros_like(decoder_input_data)
decoder_input_data_with_start[:, 1:] = decoder_input_data[:, :-1]
decoder_input_data_with_start[:, 0] = start_token_index

'''
Décalage : le premier token = <start>, le reste = séquence précédente.
C’est ce qu’on passe comme entrée du décodeur pendant l’entraînement.
'''
model.fit([encoder_input_data, decoder_input_data_with_start], decoder_target_data,
          batch_size=2,
          epochs=200)

'''
Entrée : séquences de chiffres + séquences décalées du décodeur.
Cible : séquences cibles décalées (decoder_target_data).
Le modèle apprend à transformer un chiffre en texte.
'''
# ------------------------------
# 10. Modèle encodeur pour l'inférence
# ------------------------------
# On crée un modèle encodeur séparé qui prend la séquence d'entrée
# et renvoie les états cachés (h, c) pour initialiser le décodeur
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

# ------------------------------
# 11. Modèle décodeur pour l'inférence
# ------------------------------
# Pour générer la séquence mot par mot, on doit réutiliser le décodeur
# mais en injectant à chaque pas les états cachés précédents

# Entrées pour l'état caché précédent du décodeur (LSTM)
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))  # état caché h
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))  # état de cellule c
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]  # liste pour LSTM

# On réutilise la couche d'embedding du décodeur
dec_emb2 = dec_emb_layer(decoder_inputs)

# Passage du vecteur d'entrée et des états cachés dans le LSTM
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)

# On récupère les nouveaux états pour les réinjecter au pas suivant
decoder_states2 = [state_h2, state_c2]

# On applique la couche Dense finale pour transformer l'état du LSTM
# en probabilités sur le vocabulaire de sortie (softmax)
decoder_outputs2 = decoder_dense(decoder_outputs2)

# Création du modèle décodeur pour l'inférence
# Entrées : token actuel + états précédents
# Sorties : probabilités du token suivant + nouveaux états cachés
decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

'''
En inférence, on encode l’entrée, puis on génère le texte caractère par caractère.
À chaque étape, le LSTM décodeur reçoit le token prédit précédent et son état caché.
'''
def translate_number(input_seq):
    # ------------------------------
    # 1. Création du dictionnaire index -> caractère
    # ------------------------------
    # On inverse le word_index du tokenizer de sortie pour pouvoir retrouver le caractère
    # à partir de l'index prédit par le modèle
    index_to_char = {v: k for k, v in tokenizer_out.word_index.items()}
    # On ajoute les tokens spéciaux <start> et <end>
    index_to_char[start_token_index] = "<start>"
    index_to_char[end_token_index] = "<end>"

    # ------------------------------
    # 2. Préparer la séquence d'entrée
    # ------------------------------
    # On reshape la séquence pour qu'elle soit de forme (1, longueur)
    # car le modèle attend un batch même pour un exemple unique
    input_seq = input_seq.reshape(1, -1)

    # On passe la séquence dans l'encodeur pour récupérer les états cachés
    states_value = encoder_model.predict(input_seq)

    # ------------------------------
    # 3. Initialiser le décodeur
    # ------------------------------
    # On commence la génération avec le token <start>
    target_seq = np.array([[start_token_index]])

    # Liste qui contiendra les caractères générés
    decoded_sentence = []

    # Condition pour arrêter la génération
    stop_condition = False

    # ------------------------------
    # 4. Boucle de génération mot par mot
    # ------------------------------
    while not stop_condition:
        # Prédiction du prochain token + mise à jour des états
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # On prend l'index du token ayant la probabilité la plus élevée
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # Condition d'arrêt :
        # - on rencontre le token <end>
        # - ou la longueur maximale est dépassée
        if sampled_token_index == end_token_index or len(decoded_sentence) > max_target_len:
            stop_condition = True
        else:
            # On ajoute le caractère correspondant à la prédiction
            decoded_sentence.append(index_to_char.get(sampled_token_index, ''))

        # Le token prédit devient le prochain input du décodeur
        target_seq = np.array([[sampled_token_index]])

        # On met à jour les états cachés pour la prochaine étape
        states_value = [h, c]

    # ------------------------------
    # 5. Retourner la séquence générée sous forme de string
    # ------------------------------
    return ''.join(decoded_sentence)


test_numbers = ["0","5","9"]
test_seqs = tokenizer_in.texts_to_sequences(test_numbers)
test_seqs = pad_sequences(test_seqs, maxlen=max_input_len, padding='post')

for n, seq in zip(test_numbers, test_seqs):
    print(f"{n} --> {translate_number(seq)}")




