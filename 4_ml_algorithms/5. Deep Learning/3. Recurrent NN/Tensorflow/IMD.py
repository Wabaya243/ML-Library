# ==============================
# 1. Import des bibliothèques
# ==============================
import tensorflow as tf  # TensorFlow 2.x pour les réseaux de neurones
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenisation du texte
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Padding des séquences
from tensorflow.keras import Sequential  # Pour construire des modèles séquentiels
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, Dropout, GlobalAveragePooling1D  # Couches NN

import numpy as np  # Gestion de tableaux numériques
import pandas as pd  # Gestion de dataframes
import matplotlib.pyplot as plt  # Visualisation
import re  # Expressions régulières pour nettoyer le texte
import os  # Pour naviguer dans les dossiers

# ==============================
# 2. Chargement des données
# ==============================
data_path = 'Data/IMDB Dataset.csv'
df = pd.read_csv(data_path)  # On lit le fichier CSV dans un dataframe Pandas

# On crée une colonne label numérique : 1 pour 'positive', 0 pour 'negative'
df['label'] = (df['sentiment'] == 'positive').astype(int)

# Nettoyage des balises HTML dans le texte des reviews
df['review_clean'] = df['review'].map(lambda x: re.sub('<[^<]+?>', '', x))

# Comptage du nombre de mots par review
df['review_clean_words_count'] = df['review_clean'].map(lambda x: len(x.split()))

# Visualisation de la distribution du nombre de mots
plt.hist(df['review_clean_words_count'], bins=5, range=(0, 1200))
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de reviews")
plt.show()

'''
Histogramme pour voir la longueur des reviews :
    utile pour choisir la longueur maximale pour le padding des séquences plus tard.
'''

# ==============================
# 1. Préparer les textes et labels
# ==============================
texts = df['review_clean'].tolist()  # Liste de toutes les reviews nettoyées
labels = df['label'].tolist()        # Liste des labels correspondants

# ==============================
# 2. Paramètres pour la tokenisation
# ==============================
max_words = 10000  # On ne garde que les 10 000 mots les plus fréquents
maxlen = 500       # Longueur maximale de chaque review (padding/truncating)

# ==============================
# 3. Tokenisation
# ==============================
tokenizer = Tokenizer(num_words=max_words)  # Création de l'objet tokenizer
tokenizer.fit_on_texts(texts)               # Apprentissage du vocabulaire à partir des textes
sequences = tokenizer.texts_to_sequences(texts)  # Conversion des textes en séquences d'indices
word_index = tokenizer.word_index          # Dictionnaire mot -> indice

print(f"Nombre de tokens uniques : {len(word_index)}")

# ==============================
# 4. Padding des séquences
# ==============================
data = pad_sequences(sequences, maxlen=maxlen)  # Toutes les séquences ont la même longueur

# Conversion des labels en tableau numpy
labels = np.array(labels)

print(f"Shape des données : {data.shape}")
print(f"Shape des labels : {labels.shape}")

# ==============================
# 5. Mélanger et diviser les données
# ==============================
indices = np.arange(data.shape[0])
np.random.shuffle(indices)  # Mélange aléatoire
data = data[indices]
labels = labels[indices]

# Division en train, validation, test
training_samples = 24000
validation_samples = 6000
test_samples = 20000

X_train = data[:training_samples]
y_train = labels[:training_samples]

X_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]

X_test = data[training_samples+validation_samples:
              training_samples+validation_samples+test_samples]
y_test = labels[training_samples+validation_samples:
                training_samples+validation_samples+test_samples]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

'''
Préparer les textes et labels

texts : on extrait toutes les reviews nettoyées sous forme de liste.

labels : on extrait la colonne label sous forme de liste.

Paramètres tokenisation

max_words : limite le vocabulaire aux 10 000 mots les plus fréquents pour réduire la taille du modèle.

maxlen : chaque review sera tronquée ou complétée à 500 mots pour que toutes les séquences aient la même longueur.

Tokenisation

Tokenizer transforme chaque mot en un entier unique.

fit_on_texts apprend le vocabulaire sur l’ensemble des reviews.

texts_to_sequences transforme les reviews en séquences d’entiers.

word_index contient le mapping mot → indice.

Padding des séquences

Les séquences sont tronquées ou complétées à maxlen=500 pour avoir un tableau 2D (nb_reviews, maxlen).

Les labels sont convertis en numpy array pour TensorFlow.

Mélange et division

Mélange aléatoire des données pour éviter tout biais.

Division en train, validation, et test.

Vérification des formes : (24000, 500) pour X_train, (6000, 500) pour X_val, etc.

'''

# ==============================
# 1. Définition du modèle
# ==============================
model = Sequential()

# Couche Embedding : transforme chaque mot (indice) en un vecteur dense de dimension 128
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=maxlen))

# Flatten : transforme la matrice (500 mots x 128) en un vecteur 1D de taille 64000
model.add(GlobalAveragePooling1D())

# Dense : couche cachée de 32 neurones avec activation ReLU
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# Dense finale : sortie unique avec activation sigmoid pour classification binaire
model.add(Dense(1, activation='sigmoid'))

# ==============================
# 2. Compilation du modèle
# ==============================
model.compile(optimizer='rmsprop',        # Optimiseur RMSprop
              loss='binary_crossentropy', # Loss pour classification binaire
              metrics=['accuracy'])       # On veut suivre l'accuracy

# ==============================
# 3. Affichage du résumé
# ==============================
model.summary()

'''
Embedding

input_dim=max_words : vocabulaire de 10 000 mots.

output_dim=128 : chaque mot devient un vecteur dense de 128 dimensions.

input_length=maxlen : chaque review fait 500 mots → matrice (500, 128).

Flatten

Transforme la matrice (500, 128) en vecteur 1D (64000,) pour le Dense.

C’est une approche simple pour passer d’un embedding à des couches fully connected.

'''

# ==============================
# 1. Entraînement du modèle
# ==============================
history = model.fit(
    X_train, y_train,            # données d'entraînement
    epochs=10,                   # nombre d'itérations sur tout le dataset
    batch_size=32,               # nombre d'exemples par batch
    validation_data=(X_val, y_val)  # données de validation
)


import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']           # précision entraînement
    val_acc = history.history['val_accuracy']   # précision validation
    loss = history.history['loss']              # perte entraînement
    val_loss = history.history['val_loss']      # perte validation
    
    epochs = range(1, len(acc) + 1)
    
    # Accuracy
    plt.figure(figsize=(10,4))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.figure(figsize=(10,4))
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Appel de la fonction pour afficher les courbes
plot_history(history)


test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# Exemple de texte
new_review = "I really loved this movie! The acting was fantastic and the story was very engaging."

# 1️⃣ Nettoyer le texte (enlever balises HTML)
new_review_clean = re.sub('<[^<]+?>', '', new_review)
print(new_review_clean)

# 2️⃣ Transformer le texte en séquence d'indices avec le tokenizer existant
new_seq = tokenizer.texts_to_sequences([new_review_clean])

# 3️⃣ Padding pour avoir la même longueur que le modèle
new_seq_pad = pad_sequences(new_seq, maxlen=maxlen)
print(new_seq_pad.shape)

# 4️⃣ Prédiction
prediction = model.predict(new_seq_pad)
print("Raw prediction:", prediction)

# 5️⃣ Conversion en label 0/1
pred_label = (prediction > 0.5).astype(int)
print("Predicted label:", pred_label)



