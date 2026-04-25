# ============================
# IMPORTS
# ============================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# (Remarque : plusieurs imports de ton code original étaient inutiles,
# ex: pandas, preprocessing, Bidirectional, etc. Je les ai enlevés pour clarifier)


# ============================
# LECTURE ET NETTOYAGE DU TEXTE
# ============================

# On lit le fichier texte brut (Alice au pays des merveilles)
with open("Data/alice_in_wonderland.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# On nettoie ligne par ligne
cleaned_lines = []
for line in lines:
    line = line.strip().lower()                     # supprime espaces + met en minuscule
    line = line.encode("ascii", "ignore").decode()  # supprime caractères non-ASCII
    if len(line) == 0:                              # saute les lignes vides
        continue
    cleaned_lines.append(line)

# On fusionne toutes les lignes en un seul gros texte
text = " ".join(cleaned_lines)

print("Extrait du texte :")
print(text[:500])   # affiche les 500 premiers caractères


# ============================
# ENCODAGE DES CARACTÈRES
# ============================

# On récupère la liste des caractères uniques (le vocabulaire)
chars = set([c for c in text])
nb_chars = len(chars)
print("Nombre de caractères uniques :", nb_chars)

# On crée deux dictionnaires pour passer du caractère → index et inversement
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))


# ============================
# CRÉATION DES SÉQUENCES D’APPRENTISSAGE
# ============================

SEQLEN = 10   # longueur de la fenêtre (10 caractères comme entrée)
STEP = 1      # combien on décale la fenêtre à chaque fois (ici 1)

input_chars = []   # X (séquences de 10 caractères)
label_chars = []   # y (le caractère qui suit)

# Exemple : si SEQLEN = 10
# "alice in w" → 'o'
# "lice in wo" → 'n'
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i+SEQLEN])     # séquence
    label_chars.append(text[i+SEQLEN])       # caractère suivant


# ============================
# ONE-HOT ENCODING
# ============================

# x : tableau [nb_sequences, SEQLEN, nb_chars]
# y : tableau [nb_sequences, nb_chars]
x = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=bool)
y = np.zeros((len(input_chars), nb_chars), dtype=bool)

for i, input_char in enumerate(input_chars):
    for j, char in enumerate(input_char):
        x[i, j, char2index[char]] = 1   # active la case correspondant au caractère
    y[i, char2index[label_chars[i]]] = 1  # cible = le caractère suivant


# ============================
# PARAMÈTRES DU MODÈLE
# ============================

HIDDEN_SIZE = 128           # nombre de neurones dans la couche RNN
BATCH_SIZE = 128            # taille du batch
NUM_ITERATIONS = 10         # combien de fois on entraîne + génère du texte
NUM_EPOCHS_PER_ITERATION = 1  # nombre d’époques par itération
NUM_PREDS_PER_EPOCH = 100   # combien de caractères générer après chaque itération


# ============================
# CONSTRUCTION DU MODÈLE
# ============================

model = Sequential()
# Couche RNN : lit séquence de 10 caractères encodés en one-hot
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False, input_shape=(SEQLEN, nb_chars), unroll=True))
# Couche Dense : prédit la distribution de probabilité sur tous les caractères
model.add(Dense(nb_chars, activation='softmax'))

# Compilation : crossentropy car on fait une classification multi-classes (sur les caractères)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ============================
# ENTRAÎNEMENT + GÉNÉRATION DE TEXTE
# ============================

for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: %d" % (iteration))

    # Entraînement sur tout le dataset
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    # Choix d’une séquence de départ ("seed") aléatoire
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]

    print("\nGenerating from seed: %s" % (test_chars))
    print(test_chars, end="")

    # On génère caractère par caractère
    for i in range(NUM_PREDS_PER_EPOCH):
        # Prépare une séquence test de taille [1, SEQLEN, nb_chars]
        Xtest = np.zeros((1, SEQLEN, nb_chars))
        for j, ch in enumerate(test_chars):
            Xtest[0, j, char2index[ch]] = 1

        # Prédiction de la distribution de probabilité
        pred = model.predict(Xtest, verbose=0)[0]

        # Choisir le caractère le plus probable
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")

        # Décaler la fenêtre : on supprime le premier caractère et ajoute la prédiction
        test_chars = test_chars[1:] + ypred

print("\n=== Fin de l’entraînement et génération ===")
