from tabnanny import verbose   # (Pas utile ici) Import pour analyser la syntaxe de code Python, souvent utilisé pour détecter les erreurs d'indentation.
import numpy as np            # Import de NumPy pour la manipulation de tableaux numériques (vecteurs, matrices).
import pandas as pd           # Import de Pandas pour lire et manipuler des données tabulaires (CSV).
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de test.
import tensorflow as tf       # Import de TensorFlow, framework pour construire et entraîner des modèles de Deep Learning.
import tensorflow.keras as keras   # Import de Keras (intégré dans TensorFlow) pour simplifier la création de réseaux de neurones.
from tensorflow.keras.models import Sequential   # Permet de construire un modèle séquentiel (empilement de couches).
from tensorflow.keras.layers import LSTM         # Import de la couche LSTM (Long Short-Term Memory), adaptée au traitement de séquences textuelles.
from tensorflow.keras.layers import Dense, Activation, Dropout  # Couches classiques : Dense (neurones fully connected), Activation, Dropout (régularisation).
from tensorflow.keras.layers import Embedding, Bidirectional    # Embedding pour transformer des mots en vecteurs, Bidirectional pour entraîner LSTM dans 2 directions.
from sklearn import preprocessing    # Outils de prétraitement (non utilisés ici, mais utile pour encoder/normaliser les données).
from tensorflow.keras.layers import Bidirectional  # (Déjà importé plus haut, redondant).
from tensorflow.keras.preprocessing import sequence, text  # Pour transformer les textes en séquences d'index de mots et les mettre à longueur fixe.
import os              # Module pour gérer les fichiers et chemins.
from nltk.corpus import stopwords   # Liste de mots vides ("the", "is", "and"…) à supprimer car peu informatifs.
import re              # Expressions régulières pour nettoyer le texte.
import warnings        # Gérer les avertissements.
warnings.filterwarnings('ignore')   # Ignore les warnings pour ne pas encombrer la sortie.


# Chargement des données d'entraînement et de validation
train = pd.read_csv('Data/jigsaw-toxic-comment-train.csv')   # Charge le dataset principal contenant les commentaires toxiques.
validation = pd.read_csv('Data/validation.csv')              # Charge un dataset de validation séparé.


# Suppression des colonnes inutiles (on ne garde que 'comment_text' et 'toxic')
train.drop(["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], axis=1, inplace=True)


# Chargement des stopwords anglais
stop_words = set(stopwords.words('english'))


# Fonction de prétraitement du texte
def data_text_preprocess(total_text, ind, col):
    # Vérifie que le texte n'est pas un entier (erreur de format possible)
    if type(total_text) is not int:
        string = ""
        # Remplace les caractères spéciaux par des espaces
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # Remplace les espaces multiples par un seul
        total_text = re.sub('\s+',' ', str(total_text))
        # Convertit le texte en minuscules
        total_text = total_text.lower()

        # Supprime les mots vides (stopwords)
        for word in total_text.split():
            if not word in stop_words:   # On garde uniquement les mots informatifs
                string += word + " "

        # Remplace le texte original par le texte nettoyé
        train[col][ind] = total_text


# Application de la fonction de prétraitement sur chaque commentaire
for index, row in train.iterrows():
    if type(row['comment_text']) is str:   # Vérifie que c'est bien du texte
        data_text_preprocess(row['comment_text'], index, 'comment_text')


# Division en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    train.comment_text.values,   # Les commentaires
    train.toxic.values,          # Les labels (0 = non toxique, 1 = toxique)
    stratify=train.toxic.values, # Maintenir la même proportion toxique/non-toxique dans les 2 ensembles
    random_state=42,             # Fixe la graine pour la reproductibilité
    test_size=0.2,               # 20% des données pour le test
    shuffle=True                 # Mélange les données avant la séparation
)


# Création du tokenizer (convertit mots → entiers)
token = text.Tokenizer(num_words=None)   # Aucune limite sur le nombre de mots
token.fit_on_texts(list(X_train) + list(X_test))   # Apprend le vocabulaire sur tous les commentaires

# Conversion du texte en séquences d'entiers
X_train_seq = token.texts_to_sequences(X_train)
X_test_seq = token.texts_to_sequences(X_test)

# Uniformisation de la longueur des séquences (padding)
max_len = 100   # Longueur maximale fixée à 100 mots
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len)

# Vocabulaire (dictionnaire mot → index)
word_index = token.word_index


# Construction du modèle LSTM
model = Sequential()   # Modèle séquentiel (empilement linéaire de couches)
model.add(Embedding(len(word_index) + 1, 256, input_length=max_len))  
# Couche d'embedding : chaque mot devient un vecteur de 256 dimensions

model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3)))  
# LSTM bidirectionnel avec 256 neurones, dropout pour éviter l’overfitting

model.add(Dense(1, activation='sigmoid'))  
# Couche de sortie avec activation sigmoid (car binaire : toxique ou non toxique)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
# Compilation du modèle avec loss adaptée à un problème de classification binaire


# Entraînement du modèle
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),   # On valide sur les données de test
    epochs=5,                               # Nombre d’itérations d’apprentissage
    batch_size=128                          # Nombre d’exemples traités avant mise à jour des poids
)


# Prédictions sur l'ensemble test
score = model.predict(X_test_pad)


# Évaluation du modèle
test_loss , test_acc = model.evaluate(X_test_pad, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")


# Chargement du dataset de test final
test = pd.read_csv('Data/test.csv')


# Fonction de nettoyage pour le dataset test
def preprocess_text(text):
    if type(text) is not str:
        text = str(text)
    text = text.lower()   # Met en minuscules
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)   # Supprime caractères spéciaux
    text = re.sub('\s+',' ', text)               # Supprime espaces multiples
    text = ' '.join(word for word in text.split() if word not in stop_words)   # Supprime stopwords
    return text


# Application du prétraitement au dataset test
test['comment_text'] = test['comment_text'].apply(lambda x: preprocess_text(x))

# Conversion en séquences
test_seq = token.texts_to_sequences(test['comment_text'].values)

# Mise à longueur fixe
test_pad = sequence.pad_sequences(test_seq, maxlen=max_len)

# Prédictions finales
test_pred = model.predict(test_pad)

# Ajout de la colonne "toxic" dans le dataset test
test['toxic'] = test_pred

# Sauvegarde du fichier de soumission (id, toxic)
test[['id', 'toxic']].to_csv('Data/submission.csv', index=False)
