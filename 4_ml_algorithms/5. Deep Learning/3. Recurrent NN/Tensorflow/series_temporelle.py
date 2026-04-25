import numpy as np
import pandas as pd
import pickle  # pour sauvegarder et recharger le modèle entraîné

# Prétraitement
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Évaluation
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

# Deep Learning
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, SimpleRNN, Flatten, Dropout, LSTM, GRU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model



# Colonnes du dataset NSL-KDD
feature = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
           "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
           "root_shell","su_attempted","num_root","num_file_creations","num_shells",
           "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
           "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
           "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
           "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
           "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
           "label","difficulty"]

# Chargement des fichiers CSV
train='Data/KDDTrain+.txt'
test='Data/KDDTest+.txt'
test21='Data/KDDTest-21.txt'

train_data=pd.read_csv(train,names=feature)
test_data=pd.read_csv(test,names=feature)
test_data21 = pd.read_csv(test21, names= feature)

# Concaténation train + test
data= pd.concat([train_data, test_data], ignore_index=True)

# On retire la colonne inutile 'difficulty'
data.drop(['difficulty'],axis=1,inplace=True)

'''
Pourquoi ?
On charge et fusionne toutes les données pour simplifier le prétraitement.
La colonne difficulty n’est pas utile pour la détection d’intrusion → on la supprime.
'''
# Aperçu général
data.info()
data.describe().T

# Nombre d'exemples par label
data['label'].value_counts()

'''
Pourquoi ?
Vérifier types de données et valeurs manquantes.
Comprendre la distribution des attaques (DoS, Probe, R2L, U2R, normal).
'''

####### Regroupement des attaques en classes communes

def change_label(df):
    df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable',
                      'smurf','teardrop','udpstorm','worm'],'Dos', inplace=True)
    df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named',
                      'phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L', inplace=True)
    df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe', inplace=True)
    df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R', inplace=True)

change_label(data)

# Vérification de la nouvelle distribution
data.label.value_counts()

'''
Pourquoi ?
Le dataset contient 40+ types d’attaques.
Pour simplifier, on regroupe en 4 classes principales + normal (Dos, R2L, Probe, U2R).
'''

####### Normalisation des features numériques ####

# Colonnes numériques
numeric_col = data.select_dtypes(include='number').columns

# StandardScaler
std_scaler = StandardScaler()
def standardization(df, col):
    for i in col:
        arr = df[i]
        df[i] = std_scaler.fit_transform(np.array(arr).reshape(-1,1))
    return df

data = standardization(data, numeric_col)

'''
Pourquoi ?

Normaliser les features pour que toutes aient la même échelle → améliore convergence du RNN.
'''

################ Encodage des labels et one-hot des colonnes catégorielles  ########################
from sklearn.preprocessing import LabelBinarizer

# Label -> 0,1,2,3,4
le2 = preprocessing.LabelEncoder()
data['intrusion'] = le2.fit_transform(data['label'])

# On supprime la colonne label originale
data.drop(labels=['label'], axis=1, inplace=True)

# One-hot encoding pour protocol_type, service, flag
data = pd.get_dummies(data, columns=['protocol_type','service','flag'])

# Forcer toutes les colonnes numériques en float
data = data.astype(np.float32)

'''
Pourquoi ?
Les RNN prennent uniquement des valeurs numériques.
Les labels deviennent des vecteurs one-hot pour la classification multi-classes.
'''

################### Préparation des données pour le modèle #######################

y_data = data['intrusion']          # Labels → 0,1,2,3,4
X_data = data.drop(['intrusion'], axis=1)  # Toutes les features

# Convertir en numpy arrays
X_data = np.array(X_data)
y_data = LabelBinarizer().fit_transform(y_data)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Reshape pour RNN [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

'''
Pourquoi ?

Un RNN Keras attend un input 3D :

[samples, timesteps, features]


Ici :

samples = nombre d’exemples (lignes)

timesteps = longueur de la séquence (ici 1, car chaque exemple est indépendant)

features = nombre de colonnes/features par exemple

Donc on transforme [n_samples, n_features] → [n_samples, 1, n_features].

⚠️ Résultat : tout est dans une seule “timestep”.

Chaque ligne du dataset devient une séquence d’une seule étape (1 timestep) avec toutes les features comme entrée.

C’est pour que le RNN puisse lire quelque chose de séquentiel, même si ici les données ne sont pas réellement une série temporelle.
'''

################ Construction du SimpleRNN  ##################

model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(SimpleRNN(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(64, return_sequences=True))
model.add(Flatten())
model.add(Dense(units=50))
model.add(Dense(units=5, activation='softmax'))  # sortie 5 classes

'''
SimpleRNN(64): 64 neurones pour apprendre la séquence.
Dropout(0.2): régularisation pour éviter overfitting.
Flatten() + Dense: transformer séquence en vecteur pour classification.
Softmax: pour prédire les 5 classes d’intrusion.
'''

#### Compilation et entraînement ###

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)


##### Plot #####

## Precision
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

## Perte
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


##### LSTM ####################

################ Construction du SimpleRNN  ##################

model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(64, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(64, return_sequences=True))
model_lstm.add(Flatten())
model_lstm.add(Dense(units=50))
model_lstm.add(Dense(units=5, activation='softmax'))  # sortie 5 classes

'''
SimpleRNN(64): 64 neurones pour apprendre la séquence.
Dropout(0.2): régularisation pour éviter overfitting.
Flatten() + Dense: transformer séquence en vecteur pour classification.
Softmax: pour prédire les 5 classes d’intrusion.
'''

#### Compilation et entraînement ###

model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()

history_lstm = model_lstm.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)


##### Plot #####

## Precision
plt.plot(history_lstm.history["accuracy"])
plt.plot(history_lstm.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

## Perte
plt.plot(history_lstm.history["loss"])
plt.plot(history_lstm.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


############## GRU ####################

################ Construction du SimpleRNN  ##################

model_gru = Sequential()
model_gru.add(GRU(64, return_sequences=True, input_shape=(1, X_train.shape[2])))
model_gru.add(Dropout(0.2))
model_gru.add(GRU(64, return_sequences=True))
model_gru.add(Dropout(0.2))
model_gru.add(GRU(64, return_sequences=True))
model_gru.add(Flatten())
model_gru.add(Dense(units=50))
model_gru.add(Dense(units=5, activation='softmax'))  # sortie 5 classes


#### Compilation et entraînement ###

model_gru.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gru.summary()

history_gru = model_gru.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)


##### Plot #####

## Precision
plt.plot(history_gru.history["accuracy"])
plt.plot(history_gru.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

## Perte
plt.plot(history_gru.history["loss"])
plt.plot(history_gru.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


## Sauvegarde du modèle
model.save('Save/SimpleRNN_model_serialTemp.keras') 
model_lstm.save('Save/LSTM_model_serialTemp.keras') 
model_gru.save('Save/GRU_model_serialTemp.keras') 


### Chargment du model 
model_simple = load_model('Save/SimpleRNN_model_serialTemp.keras')
model_lstm = load_model('Save/LSTM_model_serialTemp.keras')
model_gru = load_model('Save/GRU_model_serialTemp.keras')


# -----------------------------------------------
# Préparation des données de test
# -----------------------------------------------

# Conserver le DataFrame original pour les colonnes (features)
X_data_df = data.drop(['intrusion'], axis=1)  # DataFrame original (features)

# Copier les données de test
test21_data = test_data21.copy()

# Regrouper les attaques en classes communes
change_label(test21_data)

# Supprimer colonne inutile
test21_data.drop(['difficulty'], axis=1, inplace=True)

# Normalisation des features numériques
test21_data = standardization(test21_data, numeric_col)

# Transformation du label en 0,1,2,3,4
test21_data['intrusion'] = le2.transform(test21_data['label'])
test21_data.drop(['label'], axis=1, inplace=True)

# One-hot encoding pour protocol_type, service, flag
test21_data = pd.get_dummies(test21_data, columns=['protocol_type','service','flag'])

# Forcer toutes les colonnes en float32
test21_data = test21_data.astype(np.float32)

# Séparer la colonne 'intrusion' avant de réordonner les colonnes
y_test21 = LabelBinarizer().fit_transform(test21_data['intrusion'])
X_test21_df = test21_data.drop(['intrusion'], axis=1)

# Ajouter les colonnes manquantes
missing_cols = set(X_data_df.columns) - set(X_test21_df.columns)
for c in missing_cols:
    X_test21_df[c] = 0

# Réordonner les colonnes exactement comme X_data_df
X_test21_df = X_test21_df[X_data_df.columns]

# Convertir en numpy et reshape pour RNN
X_test21 = np.array(X_test21_df)
X_test21 = np.reshape(X_test21, (X_test21.shape[0], 1, X_test21.shape[1]))

# -----------------------------------------------
# Prédiction et évaluation sur les modèles RNN
# -----------------------------------------------

# SimpleRNN
y_pred_simple = model_simple.predict(X_test21, verbose=1)
y_pred_simple_classes = np.argmax(y_pred_simple, axis=1)
print("SimpleRNN Accuracy:", accuracy_score(np.argmax(y_test21, axis=1), y_pred_simple_classes))
print(classification_report(np.argmax(y_test21, axis=1), y_pred_simple_classes))

# LSTM
y_pred_lstm = model_lstm.predict(X_test21)
y_pred_lstm_classes = np.argmax(y_pred_lstm, axis=1)
print("LSTM Accuracy:", accuracy_score(np.argmax(y_test21, axis=1), y_pred_lstm_classes))
print(classification_report(np.argmax(y_test21, axis=1), y_pred_lstm_classes))

# GRU
y_pred_gru = model_gru.predict(X_test21)
y_pred_gru_classes = np.argmax(y_pred_gru, axis=1)
print("GRU Accuracy:", accuracy_score(np.argmax(y_test21, axis=1), y_pred_gru_classes))
print(classification_report(np.argmax(y_test21, axis=1), y_pred_gru_classes))

# -----------------------------------------------
# Prédiction sur nouvelle entrée utilisateur
# -----------------------------------------------

user_input = {
    'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
    'src_bytes': 181, 'dst_bytes': 5450, 'land': 0,
    'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0,
    'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 2,
    'srv_count': 2, 'serror_rate': 0, 'srv_serror_rate': 0, 'rerror_rate': 0,
    'srv_rerror_rate': 0, 'same_srv_rate': 1, 'diff_srv_rate': 0, 'srv_diff_host_rate': 0,
    'dst_host_count': 5, 'dst_host_srv_count': 5, 'dst_host_same_srv_rate': 1,
    'dst_host_diff_srv_rate': 0, 'dst_host_same_src_port_rate': 0, 'dst_host_srv_diff_host_rate': 0,
    'dst_host_serror_rate': 0, 'dst_host_srv_serror_rate': 0, 'dst_host_rerror_rate': 0,
    'dst_host_srv_rerror_rate': 0
}

user_df = pd.DataFrame([user_input])

# Normalisation des colonnes numériques
user_df = standardization(user_df, numeric_col.drop('intrusion', errors='ignore'))

# One-hot encoding
user_df = pd.get_dummies(user_df, columns=['protocol_type','service','flag'])

# Ajouter les colonnes manquantes
missing_cols = set(X_data_df.columns) - set(user_df.columns)
for c in missing_cols:
    user_df[c] = 0

# Réordonner les colonnes exactement comme X_data_df
user_df = user_df[X_data_df.columns]

# Convertir en numpy et reshape# Forcer toutes les colonnes en float32
X_user = user_df.astype(np.float32).to_numpy()  # conversion numpy
X_user = np.reshape(X_user, (X_user.shape[0], 1, X_user.shape[1]))


# Prédiction sur l'utilisateur
y_user_pred = model_lstm.predict(X_user)
predicted_class = np.argmax(y_user_pred, axis=1)
predicted_label = le2.inverse_transform(predicted_class)
print("L'entrée utilisateur est classée comme:", predicted_label[0])



