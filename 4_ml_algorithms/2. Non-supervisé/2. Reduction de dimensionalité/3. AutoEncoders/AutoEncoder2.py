'''
Jusqu’ici, tout va bien : nous avons développé un autoencodeur sans compression.
Maintenant, nous allons modifier la configuration du modèle de manière à ce que 
la couche "bottleneck" (goulot d’étranglement, espace latent) ait **moins de neurones** :
- moitié du nombre d’entrées (ex. 50 neurones si 100 entrées),
- puis on peut aussi essayer avec 25 et 10 neurones.
Pourquoi ? → Pour tester la compression et observer son effet sur la reconstruction.
'''

# Importation des librairies nécessaires
from sklearn.datasets import make_classification   # Génération d’un dataset artificiel
from sklearn.preprocessing import MinMaxScaler     # Normalisation des données entre 0 et 1
from sklearn.model_selection import train_test_split  # Division des données en train/test
from tensorflow.keras.models import Model          # API fonctionnelle Keras pour définir le modèle
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model      # Visualiser l’architecture du réseau
from matplotlib import pyplot                      # Visualisation des courbes de perte


# --- Définition du dataset ---
X, y = make_classification(
    n_samples=1000,     # Pourquoi ? → 1000 échantillons générés
    n_features=100,     # Chaque échantillon a 100 variables (features)
    n_informative=10,   # Seulement 10 variables informatives
    n_redundant=90,     # 90 variables redondantes
    random_state=1      # Résultat reproductible
)

# Nombre de colonnes (features), utile pour définir la taille des couches
n_inputs = X.shape[1]

# Division du dataset en train/test (80% apprentissage, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Mise à l’échelle des données (0 → 1)
t = MinMaxScaler()
t.fit(X_train)                  # Calculer les min/max sur le jeu d’entraînement
X_train = t.transform(X_train)  # Normaliser les données d’entraînement
X_test = t.transform(X_test)    # Normaliser les données de test avec les mêmes min/max


# --- Définition de l’encodeur ---
visible = Input(shape=(n_inputs,))  # Couche d’entrée : 100 neurones

# Couche cachée 1 (200 neurones, soit 2 * n_inputs)
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)   # Normalisation pour stabiliser l’entraînement
e = LeakyReLU()(e)            # Activation LeakyReLU

# Couche cachée 2 (100 neurones, même taille que l’entrée)
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

# Couche bottleneck (compression)
n_bottleneck = round(float(n_inputs) / 10.0)  # Pourquoi ? → on compresse à 10% de la taille (ici 10 neurones)
bottleneck = Dense(n_bottleneck)(e)


# --- Définition du décodeur ---
# Couche cachée 1 (100 neurones, reconstruction partielle)
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# Couche cachée 2 (200 neurones, expansion)
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

# Couche de sortie (100 neurones, activation linéaire car données continues)
output = Dense(n_inputs, activation='linear')(d)


# --- Autoencodeur complet ---
model = Model(inputs=visible, outputs=output)

# Compilation du modèle
# Optimiseur : Adam → efficace pour ce type de tâche
# Perte : MSE → mesurer l’écart entre les données originales et reconstruites
model.compile(optimizer='adam', loss='mse')


# Visualisation de l’architecture de l’autoencodeur
plot_model(model, 'images/autoencoder_compress.png', show_shapes=True)


# --- Entraînement du modèle ---
# Objectif : reproduire les entrées après compression
history = model.fit(
    X_train, X_train, 
    epochs=200, batch_size=16, 
    verbose=2, 
    validation_data=(X_test, X_test)
)


# --- Visualisation des courbes de perte ---
pyplot.plot(history.history['loss'], label='train')      # Perte sur entraînement
pyplot.plot(history.history['val_loss'], label='test')   # Perte sur validation
pyplot.legend()
pyplot.show()


# --- Sauvegarde uniquement de l’encodeur ---
# Pourquoi ? → L’encodeur seul sert à réduire la dimension des données
encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'images/encoder_compress.png', show_shapes=True)

# Sauvegarde sur disque
encoder.save('Save/encoder_compress.keras')  # (Attention : "kerras" semble une faute de frappe → devrait être ".keras")


'''
Dans ce cas, nous constatons que la perte est aussi faible que dans l'exemple 
précédent sans compression, ce qui suggère que le modèle fonctionne peut-être tout aussi bien 
avec un goulot d'étranglement dix fois plus petit que l'entrée. Un tracé des courbes 
d'apprentissage est créé, montrant une fois de plus que le modèle atteint un bon ajustement lors 
de la reconstruction de l'entrée, qui se maintient tout au long de l'apprentissage, sans surajustement.

'''