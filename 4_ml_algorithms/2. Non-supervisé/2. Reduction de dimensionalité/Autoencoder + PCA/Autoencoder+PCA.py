import matplotlib.pyplot as plt  # pour créer des graphiques et visualiser les données
import pandas as pd               # pour manipuler des tableaux de données (non utilisé ici mais souvent utile)
from sklearn.datasets import make_blobs  # pour générer des données synthétiques en clusters
from sklearn.model_selection import train_test_split  # pour séparer les données en train/test
from sklearn.preprocessing import MinMaxScaler  # pour normaliser les données entre 0 et 1
from sklearn.neural_network import MLPRegressor  # réseau de neurones pour l'autoencodeur
from sklearn.decomposition import PCA  # pour réduire la dimensionnalité des données
from sklearn.metrics import mean_squared_error, silhouette_score  # pour évaluer modèles et clusters
import numpy as np  # pour manipuler des tableaux et calculs numériques


# Liste de couleurs pour visualiser les clusters
cols = ['#1FC17B', '#78FECF', '#555B6E', '#CC998D', '#429EA6',
        '#153B50', '#8367C7', '#EE6352', '#C287E8', '#F0A6CA', 
        '#521945', '#361F27', '#828489', '#9AD2CB', '#EBD494', 
        '#53599A', '#80DED9', '#EF2D56', '#446DF6', '#AF929D']


'''
Dans cette tâche, nous allons créer notre jeu de données et le prétraiter afin qu'il soit prêt pour le modelage.
Nous utiliserons un jeu de données que je crée artificiellement pour ce problème spécifique.
Pour ce faire, nous utilisons la fonction make_blobs de scikit-learn, qui génère une distribution de données
en fonction des paramètres donnés.
'''
# make_blobs retourne :
# X : les coordonnées des points dans l'espace
# y : l'étiquette de cluster correspondant à chaque point

# n_features : nombre de dimensions (ici 50)
# 50 est un bon compromis pour réduire plus tard en 2D
# centers : nombre de clusters (20)
X, y = make_blobs(n_features=50, centers=20,  # 50 dimensions et 20 clusters
                  n_samples=20000,            # nombre de points total (20k)
                  cluster_std=0.2,            # écart-type intra-cluster
                  center_box=(-1, 1),         # limite des centres dans [-1,1]
                  random_state=17)            # pour reproduire les mêmes données


# Séparation du jeu de données en entraînement et test
# 90% pour entraîner, 10% pour tester
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17)

# Chaque centre peut être situé dans un cube [-1,1] pour chaque dimension


# Normalisation des données
scaler = MinMaxScaler()  # transforme chaque dimension pour que ses valeurs soient entre 0 et 1

# On ajuste le scaler sur le jeu d'entraînement et on transforme ensuite
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)  # ⚠ idéalement .transform ici pour éviter la fuite de données


## 3: Modèle de base (baseline)

# PCA pour réduire de 50 dimensions à 2 dimensions
pca = PCA(n_components=2)  # réduction à 2D pour visualisation
pca.fit(X_train)  # ajustement de la PCA sur le jeu d'entraînement

# Transformation du jeu de test selon la PCA
res_pca = pca.transform(X_test)

'''
(2000, 2)
On obtient 2000 points (10% des données) projetés en 2 dimensions.
'''

# Boucle pour visualiser chaque cluster
unique_labels = np.unique(y_test)

for index, unique_labels in enumerate(unique_labels):
    x_data = res_pca[y_test == unique_labels]  # sélection des points appartenant au cluster
    # affichage des points
    plt.scatter(x_data[:,0], 
                x_data[:,1],        # deuxième dimension
                alpha=0.3,          # transparence pour mieux voir les superpositions
                c=cols[index])      # couleur du cluster

# Explication de PCA :
# PCA trouve les directions de plus grande variance dans les données.
# La première composante capture le plus de variance,
# la seconde est orthogonale à la première et capture la deuxième plus grande variance.
plt.xlabel('Composante Principale #1')
plt.ylabel('Composante Principale #2')
plt.title('Résultats PCA')


### Autoencodeur
# On utilise MLPRegressor car il s'agit d'une tâche de type régression (on veut prédire un vecteur réel)
autoencoder = MLPRegressor(alpha=1e-15, 
                           hidden_layer_sizes=(50, 100, 50, 2, 50, 100, 50),  # encodeur puis décodeur
                           random_state=1, max_iter=20000)

# Entraînement de l'autoencodeur
autoencoder.fit(X_train, X_train)  # entrée = X_train, sortie = X_train


#### Réduction de dimensionnalité avec l'encodeur

W = autoencoder.coefs_       # poids du réseau
biases = autoencoder.intercepts_  # biais du réseau

# Affichage des dimensions de chaque matrice de poids
for w in W:
    print(w.shape)

'''
La première couche : 50 → 50
La seconde : 50 → 100
Et ainsi de suite...

On ne s'intéresse qu'aux 4 premières couches (encodeur)
les suivantes correspondent au décodeur.
'''

encoder_weights = W[0:4]
encoder_biases = biases[0:4]

# Fonction pour encoder les données en passant par chaque couche de l'encodeur
def encode(encoder_weights, encoder_biases, data):
    res_ae = data
    for index, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
        if index+1 == len(encoder_weights):
            res_ae = res_ae @ w + b  # dernière couche : pas d'activation
        else:
            res_ae = np.maximum(0, res_ae @ w + b)  # ReLU pour les autres couches
    return res_ae

# Encodage du jeu de test
res_ae = encode(encoder_weights, encoder_biases, X_test)  # données non vues pendant l'entraînement

# Cette fonction prend un vecteur 50D et le transforme en 2D


'''
On obtient 2000 échantillons chacun en 2 dimensions.
'''


# Visualisation de l'espace latent de l'autoencodeur
unique_labels = np.unique(y_test)

for index, unique_label in enumerate(unique_labels):
    latent_space = res_ae[y_test == unique_label]  # points du cluster
    plt.scatter(latent_space[:,0], latent_space[:,1], alpha=0.3, c=cols[index])
    
plt.xlabel('Latent X')
plt.ylabel('Latent Y')
plt.title('Résultats Autoencodeur')

'''
Résultat prometteur : les clusters sont plus distincts, moins de chevauchements.
L'autoencodeur a compressé 50 dimensions en 2 tout en conservant les informations de cluster.
Il fonctionne donc très bien pour représenter l'espace 2D.
'''


# Évaluation avec le score de silhouette

# Score sur les données originales (50D)
silhouette_score(X_test, y_test)

'''
Score silhouette ~0.61
-1 : mauvais regroupement, 1 : regroupement parfait
Plus le score est proche de 1, mieux chaque point est assigné à son cluster.
'''

# Score sur les données PCA (2D)
silhouette_score(res_pca, y_test)

'''
Score silhouette ~0.36
PCA est moins performant car les clusters se chevauchent.
'''

# Score sur les données encodées par l'autoencodeur (2D)
silhouette_score(res_ae, y_test)

'''
Score silhouette ~0.80
L'autoencodeur fournit une meilleure séparation des clusters que PCA ou même les données originales.
'''
