# Importation des librairies nécessaires
import matplotlib.pyplot as plt   # Pour la visualisation des graphes statiques
import pandas as pd              # Pour la manipulation de données sous forme de DataFrame
import numpy as np               # Pour les calculs numériques
import seaborn as sns            # Pour des visualisations statistiques plus esthétiques
import plotly.graph_objs as go   # Pour des graphes interactifs (ici, on l'importe mais on ne l'utilise pas vraiment)
import plotly.offline as py      # Pour exécuter plotly hors ligne
import random                    # Pour générer des couleurs aléatoires
from sklearn.preprocessing import StandardScaler   # Pour normaliser les données avant PCA
from sklearn.decomposition import PCA              # Pour appliquer l’Analyse en Composantes Principales (PCA)

# Chargement du dataset depuis le fichier CSV
dataset = pd.read_csv('Data/data.csv')

# Copie de sécurité pour éviter de modifier le dataset original directement
df = dataset.copy()

# Suppression des colonnes inutiles
# - 'id' : identifiant unique, ne contient pas d’information utile pour la classification
# - 'Unnamed: 32' : colonne vide issue du dataset original
df.drop(['id', 'Unnamed: 32'], axis= 1, inplace= True)

# Affiche un résumé de la structure du dataset (colonnes, types de données, valeurs manquantes, etc.)
df.info()

# On crée un DataFrame sans la variable cible 'diagnosis' (B ou M)
# Cela permet d’analyser uniquement les variables explicatives
df_ = df.drop('diagnosis', axis=1)

# Normalisation des données
# 👉 Pourquoi ? PCA est basé sur les variances.
#    Si une variable a une grande échelle (ex : 1000) et une autre une petite (ex : 0.1),
#    elle va dominer le calcul. La standardisation met toutes les variables sur la même échelle.
scaler = StandardScaler()
df_ = scaler.fit_transform(df_)

# Application d’une PCA sur toutes les dimensions (n_components par défaut = nb de colonnes)
pca = PCA()
pca_fit = pca.fit_transform(df_)

# Affiche la proportion de variance expliquée par chaque composante principale
# 👉 Cela montre quelle "part d'information" est capturée par chaque axe principal
pca.explained_variance_ratio_

# Affiche la variance cumulée expliquée
# 👉 Permet de voir combien de composantes il faut pour conserver, par ex., 95% de l'information
pca.explained_variance_ratio_.cumsum()

# Ajustement PCA complet (permettra de tracer le graphe cumulé de variance expliquée)
pca = PCA().fit(df_)

# Tracé de la variance cumulée expliquée en fonction du nombre de composantes
plt.xlabel("Nombre de Composantes")
plt.ylabel("Taux de variance cumulée")
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# Calcul de la variance expliquée (arrondie en %)
var1 = np.cumsum(np.round(pca.explained_variance_ratio_,decimals = 4) * 100)
print(var1)

# Tracé de la variance expliquée cumulée avec une ligne rouge à 95%
# 👉 Cela aide à identifier combien de composantes suffisent à garder 95% de l’information
plt.plot(var1)
plt.hlines(95, 0, 1300, colors='red', linestyles='dashed')  # Ligne horizontale seuil à 95%
plt.xlabel("Composantes principales")
plt.ylabel("Variance expliquée cumulée (%)")

# Nombre optimal de composantes nécessaires pour dépasser 95%
Num_components = var1 < 95
print("Nombre optimal de composantes :", Num_components.sum())

# On repart d’une copie propre du dataset original
df = dataset.copy()

# Séparation des variables explicatives (X) et de la cible (y)
y = df['diagnosis']   # Variable cible (Bénin "B" ou Malin "M")
X = df.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1)   # Variables explicatives uniquement

# Fonction pour créer un DataFrame avec les 2 premières composantes principales (PC1 et PC2)
# 👉 Pourquoi ? Cela permet de projeter les données en 2D et de visualiser si les classes sont séparables
def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)       # Normalisation obligatoire avant PCA
    pca = PCA(n_components=2)                   # Réduction à 2 dimensions principales
    pca_fit = pca.fit_transform(X)              # Transformation des données
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])  # Création DataFrame avec PC1 et PC2
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)      # Ajout de la variable cible
    return final_df

# Application de la fonction pour obtenir les données réduites à 2 dimensions
pca_df = create_pca_df(X, y)

# Fonction pour tracer un scatter plot en 2D avec PC1 et PC2
# 👉 Cela permet de voir visuellement si les points "B" et "M" se séparent bien après PCA
def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    # Génération de couleurs aléatoires pour chaque classe
    targets = list(dataframe[target].unique())
    colors = random.sample(["black", "red","yellow","green"], len(targets))

    # On trace un nuage de points par classe (Bénin vs Malin)
    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=30)
    
    # Ajout de la légende
    ax.legend(targets)
    ax.grid()
    plt.show()

# Affichage de la projection PCA en 2D
# 👉 Si les deux classes sont relativement séparées, PCA est utile pour visualisation et classification
plot_pca(pca_df, "diagnosis")
