# -------------------------------
# 📚 Importation des bibliothèques
# -------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc  # Pour dendrogrammes

# Optionnel : pour mélanger les données si nécessaire
from sklearn.utils import shuffle  

# -------------------------------
# 📌 Charger les datasets
# -------------------------------
anime = pd.read_csv('Data/anime.csv')   # Infos sur les animes
rating = pd.read_csv('Data/rating.csv') # Notes des utilisateurs

# -------------------------------
# 📌 Visualiser la distribution des ratings
# -------------------------------
cnt_pro = rating['rating'].value_counts()  # Compter les occurrences de chaque note

plt.figure(figsize=(6,4))
sns.barplot(x=cnt_pro.index, y=cnt_pro.values, alpha=0.8)  # Correction: arguments nommés
plt.ylabel('Number of rating', fontsize=12)
plt.xlabel('Rating', fontsize=12)
plt.xticks(rotation=80)  # Rotation pour meilleure lisibilité
plt.show()

# -------------------------------
# 📌 Calculer la moyenne des notes par utilisateur
# -------------------------------
Mean_rate = rating.groupby(['user_id']).mean().reset_index()
Mean_rate['mean_rating'] = Mean_rate['rating']  # Copier la moyenne dans une colonne dédiée
Mean_rate.drop(['anime_id','rating'], axis=1, inplace=True)  # Supprimer colonnes inutiles

# -------------------------------
# 📌 Garder uniquement les notes supérieures ou égales à la moyenne de chaque utilisateur
# -------------------------------
user = pd.merge(rating, Mean_rate, on=['user_id','user_id'])
user = user[user['rating'] >= user['mean_rating']]  # Correction logique: garder >= moyenne

# Vérification pour un utilisateur
user[user['user_id']==2].head(10)
user[user['user_id']==1].head(10)

# -------------------------------
# 📌 Fusionner les datasets anime et rating
# -------------------------------
Data = pd.merge(anime, user, on=['anime_id','anime_id'])

# Garder les utilisateurs jusqu'à 10000 pour réduire la taille (optionnel)
Data = Data[Data.user_id <= 10000]

# Vérification rapide
Data.head(10)
len(Data['anime_id'].unique())  # Nombre d'animes uniques
len(Data['user_id'].unique())   # Nombre d'utilisateurs uniques

# -------------------------------
# 📌 Créer une matrice utilisateur-anime (1 si l'utilisateur aime l'anime)
# -------------------------------
user_anime = pd.crosstab(Data['user_id'], Data['name'])
user_anime.head(10)

# -------------------------------
# 📌 Dendrogramme hiérarchique
# -------------------------------
plt.figure(figsize=(10, 7))
plt.title("Dendrogramme des utilisateurs")
# Ward linkage pour minimiser la variance intra-cluster
dend = shc.dendrogram(shc.linkage(user_anime, method='ward'))
plt.show()

# -------------------------------
# 📌 Clustering hiérarchique agglomératif
# -------------------------------
cluster = AgglomerativeClustering(
    n_clusters=10,        # Nombre de clusters à créer
    affinity='euclidean', # Distance Euclidienne
    linkage='ward'        # Méthode Ward pour fusion
)

cluster_labels = cluster.fit_predict(user_anime)  # Prédiction des clusters

# -------------------------------
# 📌 Visualisation des clusters (2 premières colonnes seulement)
# -------------------------------
plt.figure(figsize=(10, 7))
# Attention: user_anime a >2 colonnes, ici on ne peut afficher qu'une projection 2D
plt.scatter(user_anime.iloc[:,0], user_anime.iloc[:,1], c=cluster_labels, cmap='rainbow', s=50)
plt.xlabel(user_anime.columns[0])
plt.ylabel(user_anime.columns[1])
plt.title('Clusters d\'utilisateurs (projection 2D)')
plt.show()
