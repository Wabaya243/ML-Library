# Importation des bibliothèques nécessaires
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# Pour les graphiques interactifs 3D
import plotly as py
import plotly.graph_objs as go

# Ignorer les warnings pour avoir un affichage plus propre
import warnings
warnings.filterwarnings('ignore')

# Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder
# Pour la création du dendrogramme
import scipy.cluster.hierarchy as sch
# Pour le clustering hiérarchique
from sklearn.cluster import AgglomerativeClustering 


### 📌 1. Chargement des données
dataset = pd.read_csv('Data/Mall_Customers.csv')

# Vérifier les valeurs manquantes
dataset.isnull().sum()


### 📌 2. Visualisation de la distribution de certaines variables
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)  # Sous-graphique
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5) # Espacement entre les graphes
    sns.distplot(dataset[x] , bins = 15) # Distribution
    plt.title('Distplot of {}'.format(x))
plt.show()


### 📌 3. Encodage de la variable catégorielle "Gender"
Le = LabelEncoder()
dataset['Gender'] = Le.fit_transform(dataset['Gender'])


### 📌 4. Heatmap pour visualiser les données
plt.figure(1, figsize = (16 ,8))
sns.heatmap(dataset) # Attention : par défaut ce n’est pas très informatif
plt.show()


### 📌 5. Dendrogramme pour voir le nombre optimal de clusters
plt.figure(1, figsize = (16 ,8))
dendrogram = sch.dendrogram(sch.linkage(dataset, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


### 📌 6. Création du modèle Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters = 5, metric= 'euclidean', linkage ='average')

# Prédiction des clusters pour chaque client
y_hc = hc.fit_predict(dataset)

# Ajouter les clusters comme nouvelle colonne dans le dataset
dataset['cluster'] = pd.DataFrame(y_hc)


### 📌 7. Visualisation en 3D avec Plotly
trace1 = go.Scatter3d(
    x= dataset['Age'],  # Axe X
    y= dataset['Spending Score (1-100)'],  # Axe Y
    z= dataset['Annual Income (k$)'],  # Axe Z
    mode='markers',
    marker=dict(
        color = dataset['cluster'], # Couleur par cluster
        size= 10, # Taille des points
        line=dict(
            color= dataset['cluster'], # Bordure des points
            width= 12
        ),
        opacity=0.8
    )
)

data = [trace1]

layout = go.Layout(
    title= 'Clusters using Agglomerative Clustering',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)

fig = go.Figure(data=data, layout=layout)

# 👉 Ici dépend du contexte d'exécution
# En Jupyter Notebook on utilise `py.offline.iplot(fig)`
# Mais dans un script Python normal il faut `py.offline.plot(fig)`
py.offline.plot(fig) 

# plt.show() ne sert à rien ici car plt = matplotlib et non Plotly
# plt.show()


### 📌 8. Visualisation 2D classique avec Matplotlib
X = dataset.iloc[:, [3,4]].values
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='purple', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='orange', label ='Cluster 5')
plt.title('Clusters of Customers (Hierarchical Clustering Model)')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()


### 📌 9. Sauvegarder le dataset segmenté
dataset.to_csv("Data/segmented_customers.csv", index = False)
