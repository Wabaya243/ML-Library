# K-Means

import numpy as np	
import matplotlib.pyplot as plt
import pandas as pd


#importer le dataset

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# Utiliser la methode elbow pour trouver le nombre optimal de clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)   
    
plt.plot(range(1,11), wcss)
plt.title('La methode elbow')
plt.xlabel('Nombre de cluster')
plt.ylabel('WCSS')
plt.show()

# Construire les model

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 5, init= 'k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(x)

#visualiser les resultats

plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0, 1], c='red', label = 'Cluster 1' )
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1, 1], c='blue', label = 'Cluster 2' )
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2, 1], c='green', label = 'Cluster 3' )
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3, 1], c='cyan', label = 'Cluster 4' )
plt.scatter(x[y_kmeans == 4, 0],x[y_kmeans == 4, 1], c='yellow', label = 'Cluster 5' )
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')

plt.title('Cluster des clients')
plt.xlabel('Salaire Annuel')
plt.ylabel('Spanning Score')
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Affichage de chaque cluster
ax.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 0, c='red', label='Cluster 1')
ax.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 1, c='blue', label='Cluster 2')
ax.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], 2, c='green', label='Cluster 3')
ax.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], 3, c='cyan', label='Cluster 4')
ax.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], 4, c='yellow', label='Cluster 5')

# Affichage des centroïdes
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], [0,1,2,3,4], s=300, c='black', label='Centroids')

ax.set_xlabel('Salaire Annuel')
ax.set_ylabel('Spending Score')
ax.set_zlabel('Cluster ID (Z)')

plt.title('Cluster des clients (Vue 3D)')
ax.legend()
plt.show()
