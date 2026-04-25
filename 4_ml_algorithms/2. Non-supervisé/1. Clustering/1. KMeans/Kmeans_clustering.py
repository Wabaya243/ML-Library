import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import mode

# -----------------------------
# 1. Chargement et nettoyage
# -----------------------------
dataset = pd.read_csv('Live.csv')

# Suppression colonnes vides
dataset.drop(['Column1', 'Column2', "Column3", 'Column4'], axis=1, inplace=True)

######### On veut voir les valeurs unique dans cette variable categoriel
dataset['status_id'].unique()

######## le nombre 
len(dataset['status_id'].unique())

## 6997 sur 7050 c'est trop on c'est pas vraimet une variable qu'on peut utiliser
## Pareils pour status_published

# Suppression des colonnes inutiles
dataset = dataset.drop(['status_id', 'status_published'], axis=1)

# -----------------------------
# 2. Encodage et préparation
# -----------------------------
x = dataset.copy()
y = dataset['status_type']  # vrai label

# Encodage de la variable catégorielle
Le = LabelEncoder()
x['status_type'] = Le.fit_transform(x['status_type'])
y = Le.transform(y)

# Feature scaling
cols = x.columns
Ms = MinMaxScaler()
x = Ms.fit_transform(x)
x = pd.DataFrame(x, columns=cols)

# -----------------------------
# 3. Clustering avec KMeans
# -----------------------------
kmeans = KMeans(n_clusters=4, init="k-means++", random_state=0)
kmeans.fit(x)
labels = kmeans.labels_

# -----------------------------
# 4. Évaluation
# -----------------------------
# ARI et NMI = bonnes métriques non supervisées
print("ARI (Adjusted Rand Index):", adjusted_rand_score(y, labels))
print("NMI (Normalized Mutual Info):", normalized_mutual_info_score(y, labels))

# Accuracy corrigée via mapping cluster→classe
label_map = {}
for i in range(kmeans.n_clusters):
    mask = (labels == i)
    if np.sum(mask) > 0:
        label_map[i] = mode(y[mask], keepdims=True)[0][0]

mapped_labels = np.array([label_map[l] for l in labels])
print("Accuracy corrigée:", np.mean(mapped_labels == y))

# -----------------------------
# 5. Visualisation PCA 3D
# -----------------------------
pca = PCA(n_components=3)
x_pca3 = pca.fit_transform(x)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_pca3[labels == 0, 0], x_pca3[labels == 0, 1], x_pca3[labels == 0, 2], c='red', label='Cluster 1')
ax.scatter(x_pca3[labels == 1, 0], x_pca3[labels == 1, 1], x_pca3[labels == 1, 2], c='blue', label='Cluster 2')
ax.scatter(x_pca3[labels == 2, 0], x_pca3[labels == 2, 1], x_pca3[labels == 2, 2], c='green', label='Cluster 3')
ax.scatter(x_pca3[labels == 3, 0], x_pca3[labels == 3, 1], x_pca3[labels == 3, 2], c='cyan', label='Cluster 4')

# Centroïdes projetés en PCA
centers_pca3 = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca3[:, 0], centers_pca3[:, 1], centers_pca3[:, 2],
           s=400, c='black', marker='X', label='Centroids')

ax.set_title("Clusters projetés en 3D (PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.show()

# -----------------------------
# 6. Visualisation t-SNE 2D
# -----------------------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
x_tsne = tsne.fit_transform(x)

plt.figure(figsize=(8,6))
plt.scatter(x_tsne[:,0], x_tsne[:,1], c=labels, cmap='tab10', s=20)
plt.title("Clusters visualisés avec t-SNE (2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()
