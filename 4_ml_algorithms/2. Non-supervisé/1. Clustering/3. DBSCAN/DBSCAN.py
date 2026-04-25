# Importation des bibliothèques nécessaires
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
# Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN


### 📌 1. Chargement des données
dataset = pd.read_csv('Data/Mall_Customers.csv')

# Vérifier les valeurs manquantes
dataset.isnull().sum()
dataset.info()


dataset.rename(columns={'Annual Income (k$)':'Income','Spending Score (1-100)':'SpendScore'},inplace=True)


sns.pairplot(dataset)
##### nous pouvons voir que customer_id n'est pas lié a Income
dataset = dataset.drop('CustomerID', axis=1)

# On encode la variable categoriel
Le = LabelEncoder()
dataset['Gender'] = Le.fit_transform(dataset['Gender'])

sns.heatmap(dataset.corr())
plt.show()

plt.figure(figsize=(7,7))
size=dataset['Gender'].value_counts()
label=['Female','Male']
color=['Pink','Blue']
explode=[0,0.1]
plt.pie(size,explode=explode,labels=label,colors=color,shadow=True)
plt.legend()
plt.show()

#### nous voyons que les femmes visitent plus les mall que les hommes


plt.figure(figsize=(10,5))
sns.countplot(x='Age', data=dataset)
plt.xticks(rotation=90)
plt.show()

### les utilisateurs de 25-40 visitent les plus

sns.boxplot(x='Gender', y='SpendScore', data=dataset)
plt.show()
### Ce diagramme montre le score de dépenses moyen des femmes et des hommes. Nous pouvons observer que le score de dépenses moyen des femmes est supérieur à celui des hommes, qu'elles ont un score de dépenses plus élevé que les hommes et que leur score de dépenses le plus bas est supérieur à celui des hommes.


plt.figure(figsize=(10,5))
sns.countplot(x='Income', data=dataset)
plt.show()

### Peoples of salary 54k and 78k are the mostly visited persons in mall.

plt.bar(dataset['Income'],dataset['SpendScore'])
plt.title('Spendscore over income',fontsize=20)
plt.xlabel('Income')
plt.ylabel('Spendscore')

## les personn avec un saleur e 20-40 et 70k-100k sont les plus depensié

##### Creation du Model DBSCAN (Density Based Spacial Clustering of Applications with noise)

x = dataset.iloc[:, [2,3]].values

db = DBSCAN(eps=3, min_samples=4, metric='euclidean')

models = db.fit(x)

label = models.labels_

from sklearn import metrics

### identifer les points qui ont fait un point noyaux
sample_cores = np.zeros_like(label, dtype=bool)

sample_cores[db.core_sample_indices_]=True

### Calculé les nombres des clusters

n_clusters = len(set(label)) - (1 if -1 in label else 0)
print('Nombre des clusters:',n_clusters)

y_means = db.fit_predict(x)
plt.figure(figsize=(12,6))
plt.scatter(x[y_means == 0,0], x[y_means == 0,1], s=50 , c='pink')
plt.scatter(x[y_means == 1,0], x[y_means == 1,1], s=50 , c='green')
plt.scatter(x[y_means == 2,0], x[y_means == 2,1], s=50 , c='cyan')
plt.scatter(x[y_means == 3,0], x[y_means == 3,1], s=50 , c='blue')
plt.scatter(x[y_means == 4,0], x[y_means == 4,1], s=50 , c='violet')
plt.scatter(x[y_means == 5,0], x[y_means == 5,1], s=50 , c='black')
plt.scatter(x[y_means == 6,0], x[y_means == 6,1], s=50 , c='purple')
plt.scatter(x[y_means == 7,0], x[y_means == 7,1], s=50 , c='orange')
plt.scatter(x[y_means == 8,0], x[y_means == 8,1], s=50 , c='yellow')
plt.xlabel('Salaire Annuel')
plt.ylabel('Depnse en score 1-100')
plt.title('Données des clusters')
plt.show()

### Hierchical clustering 


import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Customers')
plt.ylabel('Ecuclidean Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 9, memory='euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 50, c = 'pink')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 50, c = 'yellow')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 50, c = 'cyan')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 50, c = 'magenta')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 50, c = 'orange')
plt.scatter(x[y_hc == 5, 0], x[y_hc == 5, 1], s = 50, c = 'blue')
plt.scatter(x[y_hc == 6, 0], x[y_hc == 6, 1], s = 50, c = 'red')
plt.scatter(x[y_hc == 7, 0], x[y_hc == 7, 1], s = 50, c = 'black')
plt.scatter(x[y_hc == 8, 0], x[y_hc == 8, 1], s = 50, c = 'violet')


plt.title('Hierarchial Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()















