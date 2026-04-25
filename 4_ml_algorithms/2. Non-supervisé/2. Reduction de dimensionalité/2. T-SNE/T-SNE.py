# Import des librairies nécessaires
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

# Chargement des datasets d'entraînement et de test
train = pd.read_csv('Data/train.csv')   # On lit les données d'entraînement
test = pd.read_csv("Data/test.csv")     # On lit les données de test (non utilisées directement ici)

# Préparation de la variable cible (le prix de vente)
y_train = train['SalePrice']  # On garde la colonne SalePrice comme variable cible

# Comme c’est un problème de régression continue, on découpe les valeurs en "bins"
# Cela permet de stratifier les prix et de mieux visualiser les résultats de t-SNE
y_train_stratified = pd.cut(y_train, bins=10, labels=False)

# Colonnes à supprimer (issues des recommandations AutoViz ou pour éviter le bruit)
# - Id : identifiant unique, inutile pour l'apprentissage
# - Alley, PoolQC, MiscFeature : beaucoup de valeurs manquantes → peu informatif
# - GarageYrBlt, GarageArea : souvent colinéaires avec d'autres infos sur le garage
# - SalePrice : variable cible, qu’on doit retirer des features
col_to_drop = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'GarageYrBlt','GarageArea','SalePrice']

train = train.drop(col_to_drop,axis=1)  # On supprime ces colonnes

# Fonction pour gérer les valeurs manquantes
def fill_null_values(df):
    """
    On traite les valeurs manquantes différemment selon le type de colonne :
    - Pour les colonnes numériques : remplacement par la moyenne (stratégie simple et standard)
    - Pour les colonnes catégorielles (type 'object') : remplacement par 'unknown' (nouvelle catégorie)
    """
    for column in df.columns:
        if df[column].dtype == 'object':  
            df[column].fillna('unknown', inplace=True)  # On crée une catégorie spéciale pour les NaN
        else:
            df[column].fillna(df[column].mean(), inplace=True)  # On impute avec la moyenne
    return df

# Application de la fonction de remplissage des NaN sur train
train = fill_null_values(train)

# Fonction pour regrouper les catégories rares dans une colonne
def group_rare_categories(df, threshold=0.05, replacement='Other'):
    """
    Pourquoi faire ça ?
    → Certaines catégories apparaissent très rarement (< 5% des données).
    → Elles n’apportent pas assez d’information et risquent de nuire à la généralisation.
    → On les regroupe donc dans une seule catégorie "Other".
    """
    for column in df.select_dtypes(include='object'):  # On applique uniquement aux colonnes catégorielles
        counts = df[column].value_counts(normalize=True)  # Fréquences relatives
        rare_categories = counts[counts < threshold].index.tolist()  # Liste des catégories rares
        
        # Si trop de catégories rares existent, on les regroupe
        if len(rare_categories) > 2:
            df[column] = df[column].apply(lambda x: replacement if x in rare_categories else x)
    return df

# Avant regroupement : on compte le nombre de catégories uniques par colonne
unique_categories = train.select_dtypes(include='object').nunique()
print(unique_categories)

# Application du regroupement
train = group_rare_categories(train)

# Après regroupement : on vérifie si le nombre de catégories a diminué
unique_categories = train.select_dtypes(include='object').nunique()
print(unique_categories)

# Séparation des colonnes par type (utile pour encodage et scaling)
object_columns = train.select_dtypes(include='object').columns.tolist()  # colonnes catégorielles
numerical_columns = train.select_dtypes(include='number').columns.tolist()  # colonnes numériques

# Normalisation des colonnes numériques
# Pourquoi ? → Certaines features ont des échelles très différentes (ex : surface en m² vs nombre de pièces)
# → StandardScaler met toutes les colonnes à une moyenne 0 et variance 1 → facilite l’apprentissage
scaler = StandardScaler()
train[numerical_columns] = scaler.fit_transform(train[numerical_columns])

# Encodage des colonnes catégorielles
# Pourquoi pas OneHotEncoder ? → t-SNE est sensible à la dimensionnalité, donc LabelEncoder est plus compact.
label_encoder = LabelEncoder()
for col in object_columns:
    train[col] = label_encoder.fit_transform(train[col])  # Chaque catégorie devient un entier

# Application de t-SNE pour réduire la dimension à 2D
# Pourquoi t-SNE ? → Permet de visualiser des données complexes en les projetant dans un espace 2D
# → Les points proches en haute dimension le restent en 2D
tsne = TSNE(n_components=2,perplexity=5,max_iter=5000,random_state=42)
train_2d = tsne.fit_transform(train)

# Fonction de visualisation des résultats t-SNE
def visualize_tsne(data,target):
    """
    Visualisation en 2D avec t-SNE :
    - Chaque point représente une maison
    - La couleur est basée sur le prix (stratifié en 10 bins)
    - Permet de voir si t-SNE réussit à séparer les groupes de prix
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(data[:,0], data[:,1],
            c=target.values,  # Couleur selon les bins de SalePrice
            edgecolor='none', 
            alpha=0.80,       # Transparence pour mieux voir les points qui se chevauchent
            s=20)             # Taille des points
    plt.axis('off')  # Pas besoin des axes pour ce type de visualisation
    plt.show()

# Visualisation initiale avec perplexity=5
visualize_tsne(train_2d,y_train_stratified)

# Expérimentation avec plusieurs valeurs de perplexity
# Pourquoi ? → La perplexity contrôle la manière dont t-SNE équilibre la proximité locale vs globale.
# Valeurs faibles : plus d’accent sur les relations locales (clusters petits mais clairs)
# Valeurs élevées : prise en compte de la structure globale
perplexities =[5,10,20,30,40,50]

for perplexity in perplexities:
    print(f"perplexity: {perplexity}")
    tsne = TSNE(n_components=2,perplexity=perplexity,max_iter=5000,random_state=42)
    train_2d = tsne.fit_transform(train)
    visualize_tsne(train_2d,y_train_stratified)


'''
Dans ce cas, il semble que les valeurs de perplexité 20-30 soient les plus efficaces pour mettre en évidence les clusters et les modèles possibles dans la valeur cible.

Interprétation des graphiques
Il y a quelques points à prendre en compte lors de l'interprétation des graphiques t-SNE.

Tout d'abord, il est impossible de comparer les tailles de clusters.

Le t-SNE dilate les clusters denses et contracte les clusters clairsemés dans l'espace de dimension inférieure afin de maintenir et d'adapter la densité dans l'espace de dimension supérieure. Par conséquent, les tailles de clusters sont uniformes, ce qui les rend non comparables.

Deuxièmement, soyez prudent quant à l'interprétation de la distance entre les clusters.

Bien que vous puissiez ajuster la « perplexité » pour obtenir une représentation correcte de la distance, les données réelles peuvent nécessiter des valeurs de « perplexité » différentes pour capturer la distance entre tous les clusters. Cela signifie que la distance entre les clusters peut ne pas nécessairement signifier quelque chose.

Enfin, le bruit aléatoire peut apparaître sous forme de clusters.

Lors de l'utilisation de valeurs de perplexité faibles, le bruit aléatoire peut apparaître sous forme de clusters. Il est donc important de vérifier avec différentes valeurs de perplexité si les clusters restent relativement constants. Si le cluster disparaît à mesure que la perplexité augmente, il se peut que vous ayez observé un bruit aléatoire.

Malgré ces difficultés, l'utilisation de t-SNE dans le cadre de votre analyse exploratoire des données peut vous aider à découvrir des schémas cachés. Vous pouvez utiliser ces informations nouvellement découvertes pour décider des prochaines étapes.
'''