# Import des librairies principales
import numpy as np  # Librairie pour l'algèbre linéaire et les tableaux multidimensionnels
import pandas as pd  # Librairie pour la manipulation de données (tableaux, CSV, etc.)
import matplotlib.pyplot as plt  # Librairie pour tracer des graphiques et visualiser des données
import seaborn as sns  # Librairie de visualisation avancée, construite sur matplotlib

# Import des outils de scikit-learn
from sklearn import datasets  # Pour charger des jeux de données standards comme MNIST
from sklearn.manifold import TSNE  # Algorithme t-SNE pour la réduction de dimensions et la visualisation

# Chargement du dataset MNIST depuis OpenML
data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)  
# Ici, 'mnist_784' correspond au dataset MNIST avec 784 pixels (28x28)
# return_X_y=True retourne directement (X, y) au lieu d’un objet dataset

# Séparation des features (pixels) et des labels (chiffres)
pixel_data, targets = data  
targets = targets.astype(int)  # Conversion des labels en entiers (ils sont chargés comme chaînes de caractères)

# Conversion des données en tableau NumPy pour faciliter le reshape et l’indexation
pixel_data = pixel_data.to_numpy()

# Reshape d’une seule image : on prend la 6ème image (index 5) et on la met en format 28x28
single_image = pixel_data[5].reshape(28, 28)

# Affichage de cette image en niveaux de gris
plt.imshow(single_image, cmap='gray')
plt.title(f"image du texte : {targets[5]}", fontsize=5)  # On affiche aussi le chiffre réel comme titre
plt.show()

### Transformation Tsne (réduction de dimensions pour visualisation)

tsne = TSNE(n_components=2, random_state=42)  
# On instancie t-SNE pour réduire les 784 dimensions (pixels) en 2 dimensions (X et Y) pour visualisation
# random_state=42 permet de rendre les résultats reproductibles

x_transformed = tsne.fit_transform(pixel_data[:3000, :])  
# Application du t-SNE sur seulement 3000 images (sinon trop lourd à calculer)
# pixel_data[:3000, :] -> on prend les 3000 premières lignes (images) et toutes les colonnes (pixels)

# Conversion du résultat t-SNE en DataFrame pour faciliter la manipulation et la visualisation
tsne_df = pd.DataFrame(
    np.column_stack((x_transformed, targets[:3000])), 
    columns=['X', 'Y', "Targets"]
)
# np.column_stack assemble X, Y (t-SNE) + leurs labels correspondants

tsne_df.loc[:, "Targets"] = tsne_df.Targets.astype(int)  
# On s'assure que la colonne "Targets" est bien de type entier

### Visualisation des données transformées par t-SNE

plt.figure(figsize=(12,6))  # Définit la taille de la figure

g = sns.FacetGrid(data=tsne_df, hue='Targets', height=8)  
# FacetGrid permet de tracer un nuage de points colorié par classe ("Targets")

g.map(plt.scatter, 'X', 'Y').add_legend()  
# On trace les points (X, Y) et on ajoute une légende pour les chiffres

plt.show()  # Affiche le graphe final








