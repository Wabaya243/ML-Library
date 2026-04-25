# ============================
# Importation des bibliothèques
# ============================

import warnings
warnings.filterwarnings('ignore')   # Permet d'ignorer les avertissements (par ex. warnings de seaborn)

import numpy as np                  # Librairie pour les calculs numériques (vecteurs, matrices…)
import pandas as pd                 # Librairie pour la manipulation de données (tableaux, csv, etc.)
import matplotlib.pyplot as plt     # Librairie de visualisation basique (graphiques, histogrammes…)
import seaborn as sns               # Librairie de visualisation avancée (graphiques plus esthétiques)

# Outils pour le Machine Learning
from sklearn.decomposition import PCA                   # Réduction de dimension (PCA)
from sklearn.svm import SVC                             # Support Vector Machine (SVM)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Évaluation
from sklearn.model_selection import train_test_split    # Séparation train/test
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Encodage + normalisation
from sklearn.impute import SimpleImputer                # Gestion des valeurs manquantes

# Outils statistiques
from scipy.stats import zscore, iqr


# ============================
# Chargement et préparation des données
# ============================

# Lecture du fichier CSV contenant les données des véhicules
dataset = pd.read_csv('Data/vehicle-2.csv')

# Transformation de la colonne "class" (qui contient des catégories textuelles : bus, car, van…)
# en valeurs numériques (0, 1, 2, etc.)
Le = LabelEncoder()
dataset['class'] = Le.fit_transform(dataset['class'])

# Création d'une copie du dataset pour ne pas modifier l’original
new_df = dataset.copy()

# Sélection des colonnes explicatives (les 19 premières colonnes contiennent les attributs du véhicule)
x = new_df.iloc[:, 0:19]

# Imputation des valeurs manquantes : si une valeur est absente, on la remplace par la médiane de la colonne
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
transformed_values = imputer.fit_transform(x)

# Reconstruction du DataFrame avec les valeurs imputées
column = x.columns
new_df = pd.DataFrame(transformed_values, columns=column)

# Vérification du nombre de valeurs manquantes avant et après imputation
print("Valeurs manquantes avant imputation :", dataset.isnull().sum())
print("\nValeurs manquantes après imputation :", new_df.isnull().sum())


# ============================
# Analyse exploratoire des données
# ============================

# Application d’un style graphique plus lisible
sns.set_style("whitegrid")

# Histogrammes de toutes les colonnes → permettent de voir la répartition des données
new_df.hist(bins=20, figsize=(60,40), color='lightblue', edgecolor='red')
plt.show()

# Distribution de certaines variables précises pour observer la symétrie (skewness)
f, ax = plt.subplots(1, 6, figsize=(30,5))
sns.distplot(new_df["scaled_variance.1"], bins=10, ax=ax[0])
sns.distplot(new_df["scaled_variance"], bins=10, ax=ax[1])
sns.distplot(new_df["skewness_about.1"], bins=10, ax=ax[2])
sns.distplot(new_df["skewness_about"], bins=10, ax=ax[3])
sns.distplot(new_df["scatter_ratio"], bins=10, ax=ax[5])
f.savefig('subplot.png')

# Calcul de l’asymétrie (skewness) des colonnes
skewValue = new_df.skew()
print("Valeur de skewness (asymétrie) pour chaque attribut : ", skewValue)


# ============================
# Détection et traitement des valeurs aberrantes (outliers)
# ============================

# Boxplot global (toutes les colonnes) pour repérer rapidement les valeurs extrêmes
sns.boxplot(data=new_df, orient="h")
plt.show()

# Exemple de boxplots pour quelques colonnes spécifiques
plt.figure(figsize=(20,15))
plt.subplot(3,3,1); sns.boxplot(x=new_df['pr.axis_aspect_ratio'], color='orange')
plt.subplot(3,3,2); sns.boxplot(x=new_df['skewness_about'], color='purple')
plt.subplot(3,3,3); sns.boxplot(x=new_df['scaled_variance'], color='brown')
plt.show()

# Calcul des quartiles (Q1 et Q3) et de l’intervalle interquartile (IQR)
Q1 = new_df.quantile(0.25)
Q3 = new_df.quantile(0.75)
IQR = Q3 - Q1
print("Intervalle interquartile (IQR) :", IQR)

# Suppression des valeurs aberrantes situées en dehors de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
cleandf = new_df[~((new_df < (Q1 - 1.5 * IQR)) | (new_df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Taille du dataset après suppression des outliers :", cleandf.shape)

# Vérification après suppression
plt.figure(figsize=(20,15))
plt.subplot(2,2,1); sns.boxplot(x=cleandf['pr.axis_aspect_ratio'], color='orange')
plt.subplot(2,2,2); sns.boxplot(x=cleandf['skewness_about'], color='purple')
plt.subplot(2,2,3); sns.boxplot(x=cleandf['scaled_variance'], color='brown')
plt.subplot(2,2,4); sns.boxplot(x=cleandf['radius_ratio'], color='red')
plt.show()


# ============================
# Analyse de corrélations
# ============================

def correlation_heatmap(dataframe, l, w):
    """Affiche une carte de chaleur des corrélations entre variables numériques"""
    correlation = dataframe.corr()
    plt.figure(figsize=(l, w))
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='viridis')
    plt.title('Corrélation entre les différentes variables')
    plt.show()

# On supprime la colonne "class" (la cible) avant PCA → car PCA s’applique seulement aux variables explicatives
cleandf = new_df.drop('class', axis=1)

# Heatmap et Pairplot
correlation_heatmap(cleandf, 30, 15)
sns.pairplot(cleandf, diag_kind="kde")

# Distribution des classes (car, bus, van…)
print("Nombre d’exemples par classe :", new_df['class'].value_counts())
sns.countplot(data=new_df, x='class')
plt.show()


# ============================
# Réduction de dimension via PCA
# ============================

# Séparation des données en X (attributs) et y (classes)
x = new_df.iloc[:, 0:18].values
y = new_df.iloc[:, -1].values

# Standardisation des données (obligatoire avant PCA)
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# Calcul de la matrice de covariance
cov_matrix = np.cov(x_std.T)
print("Taille de la matrice de covariance :", cov_matrix.shape)

# Décomposition en valeurs propres (eigenvalues) et vecteurs propres (eigenvectors)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Paires (valeur propre, vecteur propre), triées par ordre décroissant
eig_pairs = [(eigenvalues[i], eigenvectors[:,i]) for i in range(len(eigenvalues))]
eig_pairs.sort(reverse=True)

eigvalues_sorted = [pair[0] for pair in eig_pairs]
eigvectors_sorted = [pair[1] for pair in eig_pairs]

print('Valeurs propres triées :', eigvalues_sorted)

# Variance expliquée par chaque composante
tot = sum(eigenvalues)
var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_explained)

# Graphique : variance expliquée
plt.bar(range(1,19), var_explained, alpha=0.5, align='center', label='Variance expliquée (individuelle)')
plt.step(range(1,19), cum_var_exp, where='mid', label='Variance expliquée (cumulative)')
plt.ylabel('Proportion de variance expliquée')
plt.xlabel('Composantes principales')
plt.legend(loc='best')
plt.show()

# Projection des données dans l’espace PCA réduit (8 dimensions)
P_reduce = np.array(eigvectors_sorted[0:8])
X_std_8D = np.dot(x_std, P_reduce.T)

# Conversion en DataFrame pour visualisation
reduced_pca = pd.DataFrame(X_std_8D)
sns.pairplot(reduced_pca, diag_kind='kde')


# ============================
# Classification avec SVM (avant et après PCA)
# ============================

# Découpage en train/test (70% entraînement, 30% test)
Orig_X_train, Orig_X_test, Orig_y_train, Orig_y_test = train_test_split(x_std, y, test_size=0.30, random_state=1)
pca_X_train, pca_X_test, pca_y_train, pca_y_test = train_test_split(reduced_pca, y, test_size=0.30, random_state=1)

# Entraînement du SVM sur les données originales
svc = SVC()
svc.fit(Orig_X_train, Orig_y_train)
Orig_y_predict = svc.predict(Orig_X_test)

# Entraînement du SVM sur les données réduites par PCA
svc1 = SVC()
svc1.fit(pca_X_train, pca_y_train)
pca_y_predict = svc1.predict(pca_X_test)

# Comparaison des scores
print("Précision du modèle sur données originales :", svc.score(Orig_X_test, Orig_y_test))
print("Précision du modèle après PCA (8 dimensions) :", svc1.score(pca_X_test, pca_y_test))


# ============================
# Évaluation avec matrices de confusion et rapports
# ============================

def draw_confmatrix(y_test, yhat, str1, str2, str3, datatype):
    """Affiche une matrice de confusion avec heatmap"""
    cm = confusion_matrix(y_test, yhat, [0,1,2])
    print("Matrice de confusion pour", datatype, ":\n", cm)
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=[str1, str2, str3], yticklabels=[str1, str2, str3])
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')
    plt.show()

# Affichage des matrices de confusion
draw_confmatrix(Orig_y_test, Orig_y_predict, "Van", "Car", "Bus", "Données originales")
draw_confmatrix(pca_y_test, pca_y_predict, "Van", "Car", "Bus", "Données PCA réduites")

# Rapports de classification détaillés
print("Rapport de classification (Données originales) :\n", classification_report(Orig_y_test, Orig_y_predict))
print("Rapport de classification (Données PCA) :\n", classification_report(pca_y_test, pca_y_predict))
