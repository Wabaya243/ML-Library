# Importation des bibliothèques nécessaires
import numpy as np                     # Pour les calculs numériques
import pandas as pd                    # Pour la manipulation de données
import matplotlib.pyplot as plt        # Pour la visualisation graphique
import seaborn as sns                  # Pour des visualisations avancées
from sklearn.model_selection import train_test_split   # Pour séparer les données en train/test
import category_encoders as ce         # Pour encoder les variables catégorielles
from sklearn.preprocessing import RobustScaler         # Pour la normalisation des variables numériques
from sklearn.naive_bayes import GaussianNB             # Pour le modèle Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix # Pour évaluer les performances

# Chargement du dataset sans noms de colonnes (header=None) 
# et séparation par virgule + espace avec regex ",\s"
dataset = pd.read_csv('adult.csv', header=None, sep=',\s')

# Définition des noms de colonnes (selon la documentation du dataset Adult Income)
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
             'hours_per_week', 'native_country', 'income']

# Attribution des noms de colonnes au dataset
dataset.columns = col_names

# Affichage des informations sur le dataset (types, valeurs manquantes, etc.)
dataset.info()

# Sélection des variables catégorielles (type 'O' = object/string)
categorical = [var for var in dataset.columns if dataset[var].dtype == 'O']

print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

# Affiche les premières lignes des colonnes catégorielles
dataset[categorical].head()

# Vérifie le nombre de valeurs manquantes dans les variables catégorielles
dataset[categorical].isnull().sum()

# Compte la fréquence brute de chaque modalité (valeurs uniques) des variables catégorielles
for var in categorical:
    print(dataset[var].value_counts())

# Calcule la fréquence relative (proportion) des modalités
for var in categorical: 
    print(dataset[var].value_counts() / np.float64(len(dataset)))

# Vérifie les labels uniques de la variable "workclass"
dataset.workclass.unique()

# Remplace les valeurs "?" par NaN dans la colonne "workclass"
dataset['workclass'].replace('?', np.NaN, inplace=True)

# Vérifie les labels uniques de "occupation"
dataset.occupation.unique()

# Remplace les valeurs "?" par NaN dans la colonne "occupation"
dataset['occupation'].replace('?', np.NaN, inplace=True)

# Remplace aussi les "?" par NaN dans "native_country"
dataset['native_country'].replace('?', np.NaN, inplace=True)

# Vérifie le nombre de valeurs manquantes après remplacement
dataset[categorical].isnull().sum()

# Vérifie la cardinalité (nombre de catégories uniques) de chaque variable catégorielle
for var in categorical:
    print(var, ' contains ', len(dataset[var].unique()), ' labels')

# Identification des variables numériques (tout sauf 'O')
numerical = [var for var in dataset.columns if dataset[var].dtype != 'O']

print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

# Vérifie s’il y a des valeurs manquantes dans les variables numériques
dataset[numerical].isnull().sum()

### Déclaration des variables dépendantes et indépendantes
x = dataset.drop(["income"], axis=1)   # Variables explicatives
y = dataset['income']                  # Variable cible

# Séparation en train et test (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#### Pré-traitement / Feature Engineering ####

# Vérifie les types des variables dans X_train
x_train.dtypes

# Récupère les colonnes catégorielles de X_train
categorical = [col for col in x_train.columns if x_train[col].dtypes == 'O']
print(categorical)

# Récupère les colonnes numériques
numerical = [col for col in x_train.columns if x_train[col].dtypes != 'O']
print(numerical)

# Affiche la proportion de valeurs manquantes dans les variables catégorielles
x_train[categorical].isnull().mean()

# Identifie les colonnes avec des valeurs manquantes
for col in categorical:
    if x_train[col].isnull().mean() > 0:
        print(col, (x_train[col].isnull().mean()))

# Remplace les valeurs manquantes par la modalité la plus fréquente (mode) dans train/test
for df2 in [x_train, x_test]:
    df2['workclass'].fillna(x_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(x_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(x_train['native_country'].mode()[0], inplace=True)  

# Vérifie qu’il n’y a plus de valeurs manquantes
x_train[categorical].isnull().sum()

# Encodage des variables catégorielles avec One-Hot-Encoding (création de colonnes binaires)
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 
                                 'relationship', 'race', 'sex', 'native_country'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

##### Mise à l’échelle (Feature Scaling) #####

cols = x_train.columns  # Sauvegarde des noms de colonnes

scaler = RobustScaler() # Création du scaler (moins sensible aux valeurs extrêmes)

# Transformation des données
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Conversion en DataFrame avec colonnes et index
x_train = pd.DataFrame(x_train, columns=cols, index=y_train.index)
x_test = pd.DataFrame(x_test, columns=cols, index=y_test.index)

##### Modélisation #####

# Création du modèle Naive Bayes
gnb = GaussianNB()

# Entraînement du modèle
gnb.fit(x_train, y_train)

# Prédictions sur le jeu de test
y_pred = gnb.predict(x_test)

#### Évaluation du modèle ####

# Calcul de la précision globale
accuracy = accuracy_score(y_test, y_pred)
print("La précision du modèle est : ", accuracy)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

# Détails de la matrice de confusion
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# Visualisation de la matrice de confusion avec heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# Rapport de classification (précision, rappel, F1-score)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Probabilités prédites pour les 10 premières lignes
y_pred_prob = gnb.predict_proba(x_test)[0:10]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
print(y_pred_prob_df)

# Probabilités de prédiction pour la classe "salaire >50K"
y_pred1 = gnb.predict_proba(x_test)[:, 1]

# Ajustement de la taille de police pour les graphiques
plt.rcParams['font.size'] = 12

# Histogramme des probabilités prédites (10 bins)
plt.hist(y_pred1, bins=10)

# Ajout des titres et légendes
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')





