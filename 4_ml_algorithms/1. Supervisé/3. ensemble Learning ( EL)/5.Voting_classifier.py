# =====================================================================
# 📌 IMPORTATION DES LIBRAIRIES
# =====================================================================
# Numpy → calcul scientifique (vecteurs, matrices)
# Pandas → manipulation des dataframes
# Seaborn + Matplotlib + Plotly → visualisations
# Tkinter → parfois utile pour l’interface (non utilisé ici)
# Sklearn → outils machine learning (prétraitement, modèles, évaluation)
# XGBoost & CatBoost → algorithmes de boosting performants
# =====================================================================
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import tkinter
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

# =====================================================================
# 📂 CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================================
# Dataset : Telco Customer Churn → prédire si un client va résilier (Churn)
# Variables : informations client (contrat, internet, facturation, etc.)
# =====================================================================
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Définir une palette personnalisée (utile pour garder une identité visuelle)
palette = ['#008080','#FF6347', '#E50000', '#D2691E'] 

# Supprimer l’ID client car c’est un identifiant unique → pas utile pour le ML
dataset = dataset.drop('customerID', axis=1)

# Vérification de la colonne "TotalCharges" (certaines valeurs sont vides → chaînes vides)
step1 = [len(i.split()) for i in dataset['TotalCharges']]
step2 = [i for i in range(len(step1)) if step1[i] != 1]  
print('Number of entries with empty string: ', len(step2))

# Suppression des lignes problématiques (valeurs manquantes dans TotalCharges)
dataset = dataset.drop(step2, axis = 0).reset_index(drop=True)

# Conversion de TotalCharges en float pour pouvoir l’utiliser dans les modèles
dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)

# =====================================================================
# 🔄 NETTOYAGE DES DOUBLONS
# =====================================================================
print('Number of duplicated values in training dataset: ', dataset.duplicated().sum())
dataset.drop_duplicates(inplace=True)
print("Duplicated values dropped successfully")
print("*" * 100)

# =====================================================================
# 📊 SÉPARATION DES VARIABLES
# =====================================================================
# Objectif : distinguer les variables catégorielles et numériques
# Règle choisie : si une colonne a >6 modalités → numérique (par ex. "MonthlyCharges")
# =====================================================================
columns = list(dataset.columns)

categoric_columns = []
numeric_columns = []

for i in columns:
    if len(dataset[i].unique()) > 6:
        numeric_columns.append(i)
    else:
        categoric_columns.append(i)

# On retire la variable cible "Churn" de la liste des catégorielles
categoric_columns = categoric_columns[:-1] 

print('Numerical features: ', numeric_columns)
print('Categorical features: ', categoric_columns)

# =====================================================================
# 🎯 ENCODAGE DE LA VARIABLE CIBLE
# =====================================================================
# La cible "Churn" est Yes/No → on encode en 0/1 avec LabelEncoder
# =====================================================================
le = LabelEncoder()
dataset[['Churn']] = dataset[['Churn']].apply(le.fit_transform)

# Séparation features (X) et target (y)
x = dataset.drop('Churn', axis=1)
y = dataset['Churn']

# =====================================================================
# ✂️ DIVISION EN TRAIN/TEST
# =====================================================================
# Split 75% entraînement et 25% test → random_state=42 pour reproductibilité
# =====================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# =====================================================================
# ⚖️ STANDARDISATION DES VARIABLES NUMÉRIQUES
# =====================================================================
# Objectif : mettre toutes les variables numériques sur la même échelle
# Sinon les modèles sensibles aux échelles (KNN, SVM, réseaux de neurones) peuvent mal apprendre
# =====================================================================
sc = StandardScaler()
sc.fit_transform(x_train[numeric_columns])
sc.transform(x_test[numeric_columns])

# =====================================================================
# 🔤 ENCODAGE DES VARIABLES CATÉGORIELLES
# =====================================================================
# Objectif : transformer les colonnes catégorielles (texte) en colonnes binaires (OneHot)
# Exemple : "InternetService" → DSL, Fiber, No → 3 colonnes (0/1)
# handle_unknown='ignore' évite les crash si une modalité nouvelle apparaît en test
# =====================================================================
print(categoric_columns)
transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), 
     ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
      'PhoneService', 'MultipleLines', 'InternetService', 
      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
      'TechSupport', 'StreamingTV', 'StreamingMovies', 
      'Contract', 'PaperlessBilling', 'PaymentMethod']))

# Transformation de X_train
transformed = transformer.fit_transform(x_train)
transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
transformed_df.index = x_train.index
x_train = pd.concat([x_train, transformed_df], axis=1)
x_train.drop(categoric_columns, axis=1, inplace=True)

# Transformation de X_test
transformed = transformer.transform(x_test)
transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
transformed_df.index = x_test.index
x_test = pd.concat([x_test, transformed_df], axis=1)
x_test.drop(categoric_columns, axis=1, inplace=True)







# ------------------------
# 🤖 Import des modèles utiles
# ------------------------
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ------------------------
# 🧱 Définition des modèles de base
# ------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)  # probas nécessaires pour voting soft
lr = LogisticRegression(max_iter=1000, random_state=42)

# ------------------------
# 🏗️ Création du VotingClassifier
# ------------------------
# "soft" = moyenne des probabilités
# "hard" = vote majoritaire des classes
VC_light = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svc', svc), ('lr', lr)],
    voting='soft'
)

# ------------------------
# 🚀 Entraînement
# ------------------------
VC_light.fit(x_train, y_train)

# ------------------------
# 📊 Évaluation
# ------------------------
y_pred = VC_light.predict(x_test)
y_proba = VC_light.predict_proba(x_test)[:, 1]

print(f"Accuracy (train): {VC_light.score(x_train, y_train):.2f}")
print(f"Accuracy (test): {VC_light.score(x_test, y_test):.2f}")

from sklearn.metrics import classification_report, roc_auc_score
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# ------------------------
# 📈 Courbe ROC
# ------------------------
RocCurveDisplay.from_estimator(VC_light, x_test, y_test)
plt.show()

# ------------------------
# 📈 Courbe Précision-Recall
# ------------------------
from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_estimator(VC_light, x_test, y_test)
plt.show()