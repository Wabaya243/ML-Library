
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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

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

# =====================================================================
# 🤖 IMPORTATION DE NOMBREUX MODÈLES
# =====================================================================
# Idée : on veut tester beaucoup de modèles de base (niveau 0)
# Puis les combiner via Stacking (niveau 1)
# =====================================================================
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# =====================================================================
# 🧱 LISTE DES MODÈLES DE BASE (niveau 0)
# =====================================================================
# Chaque modèle apprend indépendamment → leurs prédictions seront combinées
# =====================================================================
estimators = []
estimators.append(('AdaBoostClassifier', AdaBoostClassifier(random_state=13)))
estimators.append(('Bagging Classifier', BaggingClassifier(random_state=13)))
estimators.append(('Bernoulli NB', BernoulliNB()))
estimators.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=13)))
estimators.append(('Dummy Classifier', DummyClassifier(random_state=13)))  # Baseline aléatoire
estimators.append(('Extra Tree Classifier', ExtraTreeClassifier(random_state=13)))
estimators.append(('Extra Trees Classifier', ExtraTreesClassifier(random_state=13)))
estimators.append(('Gaussian NB', GaussianNB()))
estimators.append(('Gaussian Process Classifier', GaussianProcessClassifier(random_state=13)))
estimators.append(('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=13)))
estimators.append(('Hist Gradient Boosting Classifier', HistGradientBoostingClassifier(random_state=13)))
estimators.append(('KNN', KNeighborsClassifier()))
estimators.append(('LogisticRegression', LogisticRegression(max_iter=1000, random_state=13)))
estimators.append(('Logistic Regression CV', LogisticRegressionCV(max_iter=1000, random_state=13)))
estimators.append(('MLPClassifier', MLPClassifier(max_iter=2000, random_state=13)))
estimators.append(('Nearest Centroid', NearestCentroid()))
estimators.append(('Passive Aggressive Classifier', PassiveAggressiveClassifier(random_state=13)))
estimators.append(('Perceptron', Perceptron(random_state=13)))
estimators.append(('RandomForest', RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=3, n_estimators=170, random_state=13)))
estimators.append(('Ridge Classifier', RidgeClassifier(random_state=13)))
estimators.append(('Ridge Classifier CV', RidgeClassifierCV()))
estimators.append(('SGDClassifier', SGDClassifier(random_state=13)))
estimators.append(('SVC', SVC(probability=True, random_state=13)))  # SVC doit activer probability=True pour stacking
estimators.append(('XGB', XGBClassifier(random_state=13)))
estimators.append(('CatBoost', CatBoostClassifier(logging_level='Silent', random_state=13)))

# =====================================================================
# 🏆 MODÈLE FINAL (niveau 1)
# =====================================================================
# On choisit XGBoost comme méta-modèle (souvent très performant)
# Il apprend à combiner les prédictions des modèles de base
# =====================================================================
XGB = XGBClassifier(random_state=13)

# =====================================================================
# 🏗️ CRÉATION DU STACKING CLASSIFIER
# =====================================================================
SC = StackingClassifier(estimators=estimators, final_estimator=XGB, cv=6)
SC.fit(x_train, y_train)

# =====================================================================
# 📊 ÉVALUATION DU MODÈLE
# =====================================================================
y_pred = SC.predict(x_test)

print(f"\nStacking classifier training Accuracy: {SC.score(x_train, y_train):0.2f}")
print(f"Stacking classifier test Accuracy: {SC.score(x_test, y_test):0.2f}")

# Calcul des métriques classiques
SC_Recall = recall_score(y_test, y_pred)
SC_Precision = precision_score(y_test, y_pred)
SC_f1 = f1_score(y_test, y_pred)
SC_accuracy = accuracy_score(y_test, y_pred)
SC_roc_auc = roc_auc_score(y_test, y_pred)

# Matrice de confusion → erreurs de classification
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =====================================================================
# 🔄 VALIDATION CROISÉE
# =====================================================================
# On évalue la robustesse avec cross-validation (scoring=recall car dataset déséquilibré)
# =====================================================================
from statistics import stdev
score = cross_val_score(SC, x_train, y_train, cv=5, scoring='recall', error_score="raise")
SC_cv_score = score.mean()
SC_cv_stdev = stdev(score)
print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', SC_cv_score)
print('Cross Validation Recall standard deviation: ', SC_cv_stdev)

# =====================================================================
# 📑 SYNTHÈSE DES SCORES DANS UN DATAFRAME
# =====================================================================
ndf = [(SC_Recall, SC_Precision, SC_f1, SC_accuracy, SC_roc_auc, SC_cv_score, SC_cv_stdev)]
SC_score = pd.DataFrame(data=ndf, columns=['Recall','Precision','F1 Score', 'Accuracy', 'ROC-AUC Score', 'Avg CV Recall', 'Standard Deviation of CV Recall'])
SC_score.insert(0, 'Model', 'Stacking Ensemble')
SC_score

# =====================================================================
# 📈 COURBE ROC
# =====================================================================
y_proba = SC.predict_proba(x_test)
from sklearn.metrics import roc_curve

def plot_auc_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0, 1], color='g')

plot_auc_roc_curve(y_test, y_proba[:, 1])

# =====================================================================
# 📈 COURBE PRÉCISION-RECALL
# =====================================================================
# Très utile pour datasets déséquilibrés (Churn = 1 est minoritaire)
# =====================================================================
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_estimator(
    SC, x_test, y_test, name="Average precision")
_ = display.ax_.set_title("Stacking Classifier")

# =====================================================================
# ✅ REMARQUE FINALE
# =====================================================================
# - Ce pipeline est complexe : beaucoup de modèles → temps d'entraînement élevé
# - Hyperparamètres non optimisés : tuning avec GridSearchCV ou Optuna recommandé
# - Le stacking combine les forces de chaque modèle (robustesse accrue)
# =====================================================================
