import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import category_encoders as ce
from sklearn import tree

#Importer les dataset
dataset = pd.read_csv("car_evaluation.csv")

#explorer les dataset
dataset.shape

#Voire le Top 5 de variable
dataset.head()

#Renomer les colones du dataset
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
dataset.columns = col_names

print(col_names)

dataset.info()

#Maintenant, je vais vérifier les nombres de fréquences des variables catégorielles.

for col in col_names:
    
    print(dataset[col].value_counts())   

#on explore la variables class

print(dataset['class'].value_counts())

#verifier si il y a des valeurs manquantes

dataset.isnull().sum()

#Declarer les valeurs indepandates et depandants

x = dataset.drop(['class'], axis=1)

y = dataset['class']


####### Separer en Train et Test set #############

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


######### Engeineering de variables ##############

print(x_train.dtypes)

#### On encode les valeur categorielle avec k'encodeur ordinal 1,2,3,4

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)


####### ARBRE DE DECISIONS AVEC CRITERE GINI INDEX  ######

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

#entrainer le model
clf_gini.fit(x_train, y_train)
clf_entropy.fit(x_train, y_train)

#prediciton 

y_pred_gini = clf_gini.predict(x_test)
y_pred_entropy  = clf_entropy.predict(x_test)

#### Calculer la precsision du Modele

accuracu_gini = accuracy_score(y_test, y_pred_gini)
print("la precision du model avec gini index : ", accuracu_gini)

accuracu_entropy = accuracy_score(y_test, y_pred_entropy)
print("la precision du model avec entropy : ", accuracu_entropy)


################## Verifier cas d'overfitting ou Underfitting ##########

y_pred_train_gini = clf_gini.predict(x_train)
accuracy_train_gini = accuracy_score(y_train, y_pred_train_gini)

print("la precision du model avec gini index : ", accuracy_train_gini )
print("la precision du model avec Test gini index : ", accuracu_gini)

#### 78% et 80% les deux valeurs sont presque pareils pas des risqué d'overfitting ou underfitting


######### On visualise les resultats ################

plt.figure(figsize=(12, 8))

tree.plot_tree(clf_gini.fit(x_train, y_train))
tree.plot_tree(clf_entropy.fit(x_train, y_train))


### Matrice des confusion

cm_gini = confusion_matrix(y_test, y_pred_gini)
print('matrice de confusion gini : ', cm_gini)

class_repot = classification_report(y_test, y_pred_gini)
print('raaport des classification gini : ', class_repot)