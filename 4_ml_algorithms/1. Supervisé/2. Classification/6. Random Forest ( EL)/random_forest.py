import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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


###### FORET ALEATOIRE AVEC LE PARAMS PAR DEFAULT ###########

rfc = RandomForestClassifier(random_state=42)

#   entrainer les modes

rfc.fit(x_train, y_train)

#Predire les resultat

y_pred = rfc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print('La precision de la foret aleatoire est de : ', accuracy)


###### FORET ALEATOIRE AVEC LE PARAMS N etimator = 100 (nombre de arbre des decision) ###########

rfc_100 = RandomForestClassifier(n_estimators=100 ,random_state=42)

#   entrainer les modes

rfc_100.fit(x_train, y_train)

#Predire les resultat

y_pred_100 = rfc.predict(x_test)

accuracy_100 = accuracy_score(y_test, y_pred_100)
print('La precision de la foret aleatoire est de : ', accuracy_100)


###### FORET ALEATOIRE EN TROUVANT LES MEILLEURS FEATURES ###########

clf = RandomForestClassifier(n_estimators=100 ,random_state=42)

#   entrainer les model

clf.fit(x_train, y_train)

#Voire les score des features

feature_score = pd.Series(clf.feature_importances_, index=x_train.columns).sort_values(ascending=False)

print(feature_score)

##### Visualiser les score des entrées (feature)

sns.barplot(x=feature_score, y=feature_score.index)

plt.xlabel('Importance des entrées')
plt.ylabel('Entrées')

plt.title("Visualiser l'importance des features")

plt.show()

######## Lancé avec les features selectionné

x_new = dataset.drop(["class", 'doors'], axis=1)

y_new = dataset['class']


####### Separer en Train et Test set #############

x_train_new , x_test_new, y_train_new, y_test_new = train_test_split(x_new , y_new, test_size=0.33, random_state=42)


#### On encode les valeur categorielle avec k'encodeur ordinal 1,2,3,4

encoder_new = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])

x_train_new = encoder_new.fit_transform(x_train_new)
x_test_new = encoder_new.transform(x_test_new)

clf_new = RandomForestClassifier(random_state=42)

#   entrainer les modes

clf_new.fit(x_train_new, y_train_new)

#Predire les resultat

y_pred_new = clf_new.predict(x_test_new)

accuracy_new = accuracy_score(y_test, y_pred)
print('La precision de la foret aleatoire est de : ', accuracy_new)





