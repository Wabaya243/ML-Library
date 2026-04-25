# Importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

col_names = ["company", 'job', 'degree', 'salary_more_then_100K']
data = pd.read_csv('salaries.csv', header=None, names=col_names)
print(data)

from sklearn import preprocessing

# Créer un encodeur différent pour chaque colonne catégorielle
company_encoder = preprocessing.LabelEncoder()
job_encoder = preprocessing.LabelEncoder()
degree_encoder = preprocessing.LabelEncoder()

# Encoder les colonnes séparément
data['company'] = company_encoder.fit_transform(data['company'])
data['job'] = job_encoder.fit_transform(data['job'])
data['degree'] = degree_encoder.fit_transform(data['degree'])

# Afficher les premières lignes
print(data.head(9))

features_cols = ['company', 'job', 'degree']
x = data[features_cols]
y = data['salary_more_then_100K']
#x = data.values[1:, :3]
#y = data.values[1:, 3]   #1:,3 signifi qu'on utilise pas les entetes

print (x)
print(y)

 # separation en training et Test Set

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=100)


#Creation d'un arbre de decision qui utilise l'entropie
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)

#Entrainer l'arbre de decision 
clf_entropy = clf_entropy.fit(x_train, y_train)

#predire les resultats
y_pred = clf_entropy.predict(x_test)

print('accuracy:', metrics.accuracy_score(y_test, y_pred))


from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(clf_entropy, filled=True, feature_names=x_train.columns, class_names=True)
plt.show()
