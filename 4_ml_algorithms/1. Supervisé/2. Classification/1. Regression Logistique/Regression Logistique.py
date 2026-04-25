import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


#importer les dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')

dataset.isnull().sum()

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1]


#Train et Test Set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#features scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#creation du Model
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


#prediction
y_pred = classifier.predict(x_test)
print(y_pred)

#matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(pd.DataFrame(cm), annot=True)


#La precision du model 
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



# Visualisation des résultats
from matplotlib.colors import ListedColormap


### Partie test Fin ####

# Visualisation des résultats sur l'ensemble de test
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Régression Logistique (Ensemble de test)')
plt.xlabel('Âge')
plt.ylabel('Salaire estimé')
plt.legend()
plt.show()



# Calculer les métriques de performance de base
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Accuracy (Précision globale)
accuracy = accuracy_score(y_test, y_pred)

# Précision (pour la classe positive)
precision = precision_score(y_test, y_pred)

# Sensibilité/Recall (pour la classe positive)
sensitivity = recall_score(y_test, y_pred)

# Calculer la spécificité (pour la classe négative)
# Spécificité = TN / (TN + FP) = Recall pour la classe 0
specificity = recall_score(y_test, y_pred, pos_label=0)

# Taux d'erreur
error_rate = 1 - accuracy

# Afficher les métriques
print("=== MÉTRIQUES DE PERFORMANCE ===")
print(f"Accuracy (Précision globale): {accuracy:.3f}")
print(f"Précision (pour classe positive): {precision:.3f}")
print(f"Sensibilité/Recall (pour classe positive): {sensitivity:.3f}")
print(f"Spécificité (pour classe négative): {specificity:.3f}")
print(f"Taux d'erreur: {error_rate:.3f}")

# Afficher la matrice de confusion pour référence
print("\n=== MATRICE DE CONFUSION ===")
print("Format: [TN FP]")
print("        [FN TP]")
print(cm)
