import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns



#importer les dataset
dataset = pd.read_csv('KNNAlgorithmDataset.csv')
x = dataset.drop("diagnosis", axis=1).values
y = dataset['diagnosis'].values

imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

#Separe Train et Test Set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

#standarisation et regularisation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#la methode du coude 
k_values = range(1, 21)  # on teste k de 1 à 20
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    # cross-validation 5-fold
    cv_scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    scores.append(cv_scores.mean())

k_values = range(1, 21)  # on teste k de 1 à 20
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    # cross-validation 5-fold
    cv_scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    scores.append(cv_scores.mean())

# trouver le k optimal
best_k = k_values[np.argmax(scores)]
print(f"Meilleur k = {best_k} avec une accuracy moyenne de {max(scores):.4f}")

# tracer les résultats
plt.plot(k_values, scores, marker='o')
plt.xlabel("Valeur de k")
plt.ylabel("Accuracy moyenne (CV)")
plt.title("Choix du meilleur k pour KNN")
plt.show()


#creation du model
knn_model = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=2)
#entrainer les le classifier
knn_model.fit(x_train, y_train)

#Faire la prediction
y_pred = knn_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"La precision : ", accuracy)


#matrix de confusion
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(pd.DataFrame(cm), annot=True)

report = classification_report(y_test, y_pred)
print(report)









