import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

# Chargement du dataset des transactions
dataset = pd.read_csv('Data/creditcard.csv')

# ⚠️ Le nom de colonne "Class" crée des conflits en Python. 
# On le renomme donc en "Category" pour éviter les problèmes.
dataset = dataset.rename(columns={'Class': 'Category'})

# Séparation du dataset en deux sous-ensembles selon la classe
# 0 = transactions normales
# 1 = transactions frauduleuses (anormales)
nor_obs = dataset.loc[dataset.Category == 0]   # Transactions normales
ano_obs = dataset.loc[dataset.Category == 1]   # Transactions frauduleuses

# 📌 Création des ensembles d’apprentissage et de test :
# - train_features : ensemble d’entraînement
# - X_test : observations (features) pour le test
# - Y_test : labels pour le test

# On entraîne le One-Class SVM uniquement sur les transactions normales
# (ici les 200 000 premières lignes de transactions normales)
train_features = nor_obs.loc[0:210000, :].drop('Category', axis=1)

# Création des labels de test (partie normale + anomalies)
Y_1 = nor_obs.loc[210000:, 'Category']
Y_2 = ano_obs['Category']

# Création des observations de test
X_test_1 = nor_obs.loc[210000:, :].drop('Category', axis=1)
X_test_2 = ano_obs.drop('Category', axis=1)
X_test = pd.concat([X_test_1, X_test_2], ignore_index=True)

# 🔧 Définition des hyperparamètres du One-Class SVM
oneclass = OneClassSVM(nu=0.95, gamma=0.001, kernel='linear')

# Remarque : j’ai testé plusieurs combinaisons (kernel=linear, rbf, poly ; 
# gamma=0.001 ou 0.0001 ; nu=0.25, 0.5, 0.75, 0.95).
# Cette combinaison a donné les meilleurs résultats dans mon cas.

# Construction des labels de test complets
Y_test = pd.concat([Y_1, Y_2], ignore_index=True)

# 📝 Y_test sera utilisé pour évaluer le modèle

# 🚀 Entraînement du modèle sur les features normales
# Attention : cette étape est très coûteuse en temps de calcul.
# Exemple : sur mon PC portable, l’entraînement sur 200 000 obs. a pris > 1h.
# Avec un kernel RBF, c’est encore plus long.
oneclass.fit(train_features)

# 🔎 Test du modèle sur l’ensemble de test
fraud_pred = oneclass.predict(X_test)

# Vérification du nombre d’outliers détectés
unique, counts = np.unique(fraud_pred, return_counts=True)
print(np.asarray((unique, counts)).T)

# Conversion des labels et prédictions en DataFrame pour faciliter la comparaison
Y_test = Y_test.to_frame().reset_index(drop=True)
fraud_pred = pd.DataFrame(fraud_pred, columns=['prediction'])

# ✅ Évaluation des performances du modèle
TP = FN = FP = TN = 0
for j in range(len(Y_test)):
    if Y_test['Category'][j] == 0 and fraud_pred['prediction'][j] == 1:
        TP += 1
    elif Y_test['Category'][j] == 0 and fraud_pred['prediction'][j] == -1:
        FN += 1
    elif Y_test['Category'][j] == 1 and fraud_pred['prediction'][j] == 1:
        FP += 1
    else:
        TN += 1

print("TP:", TP, "FN:", FN, "FP:", FP, "TN:", TN)

# 📊 Calcul des métriques de performance
accuracy = (TP + TN) / (TP + FN + FP + TN)
print("Accuracy:", accuracy)

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
print("Sensitivity (Recall):", sensitivity)

specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
print("Specificity:", specificity)

"""
📌 Conclusion :
Le One-Class SVM donne de très bons résultats sur ce dataset,
avec environ 90 % de détection des fraudes et peu de faux positifs.
C’est une bonne base pour affiner le modèle (ex. tuning des hyperparamètres).
Cependant, l’entraînement est coûteux en temps de calcul (surtout avec RBF).
Par rapport à Isolation Forest (testé précédemment), le One-Class SVM 
semble mieux fonctionner ici.
"""


"""
Le point important :
👉 OneClassSVM.predict retourne 1 pour inlier (normal) et -1 pour outlier (fraude),
alors que dans ton dataset 0 = normal et 1 = fraude.
Il faut donc recaler les prédictions pour que ça corresponde à tes labels.
"""


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ⚠️ Recalage des prédictions : -1 → 1 (fraude), 1 → 0 (normal)
fraud_pred_mapped = np.where(fraud_pred['prediction'] == -1, 1, 0)

# Matrice de confusion
cm = confusion_matrix(Y_test['Category'], fraud_pred_mapped)
print("Matrice de confusion :\n", cm)

# Rapport complet (precision, recall, f1-score)
print("\nRapport de classification :")
print(classification_report(Y_test['Category'], fraud_pred_mapped, target_names=["Normal", "Fraude"]))

# Accuracy
acc = accuracy_score(Y_test['Category'], fraud_pred_mapped)
print("\nAccuracy :", acc)
















