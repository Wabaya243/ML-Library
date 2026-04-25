# Importation des bibliothèques nécessaires

# Génération d'un dataset synthétique
from sklearn.datasets import make_classification
# Traitement et manipulation des données
import pandas as pd
import numpy as np
from collections import Counter
# Visualisation
import matplotlib.pyplot as plt
# Modèle et évaluation des performances
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Création d'un dataset déséquilibré (majorité vs minorité)
X, y = make_classification(n_samples=100000,    # nombre total d'échantillons
                           n_features=2,        # nombre de variables explicatives
                           n_informative=2,     # variables réellement utiles
                           n_redundant=0,       # pas de variables redondantes
                           n_repeated=0,        # pas de variables répétées
                           n_classes=2,         # deux classes (normal / anomalie)
                           n_clusters_per_class=1,
                           weights=[0.995, 0.005], # 99.5% normaux, 0.5% anomalies
                           class_sep=0.5,       # séparation entre les classes (modérée)
                           random_state=0)      # reproductibilité

# Conversion du dataset numpy en DataFrame pour faciliter l’analyse
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

# Vérification de la distribution des classes (proportion)
# utile pour confirmer le déséquilibre avant de modéliser
df['target'].value_counts(normalize=True)


## ---- Entraînement du modèle ---- ##

# Découpage du dataset en training set (80%) et test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Affichage des tailles des datasets (contrôle rapide)
print('The number of records in the training dataset is', X_train.shape[0])
print('The number of records in the test dataset is', X_test.shape[0])
# Affichage du nombre d'exemples par classe dans l'ensemble d'entraînement
print(f"The training dataset has {sorted(Counter(y_train).items())[0][1]} records for the majority class and {sorted(Counter(y_train).items())[1][1]} records for the minority class.")


# Entraînement du modèle One-Class SVM (spécialisé pour la détection d’anomalies)
# One-Class SVM apprend la "frontière" des données normales ; tout ce qui en dehors est considéré anomalie.
one_class_svm = OneClassSVM(nu=0.01,  # nu ≈ proportion maximale d'outliers attendus (≈1%)
                            kernel='rbf', # noyau RBF (gaussien) pour capturer des frontières non-linéaires
                            gamma='auto'  # paramètre gamma géré automatiquement
                           ).fit(X_train)


# Prédictions du modèle sur le jeu de test
prediction = one_class_svm.predict(X_test)
# One-Class SVM renvoie -1 pour anomalie et +1 pour "normal"
# On convertit en 1 = anomalie, 0 = normal pour correspondre au y_test binaire
prediction = [1 if i == -1 else 0 for i in prediction]


# Rapport de classification (precision, recall, f1-score) pour évaluer le modèle
# Attention : avec un fort déséquilibre, regarder le recall (sensibilité) de la classe minoritaire.
print(classification_report(y_test, prediction))


# Récupération des scores (densité / score_samples) : plus bas = plus suspect
score = one_class_svm.score_samples(X_test)

# Définition d'un seuil personnalisé : ici on prend les 2% des pires scores comme anomalies
score_threshold = np.percentile(score, 2)
print(f'The customized score threshold for 2% of outliers is {score_threshold:.2f}')


# Nouvelle prédiction en utilisant le seuil personnalisé (2% des pires)
# On marque comme anomalie tout point dont le score est inférieur au seuil
customized_prediction = [1 if i < score_threshold else 0 for i in score]

# Rapport de classification avec le seuil personnalisé
print(classification_report(y_test, customized_prediction))


#### ---- Visualisation ---- ####

# Rassembler les données test et les prédictions pour afficher côte à côte
df_test = pd.DataFrame(X_test, columns=['feature1', 'feature2'])
df_test['y_test'] = y_test
df_test['one_class_svm_prediction'] = prediction
df_test['one_class_svm_prediction_cutomized'] = customized_prediction

# Création de 3 graphiques côte à côte pour comparer
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(20, 6))

# 1) Vérité terrain (ground truth)
ax0.set_title('Original')
ax0.scatter(df_test['feature1'], df_test['feature2'], c=df_test['y_test'], cmap='rainbow')

# 2) Prédictions One-Class SVM (format par défaut du modèle)
ax1.set_title('One-Class SVM Predictions')
ax1.scatter(df_test['feature1'], df_test['feature2'], c=df_test['one_class_svm_prediction'], cmap='rainbow')

# 3) Prédictions One-Class SVM avec seuil personnalisé (2%)
ax2.set_title('One-Class SVM Predictions With Customized Threshold')
ax2.scatter(df_test['feature1'], df_test['feature2'], c=df_test['one_class_svm_prediction_cutomized'], cmap='rainbow')

plt.show()


# ---- Conclusion (ajoutée ici dans le code sous forme de commentaire, telle que fournie) ----
# In conclusion, the implementation of One-Class Support Vector Machine (SVM) for anomaly detection offers valuable insights into handling imbalanced datasets and identifying rare occurrences within a given context. Through the steps of dataset creation, model training, and visualization, several key takeaways emerge.
#
# The customization of the anomaly detection threshold in "Step 2" further underscores the adaptability of the One-Class SVM. By tailoring the threshold to a specific percentage of outliers, practitioners can fine-tune the model to meet the desired level of sensitivity to anomalies. This flexibility allows the One-Class SVM to be customized for different operational contexts, balancing false positives and false negatives based on the severity of consequences associated with missed anomalies.
#
# In "Step 3," the visualization of model predictions provides a tangible representation of the One-Class SVM's performance. The comparison of ground truth anomalies, model predictions, and the impact of a customized threshold enhances interpretability and aids in decision-making.
#
# Overall, One-Class SVM proves to be a robust tool for anomaly detection in scenarios where labeled anomalous data is scarce. Its adaptability to different thresholds and reliable performance on imbalanced datasets make it a valuable asset in fields such as cybersecurity, fraud detection, and quality control. However, practitioners should be mindful of the trade-offs between precision and recall, tailoring the model to the specific needs of their application.
#
# The identification of anomalies can also prove invaluable to draw insights on what is causing the anomalies based on their data and comparing them to regular data points.
#
# ---- Fin de la conclusion ----
