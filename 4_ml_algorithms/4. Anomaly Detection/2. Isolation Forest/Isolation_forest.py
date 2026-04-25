# ===============================
# 📌 Détection de fraude bancaire avec méthodes d'anomalie
# Dataset : creditcard.csv (Kaggle)
# ===============================

# Importation des bibliothèques nécessaires
import sklearn                        # Outils machine learning (scikit-learn)
import scipy                          # Outils scientifiques (utilitaire)
import matplotlib.pyplot as plt       # Librairie de visualisation
import seaborn as sns                 # Graphiques avancés
import numpy as np                    # Calcul numérique
from sklearn.metrics import classification_report, accuracy_score   # Évaluation des modèles
from sklearn.ensemble import IsolationForest                       # Modèle basé sur les arbres
from sklearn.neighbors import LocalOutlierFactor                   # Détection d'outliers locale
from sklearn.svm import OneClassSVM                                # SVM pour anomalies
import pandas as pd                   # Manipulation de données
from pylab import rcParams             # Pour configurer les tailles de figures

# Définir la taille des figures par défaut
rcParams['figure.figsize'] = 14, 8

# Fixer une graine aléatoire pour reproductibilité
RANDOM_SEED = 42

# Labels des classes (0 = normal, 1 = fraude)
LABELS = ["Normal", "Fraud"]

# Charger le dataset (chemin local)
data = pd.read_csv('Data/creditcard.csv')

# Compter combien de transactions normales et frauduleuses
count_classes = pd.value_counts(data['Class'], sort=True)

# Visualiser la répartition des classes
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Séparer les transactions frauduleuses et normales
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 0]

# Décrire les montants par classe (statistiques de base)
fraud.Amount.describe()
normal.Amount.describe()

# Histogrammes comparatifs des montants
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins=bins)     # Histogramme pour fraude
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins=bins)    # Histogramme pour normal
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))                  # Limiter l’axe x à 20 000
plt.yscale('log')                     # Échelle logarithmique pour comparer
plt.show()

# Prendre un échantillon de 10% pour réduire la taille (plus rapide)
data1 = data.sample(frac=0.1, random_state=1)
data1.shape

# Compter combien de fraudes et valides dans l’échantillon
Fraud = data1[data1['Class'] == 1]
Valid = data1[data1['Class'] == 0]

# Calculer le ratio fraude / normal (utile pour configurer les modèles)
outlier_fraction = len(Fraud) / float(len(Valid))

# Afficher quelques stats
print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud)))
print("Valid Cases : {}".format(len(Valid)))

# Matrice de corrélation des variables
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")

# Définir variables indépendantes (X) et dépendante (Y)
columns = data1.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]   # On enlève la colonne cible
target = "Class"

# Fixer état aléatoire
state = np.random.RandomState(42)

# Données d’entrée et labels
X = data1[columns]
Y = data1[target]

# Générer des "faux outliers" uniformes pour comparer
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

# Afficher dimensions de X et Y
print(X.shape)
print(Y.shape)

# ===============================
# 📌 Définition des modèles
# ===============================

classifiers = {
    "Isolation Forest": IsolationForest(
        n_estimators=100, max_samples=len(X), contamination=outlier_fraction,
        random_state=state, verbose=0),

    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20, algorithm='auto', leaf_size=30,
        metric='minkowski', p=2, contamination=outlier_fraction),

    "Support Vector Machine": OneClassSVM(
        kernel='rbf', degree=3, gamma=0.1, nu=0.05, max_iter=-1)
}

# Nombre total de fraudes
n_outliers = len(Fraud)

# ===============================
# 📌 Entraînement et évaluation
# ===============================
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Selon l’algorithme, l’entraînement est différent
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Transformer les prédictions pour correspondre aux labels (0 = normal, 1 = fraude)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # Compter les erreurs
    n_errors = (y_pred != Y).sum()

    # Afficher résultats du modèle
    print("{}: {}".format(clf_name, n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y, y_pred))
    print("Classification Report :")
    print(classification_report(Y, y_pred))
