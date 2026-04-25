# ----------------------------- IMPORTS -----------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import pomegranate as pg
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
import warnings

# On ignore les warnings pour ne pas polluer la sortie
warnings.simplefilter('ignore')

# On fixe la seed pour reproductibilité des résultats
np.random.seed(1)

# ----------------------------- CHARGEMENT DES DONNÉES -----------------------------
# On charge le dataset de cancer du sein depuis sklearn
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data['target']

# ----------------------------- PRÉTRAITEMENT -----------------------------
'''
Nous allons mélanger le dataset car les données ont un ordre particulier dans la source originale,
ce qui peut biaiser les modèles si on ne shuffle pas.

Ensuite, nous réduisons la dimensionnalité en supprimant des features redondantes ou peu utiles pour
simplifier les modèles semi-supervisés et Naive Bayes. Les modèles métriques (Label Propagation, Label Spreading)
peuvent être sensibles à la dimensionnalité, et plus de features ne signifie pas forcément de meilleures performances.

Colonnes supprimées :
- features avec erreur (error)
- features worst (valeurs extrêmes)
- certaines mesures corrélées fortement à d'autres features
- mean area, mean perimeter, mean concave points (redondants)
'''
df = shuffle(df, random_state=42)  # mélange du dataframe

x = df.drop([
    'target', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry',
    'worst fractal dimension', 'mean area', 'mean perimeter', 'mean concave points'
], axis=1)

y = df['target']

# ----------------------------- VISUALISATION -----------------------------
'''
Puisque nous avons seulement 7 features maintenant, on peut utiliser un pairplot pour :
- visualiser la distribution de chaque feature sur la diagonale
- visualiser la corrélation entre features sur les autres graphes

Ceci nous aidera à comprendre les relations entre features et à détecter des corrélations
avant de normaliser ou d'utiliser des modèles indépendants.
'''
sns.pairplot(x)

# ----------------------------- SPLIT DES DONNÉES -----------------------------
'''
On crée trois ensembles :
1. Labeled train data (X_1, y_1) : 10% des données
2. Unlabeled train data (X_2, y_2) : 40% des données (étiquetées -1)
3. Test data (X_3, y_3) : 50% des données

On concatène labeled + unlabeled pour les modèles semi-supervisés,
en mettant -1 sur les données non étiquetées (LabelPropagation / LabelSpreading)
'''
X_1, X_2, X_3 = np.split(x, [int(.1*len(x)), int(.5*len(x))])
y_1, y_2, y_3 = np.split(y, [int(.1*len(y)), int(.5*len(y))])
y_1_2 = np.concatenate((y_1, y_2.apply(lambda x: -1)))
X_1_2 = np.concatenate((X_1, X_2))

# ----------------------------- TABLEAU DES RÉSULTATS -----------------------------
index = ['Algorithm', 'ROC AUC']
results = pd.DataFrame(columns=index)

# ----------------------------- LOGISTIC REGRESSION -----------------------------
'''
Nous entraînons un modèle de régression logistique sur les données étiquetées (X_1, y_1)
pour avoir un benchmark simple et robuste. Même si LR n'est pas semi-supervisé,
il servira de référence pour comparer nos modèles semi-supervisés.
'''
logreg = LogisticRegression(random_state=42, class_weight='balanced')
logreg.fit(X_1, y_1)

# Évaluation ROC AUC sur le test set
new_row = pd.DataFrame([['Logistic Regression', 
                         roc_auc_score(y_3, logreg.predict_proba(X_3)[:, 1])]], columns=index)
results = pd.concat([results, new_row], ignore_index=True)

# ----------------------------- LABEL PROPAGATION -----------------------------
'''
Fonction pour tester différents hyperparamètres pour Label Propagation (RBF ou KNN)
- kernel 'rbf' : gamma
- kernel 'knn' : n_neighbors
On mesure le ROC AUC sur le test set pour chaque paramètre
'''
def label_prop_test(kernel, params_list, X_train, X_test, y_train, y_test):
    roc_scores = []
    if kernel == 'rbf':
        for g in params_list:
            lp = LabelPropagation(kernel=kernel, gamma=g, max_iter=100000, tol=0.0001)
            lp.fit(X_train, y_train)
            roc_scores.append(roc_auc_score(y_test, lp.predict_proba(X_test)[:, 1]))
    elif kernel == 'knn':
        for n in params_list:
            lp = LabelPropagation(kernel=kernel, n_neighbors=n, max_iter=100000, tol=0.0001)
            lp.fit(X_train, y_train)
            roc_scores.append(roc_auc_score(y_test, lp.predict_proba(X_test)[:, 1]))

    # Visualisation
    plt.figure(figsize=(16, 8))
    plt.plot(params_list, roc_scores, marker='o')
    plt.title(f'Label Propagation ROC AUC avec kernel {kernel}')
    plt.xlabel("gamma" if kernel == "rbf" else "n_neighbors")
    plt.ylabel("ROC AUC")
    plt.show()

    best_param = params_list[np.argmax(roc_scores)]
    print(f'Meilleur paramètre trouvé : {best_param}')
    return best_param

# Test RBF et KNN
gammas = [9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]
best_gamma = label_prop_test('rbf', gammas, X_1_2, X_3, y_1_2, y_3)

ns = np.arange(50, 60)
best_n = label_prop_test('knn', ns, X_1_2, X_3, y_1_2, y_3)

# Modèles finaux Label Propagation
lp_rbf = LabelPropagation(kernel='rbf', gamma=best_gamma, max_iter=100000, tol=0.0001)
lp_rbf.fit(X_1_2, y_1_2)
results = pd.concat([results, pd.DataFrame([['Label Propagation RBF', roc_auc_score(y_3, lp_rbf.predict_proba(X_3)[:,1])]], columns=index)], ignore_index=True)

lp_knn = LabelPropagation(kernel='knn', n_neighbors=best_n, max_iter=100000, tol=0.0001)
lp_knn.fit(X_1_2, y_1_2)
results = pd.concat([results, pd.DataFrame([['Label Propagation KNN', roc_auc_score(y_3, lp_knn.predict_proba(X_3)[:,1])]], columns=index)], ignore_index=True)

# ----------------------------- LABEL SPREADING -----------------------------
'''
Label Spreading est similaire à Label Propagation mais utilise un alpha pour contrôler la force de diffusion.
On teste différentes valeurs d'alpha pour voir laquelle maximise le ROC AUC.
'''
def labels_spread_test(kernel, hyperparam, alphas, X_train, X_test, y_train, y_test):
    roc_scores = []
    for alpha in alphas:
        if kernel == 'rbf':
            ls = LabelSpreading(kernel=kernel, gamma=hyperparam, alpha=alpha, max_iter=1000, tol=0.001)
        elif kernel == 'knn':
            ls = LabelSpreading(kernel=kernel, n_neighbors=hyperparam, alpha=alpha, max_iter=1000, tol=0.001)
        else:
            raise ValueError("kernel doit être 'rbf' ou 'knn'")
        ls.fit(X_train, y_train)
        roc_scores.append(roc_auc_score(y_test, ls.predict_proba(X_test)[:,1]))

    plt.figure(figsize=(16,8))
    plt.plot(alphas, roc_scores, marker='o')
    plt.title(f'Label Spreading ROC AUC avec kernel {kernel}')
    plt.xlabel("alpha")
    plt.ylabel("ROC AUC")
    plt.show()

    best_alpha = alphas[np.argmax(roc_scores)]
    print(f'Meilleur alpha trouvé : {best_alpha}')
    return best_alpha

# ... On continue ici pour Label Spreading RBF et KNN

# ----------------------------- LABEL SPREADING FINAL -----------------------------
# On teste différentes valeurs d'alpha pour choisir la meilleure
alphas_rbf = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_alpha_rbf = labels_spread_test('rbf', 1e-5, alphas_rbf, X_1_2, X_3, y_1_2, y_3)

alphas_knn = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
best_alpha_knn = labels_spread_test('knn', 53, alphas_knn, X_1_2, X_3, y_1_2, y_3)

# Création des modèles Label Spreading finaux avec les meilleurs hyperparamètres
ls_rbf = LabelSpreading(kernel='rbf', gamma=9e-6, alpha=0.6, max_iter=1000, tol=0.001)
ls_rbf.fit(X_1_2, y_1_2)
results = pd.concat([results, pd.DataFrame([['Label Spreading RBF', roc_auc_score(y_3, ls_rbf.predict_proba(X_3)[:,1])]], columns=index)], ignore_index=True)

ls_knn = LabelSpreading(kernel='knn', n_neighbors=53, alpha=0.01, max_iter=1000, tol=0.001)
ls_knn.fit(X_1_2, y_1_2)
results = pd.concat([results, pd.DataFrame([['Label Spreading KNN', roc_auc_score(y_3, ls_knn.predict_proba(X_3)[:,1])]], columns=index)], ignore_index=True)

# ----------------------------- NAIVE BAYES (POMEGRANATE) -----------------------------
'''
On utilise pomegranate pour Naive Bayes. On peut :
- utiliser from_samples pour créer automatiquement les distributions des features par classe
- choisir la distribution pour chaque feature (Normal, Exponential, Poisson)
Attention : si les données ne correspondent pas aux distributions choisies, le modèle ne converge pas bien.
'''

# Conversion des données labellisées en float64 (requis par pomegranate)
X_nb = np.asarray(X_1_2, dtype=np.float64)
y_nb = y_1_2
X_test = np.asarray(X_3, dtype=np.float64)

# Exemple simple avec une distribution Exponentielle pour toutes les features
nb_exp = pg.NaiveBayes.from_samples(pg.ExponentialDistribution, X_nb, y_nb, verbose=False)
score_nb_exp = roc_auc_score(y_3, nb_exp.predict_proba(X_test)[:,1])
results = pd.concat([results, pd.DataFrame([['Naive Bayes Exponential', score_nb_exp]], columns=index)], ignore_index=True)

# Tentative avec différentes distributions pour chaque feature (on "triche" en connaissant la distribution)
distr_list = [
    pg.ExponentialDistribution, pg.PoissonDistribution, pg.NormalDistribution,
    pg.ExponentialDistribution, pg.ExponentialDistribution, pg.PoissonDistribution, pg.NormalDistribution
]
nb_icd = pg.NaiveBayes.from_samples(distr_list, X_nb, y_nb, verbose=False)
score_nb_icd = roc_auc_score(y_3, nb_icd.predict_proba(X_test)[:,1])
results = pd.concat([results, pd.DataFrame([['Naive Bayes ICD Prior', score_nb_icd]], columns=index)], ignore_index=True)

# ----------------------------- VISUALISATION DES COEFFICIENTS LOGREG -----------------------------
'''
Pour comprendre l'importance des features, on peut visualiser les coefficients du modèle de régression logistique.
- Feature avec coefficient positif : augmente la probabilité de la classe 1
- Feature avec coefficient négatif : diminue la probabilité de la classe 1
'''
logReg_coeff = pd.DataFrame({
    'feature_name': list(x.columns.values),
    'model_coefficient': logreg.coef_.transpose().flatten()
})
logReg_coeff = logReg_coeff.sort_values('model_coefficient', ascending=False)

plt.figure(figsize=(12,6))
fg = sns.barplot(x='feature_name', y='model_coefficient', data=logReg_coeff)
fg.set_xticklabels(rotation=35, labels=logReg_coeff['feature_name'])
plt.title("Importance des features selon la régression logistique")
plt.show()

# ----------------------------- TEST NAIVE BAYES SUR L'ENSEMBLE COMPLET -----------------------------
'''
On peut aussi entraîner Naive Bayes sur tout le dataset pour voir le comportement sur les données complètes.
Ici, on utilise ExponentialDistribution pour toutes les features.
'''
nb_test = pg.NaiveBayes.from_samples(pg.ExponentialDistribution, np.asarray(df.drop('target', axis=1), dtype=np.float64), y, verbose=False)
nb_test.predict_proba(np.asarray(df.drop('target', axis=1), dtype=np.float64))

# ----------------------------- AFFICHAGE FINAL -----------------------------
# Résumé de tous les résultats ROC AUC
print(results)












