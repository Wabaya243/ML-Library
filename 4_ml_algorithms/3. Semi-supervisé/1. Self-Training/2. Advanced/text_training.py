# Import des bibliothèques nécessaires
import numpy as np  # pour les opérations numériques et manipulations de tableaux
import pandas as pd  # pour la lecture et manipulation des datasets tabulaires
from sklearn.feature_extraction.text import CountVectorizer  # transforme le texte en vecteurs de compte de mots
from sklearn.feature_extraction.text import TfidfTransformer  # transforme les comptes en TF-IDF
from sklearn.preprocessing import FunctionTransformer  # permet de créer des transformations personnalisées (non utilisé ici)
from sklearn.linear_model import SGDClassifier  # classifieur linéaire utilisant la descente de gradient stochastique
from sklearn.model_selection import train_test_split  # pour diviser les données en train/test
from sklearn.pipeline import Pipeline  # pour enchaîner plusieurs étapes (vecteurs → TF-IDF → classifier)
from sklearn.semi_supervised import SelfTrainingClassifier  # pour le semi-supervisé via auto-labeling
from sklearn.semi_supervised import LabelSpreading  # alternative semi-supervisée basée sur graphe
from sklearn.metrics import f1_score  # pour mesurer la performance (équilibre précision/rappel)
from sklearn.metrics import classification_report  # rapport détaillé des performances par classe

# Lecture du dataset CSV
data = pd.read_csv("Data/dataset.csv")  # le dataset doit contenir les colonnes Text et Sentiment

# Vérification des catégories uniques
data.Sentiment.unique()  # affiche toutes les valeurs uniques de la colonne Sentiment

# Nettoyage de certaines valeurs de sentiment spécifiques
data.Sentiment[data.Sentiment == ' it adds a lot of great options by opening doors to new places and experiences.'] = 'positive'
data.Sentiment[data.Sentiment == '-'] = 'neutral'

# Affichage du nombre de documents et de catégories
print("%d documents" % len(data.Text))
print("%d categories" % len(data.Sentiment))

# -----------------------------------------------
# 🔹 Paramètres pour le pipeline
# -----------------------------------------------

# Paramètres du classifieur SGD
sdg_params = dict(alpha=1e-5, penalty='l2', loss='log_loss')
# alpha=1e-5 → régularisation pour éviter le surapprentissage
# penalty='l2' → régularisation L2 classique
# loss='log' → perte logistique pour classification multi-classe

# Paramètres du vectorizer
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
# ngram_range=(1,2) → unigrammes et bigrammes
# min_df=5 → ignorer les mots apparaissant moins de 5 fois
# max_df=0.8 → ignorer les mots trop fréquents (stop words)

# -----------------------------------------------
# 🔹 Pipeline supervisé classique
# -----------------------------------------------
pipeline = Pipeline([
    ('vect', CountVectorizer(**vectorizer_params)),  # transformation texte → compte de mots
    ('tfidf', TfidfTransformer()),                   # compte → TF-IDF
    ('clf', SGDClassifier(**sdg_params)),            # classification avec SGD
])

# -----------------------------------------------
# 🔹 Pipeline Self-Training (semi-supervisé)
# -----------------------------------------------
st_pipeline = Pipeline([
    ('vect', CountVectorizer(**vectorizer_params)),  # même transformation texte → vecteurs
    ('tfidf', TfidfTransformer()),                   # même TF-IDF
    ('clf', SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)),  
    # SelfTrainingClassifier : le modèle se "réétiquette" automatiquement les données non étiquetées
    # verbose=True : affiche le détail des ajouts de labels
])

# -----------------------------------------------
# 🔹 Fonction pour entraîner et évaluer un classifieur
# -----------------------------------------------
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    # Affiche le nombre d'échantillons
    print("Number of training samples:", len(X_train))
    # Compte le nombre d'échantillons non étiquetés (-1) dans le train
    print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1))
    
    # Entraînement du classifieur
    clf.fit(X_train, y_train)
    
    # Prédiction sur le test set
    y_pred = clf.predict(X_test)
    
    # Affichage du F1 micro-averaged pour évaluer la performance globale
    print("Micro-averaged F1 score on test set: %0.3f" % f1_score(y_test, y_pred, average='micro'))
    print("-" * 10)
    print()

# -----------------------------------------------
# 🔹 Script principal
# -----------------------------------------------
if __name__ == "__main__":
    # Séparation des features et labels
    X, y = data.Text, data.Sentiment
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # ---- Test supervisé sur 100% des données
    print("Supervised SGDClassifier on 100% of the data:")
    eval_and_print_metrics(pipeline, X_train, y_train, X_test, y_test)
    
    print(classification_report(y_test, pipeline.predict(X_test)))

    # ---- Sélection aléatoire de 20% du train pour simuler un semi-supervisé
    y_mask = np.random.rand(len(y_train)) < 0.2

    # X_20 et y_20 → sous-ensemble du train correspondant au masque
    X_20, y_20 = map(list, zip(*((x, y) for x, y, m in zip(X_train, y_train, y_mask) if m)))
    
    print("Supervised SGDClassifier on 20% of the training data:")
    eval_and_print_metrics(pipeline, X_20, y_20, X_test, y_test)
    
    print(classification_report(y_test, pipeline.predict(X_test)))

    # ---- Self-Training : 20% étiquetés, le reste -1
    y_train[~y_mask] = -1  # tous les autres sont non étiquetés
    
    print("SelfTrainingClassifier on 20% of the training data (rest is unlabeled):")
    eval_and_print_metrics(st_pipeline, X_train, y_train, X_test, y_test)
    
    print(classification_report(y_test, st_pipeline.predict(X_test)))
