# Co-Training sur le dataset Breast Cancer (scikit-learn)
# Code entièrement commenté en FRANÇAIS pour expliquer le *pourquoi* de chaque étape,
# en particulier la boucle de co-training.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# 0) Réglages / reproducibilité
# ---------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------
# 1) Chargement du dataset
# ---------------------------
# Pourquoi : utiliser un dataset médian (réellement utilisé en ML) pour voir
# l'effet du co-training sur des features plus nombreuses (30 features).
data = load_breast_cancer()
X_full, y_full = data.data, data.target

# ---------------------------
# 2) Création de deux 'vues'
# ---------------------------
# Choix pédagogique : on divise les 30 features en deux groupes de 15.
# Raison : Co-Training nécessite des vues (sous-ensembles de features) qui
# portent de l'information complémentaire.
X_v1_full = X_full[:, :15]   # Vue 1 = premières 15 features
X_v2_full = X_full[:, 15:]   # Vue 2 = dernières 15 features

# ---------------------------
# 3) Split train / test
# ---------------------------
# On garde une partie en test pour évaluer la performance finale hors entraînement.
X_v1_train, X_v1_test, X_v2_train, X_v2_test, y_train, y_test = train_test_split(
    X_v1_full, X_v2_full, y_full, test_size=0.3, random_state=RANDOM_STATE, stratify=y_full
)

# ---------------------------
# 4) Créer un petit set étiqueté + beaucoup de non-étiquetés
# ---------------------------
# Pourquoi : simuler un vrai scénario semi-supervisé où les étiquettes sont rares.
labeled_fraction = 0.2  # 20% des données d'entraînement sont étiquetées
X_v1_labeled, X_v1_unlabeled, X_v2_labeled, X_v2_unlabeled, y_labeled_init, y_unlabeled = train_test_split(
    X_v1_train, X_v2_train, y_train, test_size=(1 - labeled_fraction), random_state=RANDOM_STATE, stratify=y_train
)

# On garde deux copies des labels étiquetés : une par vue.
# Pourquoi : au fil du co-training, chaque modèle recevra des labels 'pseudo' différents
# provenant des prédictions du modèle partenaire. Il faut donc tenir les labels synchronisés
# avec les X de chaque vue.
y_labeled_v1 = y_labeled_init.copy()
y_labeled_v2 = y_labeled_init.copy()

# ---------------------------
# 5) Initialiser les modèles
# ---------------------------
# Choix : un modèle linéaire (Logistic Regression) + un modèle d'ensemble (RandomForest).
# Pourquoi : modèles différents favorisent la diversité des erreurs — utile en co-training.
model_v1 = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
model_v2 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

# ---------------------------
# 6) Boucle de Co-Training (avec explications détaillées)
# ---------------------------
iterations = 10          # nombre d'itérations co-training
samples_to_add = 20     # nombre maximal d'exemples ajoutés par modèle à chaque itération

# Pour suivre l'évolution des accuracies de validation (test) au fil des itérations
history = {
    'iter': [],
    'acc_v1_test': [],
    'acc_v2_test': [],
    'labeled_size': []
}

for it in range(1, iterations + 1):
    # ---------------------------
    # 6.1 Entraîner chaque modèle sur ses données étiquetées actuelles
    # ---------------------------
    # Pourquoi : à chaque itération, les ensembles étiquetés ont changé (on a ajouté des exemples),
    # donc on réentraîne pour que chaque modèle profite des nouveaux exemples.
    model_v1.fit(X_v1_labeled, y_labeled_v1)
    model_v2.fit(X_v2_labeled, y_labeled_v2)

    # ---------------------------
    # 6.2 Prédire les probabilités sur le pool non étiqueté
    # ---------------------------
    # Pourquoi probabilités ? On veut estimer la "confiance" de chaque prédiction.
    # On choisira ensuite les exemples avec la plus haute confiance pour les pseudo-étiqueter.
    probs_v1 = model_v1.predict_proba(X_v1_unlabeled)
    probs_v2 = model_v2.predict_proba(X_v2_unlabeled)

    # ---------------------------
    # 6.3 Calculer un score de confiance pour chaque exemple non-étiqueté
    # ---------------------------
    # Ici on prend la probabilité maximale (softmax argmax) : plus elle est proche de 1,
    # plus le modèle est confiant dans sa prédiction.
    conf_v1 = np.max(probs_v1, axis=1)
    conf_v2 = np.max(probs_v2, axis=1)

    # ---------------------------
    # 6.4 Sélectionner les indices des exemples les plus confiants
    # ---------------------------
    # Important : le nombre d'exemples à ajouter doit être <= taille du pool non-étiqueté.
    n_unlabeled = len(X_v1_unlabeled)
    if n_unlabeled == 0:
        print("Aucun exemple non-étiqueté restant — arrêt du co-training.")
        break

    k = min(samples_to_add, n_unlabeled)  # ajuster si le pool est petit

    # np.argsort(-conf) donne les indices triés par confiance décroissante
    idx_v1_top = np.argsort(-conf_v1)[:k]
    idx_v2_top = np.argsort(-conf_v2)[:k]

    # ---------------------------
    # 6.5 Pseudo-étiqueter et ajouter aux ensembles étiquetés des VUES partenaires
    # ---------------------------
    # Principe : les exemples que le modèle A (v1) juge très confiants sont ajoutés
    # à l'ensemble d'entraînement du modèle B (v2), avec les labels prédits par A.
    # Pourquoi ? parce que si A est confiant sur ces exemples, il est probable
    # que la pseudo-étiquette soit correcte et utile pour B.

    # a) exemples choisis par v1 -> ajoutés à v2 (labels prédits par v1 sur la vue v1)
    X_v2_labeled = np.vstack([X_v2_labeled, X_v2_unlabeled[idx_v1_top]])
    pseudo_labels_for_v2 = model_v1.predict(X_v1_unlabeled[idx_v1_top])
    y_labeled_v2 = np.hstack([y_labeled_v2, pseudo_labels_for_v2])

    # b) exemples choisis par v2 -> ajoutés à v1 (labels prédits par v2 sur la vue v2)
    X_v1_labeled = np.vstack([X_v1_labeled, X_v1_unlabeled[idx_v2_top]])
    pseudo_labels_for_v1 = model_v2.predict(X_v2_unlabeled[idx_v2_top])
    y_labeled_v1 = np.hstack([y_labeled_v1, pseudo_labels_for_v1])

    # ---------------------------
    # 6.6 Retirer ces exemples du pool non-étiqueté
    # ---------------------------
    # On retire l'union des indices choisis par v1 et v2. On utilise np.unique pour
    # éviter de supprimer deux fois le même indice si les modèles ont sélectionné
    # certains exemples identiques.
    to_remove = np.unique(np.concatenate([idx_v1_top, idx_v2_top]))

    # Création d'un masque de la bonne taille (taille courante du pool non-étiqueté)
    mask = np.ones(n_unlabeled, dtype=bool)
    mask[to_remove] = False

    # Appliquer le filtre aux deux vues non-étiquetées et aux labels non-étiquetés
    X_v1_unlabeled = X_v1_unlabeled[mask]
    X_v2_unlabeled = X_v2_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]

    # ---------------------------
    # 6.7 Suivi / diagnostic : mesurer l'évolution sur le set de test
    # ---------------------------
    # On évalue chaque modèle sur le set de test (séparé au début) pour voir si
    # le co-training améliore réellement la generalisation.
    acc_v1_test = accuracy_score(y_test, model_v1.predict(X_v1_test))
    acc_v2_test = accuracy_score(y_test, model_v2.predict(X_v2_test))

    history['iter'].append(it)
    history['acc_v1_test'].append(acc_v1_test)
    history['acc_v2_test'].append(acc_v2_test)
    history['labeled_size'].append(len(y_labeled_v1))  # tailles corrélées

    print(f"It {it:02d}: labeled_size={len(y_labeled_v1)} | acc_v1_test={acc_v1_test:.3f} | acc_v2_test={acc_v2_test:.3f}")

# ---------------------------
# 7) Évaluation finale après co-training
# ---------------------------
final_acc_v1 = accuracy_score(y_test, model_v1.predict(X_v1_test))
final_acc_v2 = accuracy_score(y_test, model_v2.predict(X_v2_test))

print("\n=== Résultats finaux ===")
print(f"Accuracy Vue 1 (Logistic Regression) : {final_acc_v1:.3f}")
print(f"Accuracy Vue 2 (Random Forest)       : {final_acc_v2:.3f}")

# ---------------------------
# Remarques pédagogiques (pourquoi certaines décisions ont été prises):
# - On utilise predict_proba + max probabilité comme mesure de confiance simple et
#   intuitive. D'autres mesures existent (marge entre top2 classes, entropie, etc.).
# - On garde des labels séparés pour chaque vue (y_labeled_v1 / y_labeled_v2) parce
#   que les pseudo-étiquettes ajoutées viennent du modèle partenaire et ne sont
#   pas forcément identiques d'une vue à l'autre.
# - On retire l'union des indices choisis pour éviter de réutiliser les mêmes exemples.
# - Le choix des modèles différents (linéaire vs non-linéaire) favorise la diversité
#   des erreurs — un comportement utile dans la plupart des scénarios Co-Training.
# - Attention : le co-training peut propager des erreurs si on ajoute des pseudo-labels
#   très incorrects avec trop de confiance. Dans des applications réelles, on peut
#   appliquer des heuristiques additionnelles : seuil de confiance absolu,
#   calibration des probabilités, vérification par un oracle humain, etc.

# Si tu veux, je peux :
# - Ajouter une visualisation (graphiques) de l'évolution des accuracies par itération.
# - Transformer le code en notebook Jupyter (avec cellules et affichages intermédiaires).
# - Adapter le nombre d'exemples ajoutés, la fraction étiquetée, ou les modèles.


# ---------------------------
# 8) Test sur de nouvelles données externes
# ---------------------------
# Pour simuler un "jeu externe", on prélève un petit échantillon aléatoire de X_full
# qui n'a PAS été utilisé dans train/test (par ex. 5 nouveaux points). Dans un cas réel,
# on utiliserait un vrai dataset externe.
new_samples = X_full[:5] # ici juste 5 premières instances comme simulation
new_samples_v1 = new_samples[:, :15]
new_samples_v2 = new_samples[:, 15:]


pred_new_v1 = model_v1.predict(new_samples_v1)
pred_new_v2 = model_v2.predict(new_samples_v2)


print("\n=== Prédictions sur nouvelles données (simulation) ===")
for i in range(len(new_samples)):
    print(f"Sample {i+1}: pred_v1={pred_new_v1[i]} | pred_v2={pred_new_v2[i]}")




