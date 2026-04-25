import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Séparer en jeu d'entraînement et jeu de test (30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Séparer le jeu d'entraînement en petite partie étiquetée et partie non étiquetée
labeled_size = int(0.3 * len(X_train))  # 30% étiqueté
X_labeled, X_unlabeled = X_train[:labeled_size], X_train[labeled_size:]
y_labeled, y_unlabeled = y_train[:labeled_size], y_train[labeled_size:]

# Vues (features différentes)
X_labeled_v1, X_unlabeled_v1, X_test_v1 = X_labeled[:, :2], X_unlabeled[:, :2], X_test[:, :2]
X_labeled_v2, X_unlabeled_v2, X_test_v2 = X_labeled[:, 2:], X_unlabeled[:, 2:], X_test[:, 2:]

# Labels distincts pour chaque vue
y_labeled_v1 = y_labeled.copy()
y_labeled_v2 = y_labeled.copy()

# Modèles différents
model_v1 = RandomForestClassifier(n_estimators=50, random_state=42)   # Vue 1
model_v2 = GradientBoostingClassifier(n_estimators=50, random_state=42)  # Vue 2

num_iterations = 5
num_add = 5

for i in range(num_iterations):
    # 1️⃣ Entraîner chaque modèle
    model_v1.fit(X_labeled_v1, y_labeled_v1)
    model_v2.fit(X_labeled_v2, y_labeled_v2)
    
    # 2️⃣ Prédire sur les données non étiquetées
    probs_v1 = model_v1.predict_proba(X_unlabeled_v1)
    probs_v2 = model_v2.predict_proba(X_unlabeled_v2)
    
    # 3️⃣ Sélectionner les plus confiants
    conf_v1 = np.max(probs_v1, axis=1)
    conf_v2 = np.max(probs_v2, axis=1)
    
    idx_v1 = conf_v1.argsort()[-num_add:]
    idx_v2 = conf_v2.argsort()[-num_add:]
    
    # 4️⃣ Ajouter les exemples de v1 dans v2 et vice versa
    X_labeled_v2 = np.vstack([X_labeled_v2, X_unlabeled_v2[idx_v1]])
    y_labeled_v2 = np.hstack([y_labeled_v2, model_v1.predict(X_unlabeled_v1[idx_v1])])
    
    X_labeled_v1 = np.vstack([X_labeled_v1, X_unlabeled_v1[idx_v2]])
    y_labeled_v1 = np.hstack([y_labeled_v1, model_v2.predict(X_unlabeled_v2[idx_v2])])
    
    # 5️⃣ Retirer les exemples utilisés
   # On récupère les indices à retirer (union des deux sets)
    to_remove = np.unique(np.concatenate([idx_v1, idx_v2]))
    
    # Création d’un masque de la bonne taille
    mask = np.ones(len(X_unlabeled_v1), dtype=bool)
    mask[to_remove] = False
    
    # Mise à jour des données non étiquetées
    X_unlabeled_v1 = X_unlabeled_v1[mask]
    X_unlabeled_v2 = X_unlabeled_v2[mask]
    y_unlabeled = y_unlabeled[mask]


# Évaluation finale
acc_v1 = accuracy_score(y_test, model_v1.predict(X_test_v1))
acc_v2 = accuracy_score(y_test, model_v2.predict(X_test_v2))

print(f"Accuracy Vue 1 (Random Forest) : {acc_v1:.2f}")
print(f"Accuracy Vue 2 (Gradient Boosting) : {acc_v2:.2f}")
