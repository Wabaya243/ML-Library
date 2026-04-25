# Bibliothèques
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Charger le dataset Digits
digits = load_digits()
X = digits.data
y = digits.target
target_names = [str(i) for i in range(10)]

# Masquer 70% des labels
rng = np.random.RandomState(42)
y_semi = y.copy()
mask = rng.rand(len(y)) < 0.7
y_semi[mask] = -1

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_semi, test_size=0.3, random_state=42, stratify=y
)

# Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2️⃣ Self-Training avec MLPClassifier et suivi des labels ajoutés
base_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# On stockera les informations de chaque itération
labels_added_history = []

self_training_model = SelfTrainingClassifier(
    base_clf,
    criterion='threshold',
    threshold=0.9,
    max_iter=20
)

# Boucle personnalisée pour suivre la propagation
y_current = y_train.copy()
for i in range(1, 21):
    self_training_model.max_iter = 1  # on fait 1 itération à la fois
    self_training_model.fit(X_train, y_current)
    
    y_next = self_training_model.transduction_.copy()
    # Compter combien de nouveaux labels ont été ajoutés
    newly_added = np.sum((y_current == -1) & (y_next != -1))
    labels_added_history.append(newly_added)
    
    print(f"Itération {i} : labels ajoutés = {newly_added}")
    
    # Mettre à jour y_current pour la prochaine itération
    y_current = y_next.copy()
    
    # Arrêter si plus de labels ajoutés
    if newly_added == 0:
        print("Aucun label ajouté, convergence atteinte.")
        break

# Dataset final
y_final = self_training_model.transduction_
df_final = pd.DataFrame(X_train, columns=[f'pixel_{i}' for i in range(X_train.shape[1])])
df_final['Label'] = y_final
df_final['Label_name'] = df_final['Label'].apply(lambda x: str(x) if x != -1 else 'Non étiqueté')

print("Exemple du dataset final complet :")
print(df_final.head(10))

# 3️⃣ Visualisation de la propagation des labels
plt.figure(figsize=(8,5))
plt.plot(range(1,len(labels_added_history)+1), labels_added_history, marker='o')
plt.xlabel("Itération")
plt.ylabel("Nombre de labels ajoutés")
plt.title("Propagation des labels au fil des itérations")
plt.grid(True)
plt.show()

# 4️⃣ Évaluation sur le test set
mask_test = y_test != -1
y_pred = self_training_model.predict(X_test[mask_test])

print("Accuracy:", accuracy_score(y_test[mask_test], y_pred))
print("F1-score:", f1_score(y_test[mask_test], y_pred, average='macro'))
print(classification_report(y_test[mask_test], y_pred, target_names=target_names))

# 5️⃣ Matrice de confusion
cm = confusion_matrix(y_test[mask_test], y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.title("Matrice de confusion - Self-Training Digits dataset")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.show()

# 6️⃣ Analyse visuelle des erreurs
misclassified_indices = np.where(y_test[mask_test] != y_pred)[0]

plt.figure(figsize=(12,4))
for i, idx in enumerate(misclassified_indices[:9]):  # montrer seulement 9 exemples
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[mask_test][idx].reshape(8,8), cmap='gray')
    plt.title(f"Vrai: {y_test[mask_test][idx]}, Prédit: {y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Exemples d'erreurs du Self-Training")
plt.show()
