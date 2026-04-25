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
X = digits.data        # 64 features (8x8 pixels)
y = digits.target      # 10 classes (0 à 9)
target_names = [str(i) for i in range(10)]

print("Nombre d'échantillons:", X.shape[0])
print("Nombre de features:", X.shape[1])
print("Nombre de classes:", len(np.unique(y)))

# 2️⃣ Scénario semi-supervisé : masquer 70% des labels
rng = np.random.RandomState(42)
y_semi = y.copy()
mask = rng.rand(len(y)) < 0.7
y_semi[mask] = -1  # -1 = non étiqueté

print("Labels connus:", np.sum(y_semi != -1))
print("Labels masqués:", np.sum(y_semi == -1))

# 3️⃣ Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_semi, test_size=0.3, random_state=42, stratify=y
)

# 4️⃣ Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5️⃣ Création du modèle Self-Training avec MLPClassifier
base_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

self_training_model = SelfTrainingClassifier(
    base_clf,
    criterion='threshold',
    threshold=0.9,
    max_iter=20
)

# 6️⃣ Entraînement
self_training_model.fit(X_train, y_train)

# 7️⃣ Dataset final complet
y_final = self_training_model.transduction_
df_final = pd.DataFrame(X_train, columns=[f'pixel_{i}' for i in range(X_train.shape[1])])
df_final['Label'] = y_final
df_final['Label_name'] = df_final['Label'].apply(lambda x: str(x) if x != -1 else 'Non étiqueté')

print("Exemple du dataset final complet :")
print(df_final.head(10))

# 8️⃣ Analyse des labels ajoutés
added_indices = np.where((y_train == -1) & (y_final != -1))[0]
print(f"Nombre de labels ajoutés : {len(added_indices)}")
remaining_indices = np.where(y_final == -1)[0]
print(f"Nombre de labels restant non étiquetés : {len(remaining_indices)}")

# 9️⃣ Évaluation sur les labels connus du test set
mask_test = y_test != -1
y_pred = self_training_model.predict(X_test[mask_test])

print("Accuracy:", accuracy_score(y_test[mask_test], y_pred))
print("F1-score:", f1_score(y_test[mask_test], y_pred, average='macro'))
print(classification_report(y_test[mask_test], y_pred, target_names=target_names))

# Matrice de confusion
cm = confusion_matrix(y_test[mask_test], y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.title("Matrice de confusion - Self-Training Digits dataset")
plt.show()

# 10️⃣ Test sur nouvelles données (exemples artificiels)
# On simule quelques images simplifiées pour prédiction
new_data = np.array([
    X[0],  # première image connue
    X[100], # une autre image
    X[500]  # et une dernière image
])
new_data_scaled = scaler.transform(new_data)
predictions = self_training_model.predict(new_data_scaled)

for i, pred in enumerate(predictions):
    print(f"Exemple {i+1}: classe prédite = {pred}")
