import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# Charger les données
dataset = pd.read_csv('UniversalBank.csv')
dataset.isnull().sum()

# Supprimer ID et ZIP Code
df1 = dataset.drop(['ID', 'ZIP Code'], axis=1)
df1.head()

# Heatmap des corrélations
plt.figure(figsize=(15,8))
plt.title("Heatmap qui montre la correlatione entres tous les variables", fontsize=20)
sns.heatmap(df1.corr(), annot=True, cmap='mako')
plt.show()

# -------------------------
# Modèle complet (avec toutes les variables)
# -------------------------
X = df1.iloc[:, :-1].values
y = df1.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

svm = SVC(kernel='linear', random_state=42)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('La précision du modèle (toutes variables): ', accuracy)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
plt.show()

# -------------------------
# Visualisation avec seulement 2 variables (Age et Income)
# -------------------------
X2 = df1[['Age', 'Income']].values
y2 = df1['Personal Loan'].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

sc2 = StandardScaler()
X2_train = sc2.fit_transform(X2_train)
X2_test = sc2.transform(X2_test)

svm2 = SVC(kernel='linear', random_state=42)
svm2.fit(X2_train, y2_train)

# Grille de décision
X1, X2g = np.meshgrid(
    np.arange(start = X2_test[:, 0].min() - 1, stop = X2_test[:, 0].max() + 1, step = 0.01),
    np.arange(start = X2_test[:, 1].min() - 1, stop = X2_test[:, 1].max() + 1, step = 0.01)
)

Z = svm2.predict(np.array([X1.ravel(), X2g.ravel()]).T)
Z = Z.reshape(X1.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X1, X2g, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2g.min(), X2g.max())

for i, j in enumerate(np.unique(y2_test)):
    plt.scatter(X2_test[y2_test == j, 0], X2_test[y2_test == j, 1], 
                c=ListedColormap(('red', 'green'))(i), label=j, edgecolor='k')

plt.title('SVM (Age vs Income)')
plt.xlabel('Âge (standardisé)')
plt.ylabel('Revenu (standardisé)')
plt.legend()
plt.show()

# -------------------------
# Métriques de performance
# -------------------------
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
error_rate = 1 - accuracy

print("=== MÉTRIQUES DE PERFORMANCE ===")
print(f"Accuracy (Précision globale): {accuracy:.3f}")
print(f"Précision (pour classe positive): {precision:.3f}")
print(f"Sensibilité/Recall (pour classe positive): {sensitivity:.3f}")
print(f"Spécificité (pour classe négative): {specificity:.3f}")
print(f"Taux d'erreur: {error_rate:.3f}")

print("\n=== MATRICE DE CONFUSION ===")
print("Format: [TN FP]")
print("        [FN TP]")
print(cm)
