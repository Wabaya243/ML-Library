# 1. Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 2. Charger le dataset
data = load_breast_cancer()
x, y = data.data, data.target

# 3. Standarisation
scaler = StandardScaler()
x = scaler.fit_transform(x)

# 4. Separer train et test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

# 5. Modele de base : abre de decision seul
model_tree = DecisionTreeClassifier(random_state=42)
model_tree.fit(x_train, y_train)
pred_tree = model_tree.predict(x_test)

#BagginClassifier avec Arbre comme bases
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    bootstrap=True,
    random_state=42
    )

bagging.fit(x_train, y_train)
pred_bag = bagging.predict(x_test)

# 7. RandomForestClassifier (bagging + feature randomness)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)

# 8. Évaluation
print("=== Arbre de Décision ===")
print(f"Accuracy: {accuracy_score(y_test, pred_tree):.4f}")
print(confusion_matrix(y_test, pred_tree))
print()

print("=== BaggingClassifier ===")
print(f"Accuracy: {accuracy_score(y_test, pred_bag):.4f}")
print(confusion_matrix(y_test, pred_bag))
print()

print("=== RandomForestClassifier ===")
print(f"Accuracy: {accuracy_score(y_test, pred_rf):.4f}")
print(confusion_matrix(y_test, pred_rf))
print()



#Definir la grille d'hyper parametres pour voir les meilleurs parametres qu'on puissent utilisers

params = {
    'n_estimators' : [50,100,150, 200],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None],
    }

grid_search = GridSearchCV(
    estimator = RandomForestClassifier(random_state=42),
     param_grid = params,
     cv = 5,
     scoring='accuracy',
     n_jobs=1
     )

grid_search.fit(x_train, y_train)

# On affiche les meilleurs parametres des l'arbrep
print(f" Meilleur parametres : {grid_search.best_params_}")
print(f" Meilleur Score : {grid_search.best_score_}")

