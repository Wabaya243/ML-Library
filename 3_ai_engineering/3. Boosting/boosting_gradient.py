# 1. Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# 2. Charger les données
data = load_breast_cancer()
x, y = data.data, data.target

# 3. Standardisation (optionnelle pour arbres, mais bonne pratique générale)
scaler = StandardScaler()
x = scaler.fit_transform(x)


# 4. separer en train et test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2 ,random_state=42)


# 5. Random Forest (Bagging)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(x_train, y_train)
pred_rf = model_rf.predict(x_test)


# 6. AdaBoost (boosting classique)

model_ada = AdaBoostClassifier(n_estimators=100, random_state=42)
model_ada.fit(x_train, y_train)
pred_ada = model_ada.predict(x_test)


# 7. Gradient Boost
model_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model_gb.fit(x_train, y_train)
pred_gb = model_gb.predict(x_test)



# 8. Affichage des résultats
def print_results(name, y_true, y_pred):
    print(f"\n==== {name} ====")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(y_true, y_pred))
    print("Rapport de classification :")
    print(classification_report(y_true, y_pred))
    # Matrice de confusion
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédits')
    plt.ylabel('Réels')
    plt.title('Matrice de confusion')
    plt.show()


print_results("Random Forest", y_test, pred_rf)
print_results("AdaBoost", y_test, pred_ada)
print_results("Gradient Boosting", y_test, pred_gb)


#Definir la grille d'hyper parametres pour voir les meilleurs parametres qu'on puissent utilisers

params = {
    'n_estimators' : [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    }

grid_search = GridSearchCV(
    estimator = GradientBoostingClassifier(random_state=42),
     param_grid = params,
     cv = 5,
     scoring='accuracy',
     n_jobs=1
     )

grid_search.fit(x_train, y_train)

# On affiche les meilleurs parametres des l'arbrep
print(f" Meilleur parametres : {grid_search.best_params_}")
print(f" Meilleur Score : {grid_search.best_score_}")
