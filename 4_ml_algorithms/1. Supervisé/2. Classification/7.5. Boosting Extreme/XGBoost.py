from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Charger le dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Séparer train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modèles de base
model_ada = AdaBoostClassifier(random_state=42)
model_gb = GradientBoostingClassifier(random_state=42)
model_xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# Ajustement des hyperParams avec GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250,300],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'subsample' : [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_xgb = GridSearchCV(model_xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(x_train, y_train)

# Meilleurs paramètres
print("Meilleurs paramètres XGBoost :", grid_xgb.best_params_)
print("Meilleurs Score :", grid_xgb.best_score_)

# Entrainer les 3 Modeles

# On recupere les meilleurs parametre
model_xgb = grid_xgb.best_estimator_

model_ada.fit(x_train, y_train)
model_gb.fit(x_train, y_train)
model_xgb.fit(x_train, y_train)


#Prediction sur le Test Set

models = {
    "AdaBoost": model_ada,
    "GradientBoosting": model_gb,
    "XGBoost (tuned)": model_xgb
}

for name, model in models.items():
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🧪 {name} Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # Matrice de confusion
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='magma')
    plt.xlabel('Prédits')
    plt.ylabel('Réels')
    plt.title('Matrice de confusion')
    plt.show()
