# -------------------- 1. IMPORTS & SETUP --------------------

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
import warnings
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

# -------------------- 2. DATA PREP --------------------

# Chargement du dataset Titanic
dataset = pd.read_csv('titanic_data.csv')

# Définition des variables indépendantes (features)
x = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Définition de la variable cible (target)
y = dataset['Survived']

# -------------------- 2.1 GESTION DES VALEURS MANQUANTES --------------------

# Age → remplissage par la médiane
x['Age'] = x['Age'].fillna(x['Age'].median())

# Embarked → remplissage par la valeur la plus fréquente (mode)
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])

# Vérif finale
print("🔍 Valeurs manquantes restantes :\n", x.isnull().sum())


# Récupération des colonnes catégorielles (CatBoost les accepte nativement)
cat_cols = x.select_dtypes(include=['object', 'category']).columns.tolist()

# Pour XGBoost et LightGBM : encodage LabelEncoder nécessaire
x_encoded = x.copy()
for col in cat_cols:
    le = LabelEncoder()
    x_encoded[col] = le.fit_transform(x_encoded[col].astype(str))  # important de caster en str

# Pour CatBoost : on garde les indices des colonnes catégorielles
cat_features_indices = [x.columns.get_loc(col) for col in cat_cols]

# Split stratifié train/test
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, test_size=0.2, stratify=y, random_state=42
)


# -------------------- 3. HYPERPARAMS & CV --------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grille d’hyperparamètres adaptée à chaque modèle
param_grids = {
    'XGBoost': {
        'n_estimators': [50, 100, 200, 250, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0],
    },
    'LightGBM': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [-1, 3, 5, 7, 9],
        'learning_rate': [0.05, 0.1, 0.2],
    },
    'CatBoost': {
        'iterations': [50, 100, 150,200],
        'depth': [ 4, 5, 6,7],
        'learning_rate': [0.05, 0.1, 0.2],
    },
}

# Modèles à évaluer
models = {
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

# -------------------- 4. TRAINING & TUNING --------------------

best_models = {}

for name, model in models.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(
        model,
        param_grids[name],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    start = time.time()
    if name == 'CatBoost':
        # CatBoost peut bosser avec données brutes + indices catégories
        grid.fit(x_train, y_train, cat_features=cat_features_indices, early_stopping_rounds=50, eval_set=[(x_test, y_test)], verbose=0)
    else:
        # Les autres modèles utilisent les données encodées
        grid.fit(x_train, y_train)
    end = time.time()

    best_models[name] = {
        'model': grid.best_estimator_,
        'cv_score': grid.best_score_,
        'time': end - start
    }

    print(f"✅ {name} - Best CV Accuracy: {grid.best_score_:.4f} | Time: {end - start:.1f}s")
    print(f"Best Params: {grid.best_params_}")

# -------------------- 5. ÉVALUATION TEST SET --------------------

print("\n=== 📈 Résultats sur test set ===")
for name, result in best_models.items():
    model = result['model']

    if name == 'CatBoost':
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred))

# -------------------- 6. FEATURE IMPORTANCE --------------------

for name, result in best_models.items():
    model = result['model']
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(8, 5))
        plt.title(f"{name} - Feature Importances")
        plt.barh(x.columns, model.feature_importances_)
        plt.tight_layout()
        plt.show()
