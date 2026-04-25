# =======================
# 1. Import des librairies
# =======================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# =======================
# 2. Charger le dataset
# =======================
dataset = pd.read_csv('ionosphere.csv')   # Lecture du fichier CSV
dataset.dropna(inplace=True)              # Suppression des valeurs manquantes

# Conversion des labels en numérique : g → 1 (good), b → 0 (bad)
dataset['class'] = dataset['class'].map({'g':1 , "b":0})

# Séparer les features (X) et la cible (y)
x = dataset.drop(['class'], axis=1) 
y = dataset['class']

# =======================
# 3. Création du modèle GBM de base
# =======================
gbm_model = GradientBoostingClassifier(
    n_iter_no_change=5,        # early stopping si pas d'amélioration pendant 5 itérations
    validation_fraction=0.20,  # 20% des données train servent de validation interne
    random_state=42
).fit(x, y)


# =======================
# 4. Validation croisée (cross-validation)
# =======================
cv_results = cross_validate(
    gbm_model, x, y, cv=10,    # validation croisée en 10 folds
    scoring=["f1"],            # métrique choisie : F1-score (mieux que accuracy si dataset déséquilibré)
    return_train_score=True
)

# Afficher la moyenne des scores sur 10 folds
print("train f1 score:", cv_results['train_f1'].mean())
print("test f1 score:", cv_results['test_f1'].mean())


# =======================
# 5. Courbe d'apprentissage (Learning curve)
# =======================
train_sizes, train_scores, test_scores = learning_curve(
    gbm_model, x, y, cv=10, scoring='f1', n_jobs=1,
    train_sizes=np.linspace(0.01, 1.0, 100)   # tester avec 1% à 100% des données
)

# Moyennes des scores (train et test)
train_mean = np.mean(train_scores, axis=1)
validation_mean = np.mean(test_scores, axis=1)

# Visualisation
sns.set_style("darkgrid")
plt.plot(train_sizes, train_mean, label="Training score")
plt.plot(train_sizes, validation_mean, label="Validation score")
plt.ylabel("F1 Score", fontsize=14)
plt.xlabel("Training set size", fontsize=14)
plt.title("Learning curve for GBM", fontsize=18, y=1.03)
plt.legend()
plt.show()



# =======================
# 6. Fonction pour tracer une courbe de validation (Validation Curve)
# =======================
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    # validation_curve entraîne plusieurs modèles en faisant varier 1 paramètre
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name,
        param_range=param_range, scoring=scoring, cv=cv
    )

    # Moyenne des scores
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    # Plot
    plt.plot(param_range, mean_train_score, label="Training Score", color='b')
    plt.plot(param_range, mean_test_score, label="Validation Score", color='g')
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()



# =======================
# 7. Tester différents hyperparamètres avec validation_curve
# =======================
val_curve_params(gbm_model, x, y, "learning_rate", np.arange(0.03,0.1,0.01), scoring="f1")
val_curve_params(gbm_model, x, y, "max_depth", range(1,6), scoring="f1")
val_curve_params(gbm_model, x, y, "min_samples_leaf", range(1,41,5), scoring="f1")



# =======================
# 8. Grid Search pour trouver les meilleurs hyperparamètres
# =======================
gbm_params = {
    "learning_rate": [0.07,0.08],
    "max_depth": [1,2,3],
    "n_estimators": [10,20,30,40,50],
    "subsample": [0.5,0.6],
    "min_samples_split": range(12,16),
    "min_samples_leaf" : range(14,19),
    "max_features":[7,10,13]
}

gbm_best_grid = GridSearchCV(
    gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True
).fit(x, y)

print("Meilleurs paramètres :", gbm_best_grid.best_params_)
print("Meilleur score (cv) :", gbm_best_grid.best_score_)



# =======================
# 9. Réentraîner un modèle avec les meilleurs paramètres
# =======================
gbm_final = gbm_model.set_params(
    **gbm_best_grid.best_params_, random_state=17
).fit(x, y)


# =======================
# 10. Validation croisée finale avec le modèle optimisé
# =======================
cv_results_final = cross_validate(
    gbm_final, x, y, cv=10, scoring=["f1"], return_train_score=True
)
print("train f1 score:", cv_results_final['train_f1'].mean())
print("test f1 score:", cv_results_final['test_f1'].mean())


# =======================
# 11. Courbe d'apprentissage finale
# =======================
train_sizes_final, train_scores_final, test_scores_final = learning_curve(
    gbm_final, x, y, cv=10, scoring='f1', n_jobs=-1,
    train_sizes=np.linspace(0.01, 1.0, 100)
)

train_mean_final = np.mean(train_scores_final, axis=1)
validation_mean_final = np.mean(test_scores_final, axis=1)

sns.set_style("whitegrid")
plt.plot(train_sizes_final, train_mean_final, label='Training score')
plt.plot(train_sizes_final, validation_mean_final, label='Validation score')
plt.ylabel('F1 Score', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
plt.title('Learning curve of Final GBM', fontsize=18, y=1.03)
plt.legend()
plt.show()



# =======================
# 12. Importance des variables
# =======================
def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1)
    sns.barplot(
        x="Value", y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num]
    )
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('importances.png')

# Afficher l’importance des features
plot_importance(gbm_final, x)
