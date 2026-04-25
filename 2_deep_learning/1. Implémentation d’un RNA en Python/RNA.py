import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.metrics import f1_score

#On Importe le DataSet et on l'assigne
dataset = pd.read_csv('4.2 Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# On encode les donnees categorielle
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_2 = LabelEncoder()
x[:,2] = labelencoder_X_2.fit_transform(x[:,2]) # on a encoder les sexe

#On encode les variable geographique 
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('ohe', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x), dtype=str)
x = x[:, 1:]



#la separation en phase d'entrainement et Test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling ou mise a l'echelle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #on fit le training set car on veut que le training set soit influencé par le training set
x_test = sc.transform(x_test)




# 5. SMOTE pour egaliser les sorti qui sont desequilibré il y a plus des 0 que des valeurs 1 donc risque de mauvais apprentissage
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

print("Après SMOTE :", np.bincount(y_train))




## On recherche les meilleurs hyper Parametres avec Optuna (optimisation Bayezien)
def objective(trial):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    
    # Hyperparamètres à optimiser
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    units_1 = trial.suggest_int('units_1', 8, 128)
    units_2 = trial.suggest_int('units_2', 8, 64)
    units_3 = trial.suggest_int('units_3', 8, 32)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Construction du modèle
    model = Sequential()
    model.add(Dense(units=units_1, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(units=units_2, activation='relu'))
    model.add(Dense(units=units_3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement rapide pour optuna
    model.fit(x_train, y_train, epochs=10, validation_split=0.2 , batch_size=batch_size, verbose=0)

    # Prédiction + calcul F1
    y_pred_proba = model.predict(x_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    return f1_score(y_test, y_pred)


# Lancer la recherche
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("\n Meilleurs hyperparamètres trouvés :")
print(study.best_params)
print("\n Meilleurs score : ")
print(study.best_value)

# pour recuper les meilleurs parametres 
best_params = study.best_params # on l'utilise comme ça classifier.add(Dense(units=best_params['units_2'], activation='relu'))


#Construction du modele de Reseau de neurones
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.utils import class_weight

# Calcul automatique des poids par classe parce que les class sont desequilibré
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))


# Création du modèle
classifier = Sequential()
classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Sauvegarde du meilleur modèle
checkpoint = ModelCheckpoint('Save/temp_modele.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Entraînement
history = classifier.fit(x_train, y_train, batch_size=16, epochs=100, validation_split=0.2, callbacks=[checkpoint])



# Chargement du meilleur modèle
from sklearn.metrics import roc_auc_score
import os

# Charger le modèle temporaire (le nouveau)
nouveau_model_path = "Save/temp_modele.keras"
ancien_model_path = "Save/meilleur_modele.keras"

if os.path.exists(nouveau_model_path):
    nouveau_modele = load_model(nouveau_model_path)
    y_proba_nouveau = nouveau_modele.predict(x_test)
    auc_nouveau = roc_auc_score(y_test, y_proba_nouveau)

    if os.path.exists(ancien_model_path):
        ancien_modele = load_model(ancien_model_path)
        y_proba_ancien = ancien_modele.predict(x_test)
        auc_ancien = roc_auc_score(y_test, y_proba_ancien)

        print(f"\n Ancien AUC : {auc_ancien:.3f}")
        print(f" Nouveau AUC : {auc_nouveau:.3f}")

        if auc_nouveau > auc_ancien:
            print(" Nouveau modèle meilleur → Remplacement effectué.")
            nouveau_modele.save(ancien_model_path)
            meilleur_modele = nouveau_modele
        else:
            print(" Ancien modèle conservé (meilleur AUC).")
            meilleur_modele = ancien_modele
    else:
        print(" Aucun modèle précédent. Le modèle actuel devient le meilleur.")
        nouveau_modele.save(ancien_model_path)
        meilleur_modele = nouveau_modele
else:
    raise FileNotFoundError("Aucun modèle temporaire trouvé après l'entraînement.")


# Prédiction avec le meilleur modèle
# Probabilités (score entre 0 et 1)
y_proba = meilleur_modele.predict(x_test)



#On test les seuil pour trouver la meilleur vu que les class partante et restant sont desequilibré

from sklearn.metrics import f1_score

# Tester plusieurs seuils entre 0.1 et 0.9
# Recherche du meilleur seuil basé sur F1-score
seuils = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for seuil in seuils:
    y_pred_test = (y_proba > seuil)
    score = f1_score(y_test, y_pred_test)
    f1_scores.append(score)

meilleur_seuil = seuils[np.argmax(f1_scores)]
meilleur_f1 = max(f1_scores)

print(f"\n Meilleur seuil selon F1-score : {meilleur_seuil:.2f}")
print(f" F1-score associé : {meilleur_f1:.3f}")

# Prédictions finales avec le meilleur seuil
y_pred = (y_proba > meilleur_seuil)

# Matrice de confusion avec meilleur seuil
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédits')
plt.ylabel('Réels')
plt.title('Matrice de confusion')
plt.show()



# Calculer les métriques de performance
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
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

# Tracer la courbe F1 en fonction du seuil
plt.figure()
plt.plot(seuils, f1_scores, label='F1-score')
plt.xlabel("Seuil")
plt.ylabel("F1-score")
plt.title("Optimisation du seuil de classification")
plt.axvline(x=meilleur_seuil, color='r', linestyle='--', label=f'Seuil optimal: {meilleur_seuil:.2f}')
plt.legend()
plt.grid(True)
plt.show()

# Courbe ROC & AUC
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Taux de faux positifs (1 - Spécificité)")
plt.ylabel("Taux de vrais positifs (Recall)")
plt.title("Courbe ROC")
plt.legend()
plt.grid(True)
plt.show()
