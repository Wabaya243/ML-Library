import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
x = np.array(ct.fit_transform(x))
x = x[:, 1:]

#la separation en phase d'entrainement et Test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling ou mise a l'echelle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #on fit le training set car on veut que le training set soit influencé par le training set
x_test = sc.transform(x_test)

#Construction du modele de Reseau de neurones
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model

# Car desequilibre dans les class
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))


# les amelioration a aporté
optimizer = Adam(learning_rate=0.001)  # on peut essayer 0.0005, 0.0001 aussi
early_stop = EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-5, restore_best_weights=True)

#reduit les learning automatiquement quand la validation stagne ou se degrade doit etre appeler dans callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)

classifier = Sequential()
#on creer les reseau

# Couche d'entrée
classifier.add(Input(shape=(11,)))

# Couche cachée 1
classifier.add(Dense(256, kernel_initializer='glorot_uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.1))

# Couche cachée 1
classifier.add(Dense(128, kernel_initializer='glorot_uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.1))

# Couche cachée 1
classifier.add(Dense(64, kernel_initializer='glorot_uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.1))

# Couche cachée 2
classifier.add(Dense(32, kernel_initializer='glorot_uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.1))

# Couche cachée 3
classifier.add(Dense(16, kernel_initializer='glorot_uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.1))

# Couche cachée 4
classifier.add(Dense(8, kernel_initializer='glorot_uniform'))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compilation du modèle
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

#pour sauvegarder automatiquement pendant l'entrainement
checkpoint = ModelCheckpoint('Save/temp_modele_upgrade.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Entraînement du modèle# Recommande de faire ça :
history = classifier.fit(x_train, y_train, batch_size = 16, epochs = 100, validation_split=0.25, callbacks=[checkpoint, early_stop, reduce_lr ], class_weight=class_weights)



plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

# Chargement du meilleur modèle
from sklearn.metrics import roc_auc_score
import os

# Charger le modèle temporaire (le nouveau)
nouveau_model_path = "Save/temp_modele_upgrade.keras"
ancien_model_path = "Save/meilleur_modele_upgrade.keras"

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


# Matrice de confusion
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédits')
plt.ylabel('Réels')
plt.title('Matrice de confusion')
plt.show()

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

# Calculer les métriques de performance de base
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Accuracy (Précision globale)
accuracy = accuracy_score(y_test, y_pred)

# Précision (pour la classe positive)
precision = precision_score(y_test, y_pred)

# Sensibilité/Recall (pour la classe positive)
sensitivity = recall_score(y_test, y_pred)

# Calculer la spécificité (pour la classe négative)
# Spécificité = TN / (TN + FP) = Recall pour la classe 0
specificity = recall_score(y_test, y_pred, pos_label=0)

# Taux d'erreur
error_rate = 1 - accuracy

# Afficher les métriques
print("=== MÉTRIQUES DE PERFORMANCE ===")
print(f"Accuracy (Précision globale): {accuracy:.3f}")
print(f"Précision (pour classe positive): {precision:.3f}")
print(f"Sensibilité/Recall (pour classe positive): {sensitivity:.3f}")
print(f"Spécificité (pour classe négative): {specificity:.3f}")
print(f"Taux d'erreur: {error_rate:.3f}")

# Afficher la matrice de confusion pour référence
print("\n=== MATRICE DE CONFUSION ===")
print("Format: [TN FP]")
print("        [FN TP]")
print(cm)


## Juste pour tester et voir qu'on peut faire autres choses que le Deep learning

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Créer le modèle avec gestion des classes déséquilibrées
rf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')

# Entraîner
rf.fit(x_train, y_train)

# Prédire
y_pred_rf = rf.predict(x_test)
y_proba_rf = rf.predict_proba(x_test)[:,1]

# Évaluer
print("=== RANDOM FOREST REPORT ===")
print(classification_report(y_test, y_pred_rf))

# Calcul du AUC
from sklearn.metrics import roc_auc_score
auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"AUC - Random Forest : {auc_rf:.3f}")



#### la partie pour sauvegarder les models et les poids ###

#pour sauvegarder les poids
classifier.save_weights('modele_poids.weights.h5')

#pour recharger 
classifier.load_weights('modele_poids.weights.h5')


#pour sauvegarder les modesl architec poids et compile 
classifier.save('modele_complet.h5')

#pour les charger
from tensorflow.keras.models import load_model
classifier = load_model('modele_complet.h5')
