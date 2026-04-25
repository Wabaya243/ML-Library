# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Chargement du dataset
dataset = pd.read_csv("creditcard.csv")

# 3. Prétraitement
x = dataset.drop(['Class'], axis = 1)
y=dataset['Class']

# Normalisation (car les colonnes 'Amount' ne sont pas transformées)
from sklearn.preprocessing import StandardScaler
x['Amount'] = StandardScaler().fit_transform(x['Amount'].values.reshape(-1, 1))

# 4. Split les données avant SMOTE (pour evité les data leakage)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# 5. SMOTE  SMOTE pour egaliser les sorti qui sont desequilibré il y a plus des 0 que des valeurs 1 donc risque de mauvais apprentissage
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

print("Après SMOTE :", np.bincount(y_train))

 
# 6. entrainement avec Random Forest
model_RF = RandomForestClassifier(n_estimators=100, random_state=42)
model_RF.fit(x_train, y_train)


# 7. Prediction 
y_pred = model_RF.predict(x_test)
y_proba = model_RF.predict_proba(x_test)[:,1] # Pour la courbe ROC

# 8. Evaluation

print("Les rapport des classification : \n", classification_report(y_test, y_pred))
print("Matrice des confusion : \n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score : \n", roc_auc_score(y_test, y_proba))


# 9. Courbe ROC

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(10,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel('Taux de Faux Positive ')
plt.ylabel('Taux de Vrai Positive ')
plt.title('ROC_Curve  -  RandomForest')
plt.legend()
plt.gray()
plt.show()