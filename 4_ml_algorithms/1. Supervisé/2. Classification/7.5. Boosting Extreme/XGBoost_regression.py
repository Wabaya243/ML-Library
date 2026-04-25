import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Charger dataset
dataset = pd.read_csv('train.csv')
dataset.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = dataset.SalePrice
x = dataset.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
x = x.drop(["Id"], axis=1)

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Imputation des valeurs manquantes
imputer = SimpleImputer()
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# ---- Modèle de base ----
xgb_model = XGBRegressor()
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("RMSE modèle simple :", np.sqrt(mse))

# ---- Modèle avec tuning ----
xgb_1000 = XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=5,   # 👉 placé dans le constructeur
    eval_metric="rmse",        # 👉 placé dans le constructeur
    random_state=42,
    learning_rate=0.01
)

xgb_1000.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    verbose=False
)

# Évaluation
y_pred2 = xgb_1000.predict(x_test)
mse2 = mean_squared_error(y_test, y_pred2)
print("RMSE modèle avec tuning :", np.sqrt(mse2))
