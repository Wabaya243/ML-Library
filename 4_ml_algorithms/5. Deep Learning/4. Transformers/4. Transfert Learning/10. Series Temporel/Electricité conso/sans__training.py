from transformers import pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############ 1. Charger le dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
df = pd.read_csv(url, sep=";", compression="zip")

# On prend la première colonne de consommation
electricity = df.iloc[:, 0].values.astype("float32")

# Normalisation
min_val, max_val = electricity.min(), electricity.max()
series_norm = (electricity - min_val) / (max_val - min_val)
series = series_norm.tolist()

############ 2. Charger Chronos via le pipeline
forecast_pipe = pipeline(
    task="time-series-forecasting",
    model="amazon/chronos-t5-small"
)

############ 3. Prédiction
# On lui donne un bout de la série + on demande combien de pas à prévoir
result = forecast_pipe(
    series_norm[-720:],      # les 720 derniers points comme contexte
    prediction_length=72,    # prédire les 72 prochaines valeurs
    quantiles=[0.1, 0.5, 0.9]  # intervalles de confiance optionnels
)

############ 4. Récupérer les valeurs prédites (médianes)
pred = np.array(result["mean"]) * (max_val - min_val) + min_val
real = electricity[-792:-720]  # valeurs réelles correspondantes (si disponibles)

print("Prévisions :", pred)

############ 5. Visualisation
plt.figure(figsize=(10,4))
plt.plot(pred, label="Prédiction (Chronos pipeline)")
plt.plot(real, label="Réel")
plt.legend()
plt.show()
