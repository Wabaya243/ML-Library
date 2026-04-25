import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt

# ==============================
# 1) Charger le dataset
# ==============================
url = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
df = pd.read_csv(url, compression="zip")

# On garde seulement la température
temps = df["T (degC)"].values.astype("float32")

# Normalisation 0-1
min_val, max_val = temps.min(), temps.max()
temps_norm = (temps - min_val) / (max_val - min_val)

# Convertir en liste pour Chronos
series = temps_norm.tolist()

# ==============================
# 2) Charger le modèle pré-entraîné Chronos
# ==============================
model_name = "amazon/chronos-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Déplacer sur GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==============================
# 3) Préparer le contexte pour la prédiction
# ==============================
# On prend par exemple les 720 derniers pas (5 jours)
context = series[-720:]
input_enc = tokenizer(context, is_split_into_words=True, return_tensors="pt").to(device)

# ==============================
# 4) Générer la prédiction
# ==============================
# Prédire les 72 prochaines valeurs (12 heures)
with torch.no_grad():
    output = model.generate(**input_enc, max_new_tokens=72)

# Décoder en tokens → nombres
predicted_tokens = tokenizer.decode(output[0], skip_special_tokens=True)
predicted_nums = [float(x) for x in predicted_tokens.split()]

# Dénormaliser
predicted_nums = np.array(predicted_nums) * (max_val - min_val) + min_val

print("Prédictions dénormalisées:", predicted_nums)

# ==============================
# 5) Visualiser (optionnel)
# ==============================
plt.figure(figsize=(10,4))
plt.plot(predicted_nums, label="Prédiction Chronos")
plt.plot(temps[-720:], label="Réel")
plt.legend()
plt.show()
