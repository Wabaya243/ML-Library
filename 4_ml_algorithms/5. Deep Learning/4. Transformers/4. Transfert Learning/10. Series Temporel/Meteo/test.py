############ 1.  Importer les librairies et télécharger le dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

###### 2. Charger et préparer les données

# Charger le dataset météo Jena Climate
url = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
df = pd.read_csv(url, compression="zip")

# Gardons seulement la température (T (degC)) comme cible
temps = df["T (degC)"].values.astype("float32")

#normalisé a 0 - 1
min_val, max_val = temps.min(), temps.max()
temps_norm = (temps - min_val) / (max_val - min_val)

# On convertit en liste (Chronos attend une séquence de nombres comme "tokens")
series  = temps_norm.tolist()

########## 3.  Charger Chronos pré-entraîné

model_name = "amazon/chronos-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


######### 4.  Créer Dataset PyTorch pour fine-tuning

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size=720, pred_size=72):
        self.series = series
        self.window_size = window_size
        self.pred_size = pred_size

    def __len__(self):
        return len(self.series) - self.window_size - self.pred_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size : idx + self.window_size + self.pred_size]

        # Tokenisation : Chronos prend une liste de nombres
        input_enc = tokenizer(x, is_split_into_words=True, return_tensors="pt", truncation=True, padding="max_length", max_length=self.window_size)
        label_enc = tokenizer(y, is_split_into_words=True, return_tensors="pt", truncation=True, padding="max_length", max_length=self.pred_size)

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": label_enc["input_ids"].squeeze(),
        }

# Créer dataset train/test
train_size = int(len(series) * 0.8)
train_series = series[:train_size]
test_series = series[train_size:]

train_dataset = TimeSeriesDataset(train_series)
test_dataset = TimeSeriesDataset(test_series)

'''
window_size=720 = on prend 5 jours de contexte (10min × 720 = 5 jours).
pred_size=72 = on prédit 12 heures (10min × 72 = 720 min).
Chaque échantillon = [contexte passé] → [future à prédire].
On encode avec tokenizer de Chronos.
'''

####### 5. Fine-tuning Chronos

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convertir en prédictions réelles
    preds = np.argmax(logits, axis=-1)

    # Décoder tokens en valeurs
    preds_decoded = []
    labels_decoded = []

    for p, l in zip(preds, labels):
        p_dec = tokenizer.decode(p, skip_special_tokens=True).split()
        l_dec = tokenizer.decode(l, skip_special_tokens=True).split()
        try:
            preds_decoded.extend([float(x) for x in p_dec])
            labels_decoded.extend([float(x) for x in l_dec])
        except:
            continue

    if len(preds_decoded) == 0:
        return {"mse": None, "mae": None, "mape": None}

    mse = mean_squared_error(labels_decoded, preds_decoded)
    mae = mean_absolute_error(labels_decoded, preds_decoded)
    mape = np.mean(np.abs((np.array(labels_decoded) - np.array(preds_decoded)) / np.array(labels_decoded))) * 100

    return {"mse": mse, "mae": mae, "mape": mape}



training_args = TrainingArguments(
    output_dir="./chronos_weather",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,   # pour démo, mettre plus (5–10) en pratique
    learning_rate=5e-5,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

############# 6. Tester sur de nouvelles entrées utilisateur

# On prend les 5 derniers jours du dataset test comme contexte
context = test_series[:720]

# Tokeniser
input_enc = tokenizer(context, is_split_into_words=True, return_tensors="pt")

# Générer la prédiction des 12 prochaines heures
with torch.no_grad():
    output = model.generate(**input_enc, max_new_tokens=72)

# Décoder
predicted_tokens = tokenizer.decode(output[0], skip_special_tokens=True)
print("Prédiction brute:", predicted_tokens)


'''
On prend les 5 derniers jours de données test.
On demande au modèle : “prédis-moi les 12 prochaines heures”.
On obtient une suite de valeurs (températures normalisées → qu’on peut dénormaliser après).
'''


# On convertit les tokens prédits en nombres
predicted_nums = tokenizer.convert_tokens_to_ids(predicted_tokens.split())
# On dénormalise les prédictions
predicted_nums = np.array(predicted_nums) * (max_val - min_val) + min_val
print("Prédictions dénormalisées:", predicted_nums)









