############ 1. Importer les librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import  Trainer, TrainingArguments
from transformers import AutoModelForTimeSeriesForecasting, AutoProcessor
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

############ 2. Charger le dataset électricité
# Dataset officiel UCI : Electricity Load Diagrams (1997-2011)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
df = pd.read_csv(url, sep=";", compression="zip", index_col=0, parse_dates=True)


# Nettoyage : convertir en datetime + numeric
df.index.name = "datetime"
df = df.dropna(axis=1, how="all")  # supprimer colonnes vides

# On choisit UN client (par ex. la première colonne)
electricity = df.iloc[:, 0].values.astype("float32")

# Normalisation [0,1]
min_val, max_val = electricity.min(), electricity.max()
series_norm = (electricity - min_val) / (max_val - min_val)
series = series_norm.tolist()

############ 3. Charger Chronos
model_id = "huggingface/time-series-transformer-tourism-monthly"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTimeSeriesForecasting.from_pretrained(model_id)

############ 4. Dataset PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size=512, pred_size=72):
        self.series = series
        self.window_size = window_size
        self.pred_size = pred_size

    def __len__(self):
        return len(self.series) - self.window_size - self.pred_size

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.window_size]
        y = self.series[idx + self.window_size : idx + self.window_size + self.pred_size]

        x = [f"{v:.3f}" for v in x]
        y = [f"{v:.3f}" for v in y]


        input_enc = tokenizer(x, is_split_into_words=True, return_tensors="pt", truncation=True, padding="max_length", max_length=min(self.window_size, tokenizer.model_max_length))
        label_enc = tokenizer(y, is_split_into_words=True, return_tensors="pt", truncation=True, padding="max_length", max_length=self.pred_size)

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": label_enc["input_ids"].squeeze(),
        }

# Split train/test
train_size = int(len(series) * 0.8)
train_series, test_series = series[:train_size], series[train_size:]
train_dataset, test_dataset = TimeSeriesDataset(train_series), TimeSeriesDataset(test_series)

############ 5. Fine-tuning Chronos
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    preds_decoded, labels_decoded = [], []
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
    output_dir="./chronos_electricity",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,   # augmenter en pratique
    learning_rate=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True,
)


names = [n for n, _ in model.named_parameters() if "encoder.block." in n]
unique_blocks = sorted({int(n.split("block.")[1].split(".")[0]) for n in names})
print("Nombre de blocs d'encodeur :", len(unique_blocks))


# Tout geler d’abord
for param in model.parameters():
    param.requires_grad = False

# Dégeler les deux derniers blocs de l’encodeur
last_layers = [4, 5]
for name, param in model.named_parameters():
    if any(f"encoder.block.{i}." in name for i in last_layers):
        param.requires_grad = True

# Dégeler les deux derniers blocs du décodeur
last_layers = [4, 5]
for name, param in model.named_parameters():
    if any(f"decoder.block.{i}." in name for i in last_layers):
        param.requires_grad = True

# Dégeler aussi la tête de sortie (lm_head)
for name, param in model.named_parameters():
    if "lm_head" in name:
        param.requires_grad = True

trainable = sum(p.requires_grad for p in model.parameters())
total = sum(1 for _ in model.parameters())
print(f"Paramètres entraînables : {trainable}/{total}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

############ 6. Prédiction sur 12 prochaines heures
context = test_series[-512:]
input_enc = tokenizer(context, is_split_into_words=True, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**input_enc, max_new_tokens=72)

predicted_tokens = tokenizer.decode(output[0], skip_special_tokens=True)
predicted_nums = [float(x) for x in predicted_tokens.split()]

# Dénormalisation
# Dénormalisation
predicted_nums = predicted_nums * (max_val - min_val) + min_val
real_values = np.array(test_series[:72]) * (max_val - min_val) + min_val

print("Prédictions (prix/charge élec):", predicted_nums)

plt.figure(figsize=(10, 4))
plt.plot(predicted_nums, label="Prédictions")
plt.plot(real_values, label="Réel")
plt.legend()
plt.show()