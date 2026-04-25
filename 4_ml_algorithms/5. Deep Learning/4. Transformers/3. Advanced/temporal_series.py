# -------------------------
# Étape 0 : Import
# -------------------------
from modulefinder import test
from turtle import forward
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Étape 1 : Charger le dataset
# -------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url, parse_dates=['Date'])
temps = df['Temp'].values.reshape(-1, 1)

#normalisation
scaler = MinMaxScaler()
temps_scaled = scaler.fit_transform(temps)

# -------------------------
# Étape 2 : Préparer les séquences
# -------------------------

sequence_length = 30 #  regarder les 30 derniers jours pour prédire le suivant

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

dataset = TimeSeriesDataset(temps_scaled, sequence_length)

train_size = int(0.8 * len(dataset))
test_size = int(0.2 * len(dataset))
train_dataset, test_datatset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_datatset, batch_size=64)

# -------------------------
# Étape 3 : Créer le Transformer from scratch
# -------------------------

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=4, ff_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = x.permute(1, 0, 2) # transformer attend(seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        x = x[-1] #prendre le dernier token
        x = self.regressor(x)
        return x

device = "cuda" if torch.cuda.is_available() else 'cpu'
model = TimeSeriesTransformer().to(device)

# -------------------------
# Étape 4 : Définir loss et optim
# ------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Étape 5 : Entraînement
# -------------------------

epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# -------------------------
# Étape 6 : Évaluation
# -------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error

model.eval()
preds = []
targets = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        output = model(x_batch)
        preds.append(output.cpu().numpy())
        targets.append(y_batch.cpu().numpy())


preds = np.concatenate(preds).flatten()
targets = np.concatenate(targets).flatten()

mse = mean_squared_error(targets, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets, preds)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Inverser pour RMSE et MAE
preds_orig = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
targets_orig = scaler.inverse_transform(targets.reshape(-1,1)).flatten()

rmse_orig = np.sqrt(mean_squared_error(targets_orig, preds_orig))
mae_orig = mean_absolute_error(targets_orig, preds_orig)

print(f"RMSE réel: {rmse_orig:.2f}°C, MAE réel: {mae_orig:.2f}°C")




import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(targets, label='Vraies valeurs')
plt.plot(preds, label='Prédictions')
plt.legend()
plt.show()


######### Étape 9 : Tester avec une nouvelle entrée utilisateur

def predict_next_day(user_sequence):
    """
    user_sequence: liste ou array des derniers 30 jours
    """
    if len(user_sequence) != sequence_length:
        raise ValueError(f"il faut exactement {sequence_length} des jours")

    #normalisation
    user_seq_scaled = scaler.transform(np.array(user_sequence).reshape(-1, 1)).T
    user_seq_scaled = torch.tensor(user_seq_scaled, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(user_seq_scaled)
        
    #inverser scale
    pred = scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1))[0,0]
    return pred

#exemple utilisateur
last_30_days = temps[-30:].flatten()
next_day_pred = predict_next_day(last_30_days)
print(f"Prédiction température pour demain : {next_day_pred:.2f}°C")

import matplotlib.pyplot as plt

# Vraies valeurs et prédictions pour le test set
plt.figure(figsize=(14,6))
plt.plot(targets, label='Vraies valeurs', color='blue')
plt.plot(preds, label='Prédictions Transformer', color='red', alpha=0.7)

# Ajouter la prédiction utilisateur (demain)
plt.scatter(len(targets), next_day_pred, color='green', label='Prédiction utilisateur (demain)', s=100, marker='X')

plt.title("Prédictions vs Valeurs réelles - Températures quotidiennes")
plt.xlabel("Jour dans le test set")
plt.ylabel("Température (°C)")
plt.legend()
plt.show()
