# ---------------------------
# 1) Imports et configuration
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Fixer la graine pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

#verifier si GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2) Chargement et préparation des données
# ---------------------------
X, y = load_diabetes(return_X_y=True)
y = y.reshape(-1, 1)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# envoyer vers GPU si nécessaire
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ---------------------------
# 3) Construction du modèle MLP en PyTorch
# ---------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # sortie unique pour régression
        )
        
    def forward(self, x):
        return self.model(x)

model = MLPRegressor(X_train.shape[1]).to(device)

# ---------------------------
# 4) Définition de la loss et de l'optimizer
# ---------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 5) Entraînement
# ---------------------------
num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        #envoyer sur GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor).item()
        val_losses.append(val_loss)
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# ---------------------------
# 6) Évaluation finale
# ---------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).cpu().numpy()

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nRésultats sur le jeu de test :")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ---------------------------
# 7) Visualisation des courbes d'apprentissage
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss (MSE)')
plt.plot(val_losses, label='Validation Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Courbe d\'apprentissage MLP (régression)')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# 8) Prédictions sur nouvelles données utilisateur
# ---------------------------
print("\n--- Prédictions sur nouvelles données ---")

# Exemple de nouvelle donnée simulée (10 features)
nouvelle_donnee = np.array([[0.05, -0.02, 0.03, -0.01, 0.04, 0.02, -0.03, 0.01, -0.02, 0.03]])
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)
nouvelle_donnee_tensor = torch.FloatTensor(nouvelle_donnee_scaled)

with torch.no_grad():
    prediction = model(nouvelle_donnee_tensor).item()

print("Prédiction pour la nouvelle donnée :", prediction)

# ---------------------------
# 9) Saisie interactive par l'utilisateur
# ---------------------------
valeurs = []
for i in range(X_train.shape[1]):
    val = float(input(f"Entrer la valeur de la feature {i+1}: "))
    valeurs.append(val)

nouvelle_donnee_user = np.array(valeurs).reshape(1, -1)
nouvelle_donnee_user_scaled = scaler.transform(nouvelle_donnee_user)
nouvelle_donnee_user_tensor = torch.FloatTensor(nouvelle_donnee_user_scaled)

with torch.no_grad():
    prediction_user = model(nouvelle_donnee_user_tensor).item()

print("Prédiction pour la donnée utilisateur :", prediction_user)
