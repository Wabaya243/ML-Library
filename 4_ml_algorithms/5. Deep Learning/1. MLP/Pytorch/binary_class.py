# ---------------------------
# 1) Imports et configuration
# ---------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

#verifier Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Utilisation du device : ", device)

# Fixer la graine pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------
# 2) Chargement et préparation des données
# ---------------------------
data = load_breast_cancer()
X, y = data.data, data.target

print("Shape X :", X.shape)
print("Shape y :", y.shape)
print("Classes :", np.unique(y))

# Split train/test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)  # binaire → reshape (N,1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

#envoyer le tenseur au GPU si dispo
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------------------------
# 3) Construction du modèle MLP pour classification binaire
# ---------------------------
class BinaryMLP(nn.Module):
    def __init__(self, input_dim):
        super(BinaryMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # sortie binaire
        )
        
    def forward(self, x):
        return self.model(x)

model = BinaryMLP(X_train.shape[1]).to(device) #model sur GPU

# ---------------------------
# 4) Loss et optimizer
# ---------------------------
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 5) Entraînement
# ---------------------------
num_epochs = 50
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        #Envoyez le batch sur GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    train_loss = np.mean(batch_losses)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_pred = (val_outputs > 0.5).float()
        val_acc = (val_pred == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ---------------------------
# 6) Évaluation finale
# ---------------------------
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_tensor).cpu().numpy().ravel() # repassé sur cpu pour Sklearn
    y_pred = (y_pred_probs > 0.5).astype(int)

print("\nClassification Report :\n", classification_report(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_pred_probs))

cm = confusion_matrix(y_test, y_pred)
print("\n Matrice de confusion :\n", cm)


# ---------------------------
# 7) Courbes de l'entraînement
# ---------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.title("Courbe de perte")

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Courbe de précision")

plt.show()

# ---------------------------
# 8) Prédiction sur nouvelle donnée
# ---------------------------
# Exemple : utilisateur fournit 30 features
nouvelle_donnee = np.array([14.5, 20.0, 95.0, 550.0, 0.10, 0.15, 0.05, 0.06, 0.18, 0.06,
                            0.25, 1.2, 2.0, 20.0, 0.005, 0.02, 0.02, 0.01, 0.02, 0.003,
                            16.0, 25.0, 105.0, 800.0, 0.12, 0.22, 0.07, 0.08, 0.30, 0.08]).reshape(1, -1)

nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)
nouvelle_donnee_tensor = torch.FloatTensor(nouvelle_donnee_scaled)

with torch.no_grad():
    #Envoyez la donnée sur GPU
    nouvelle_donnee_tensor = nouvelle_donnee_tensor.to(device)
    proba = model(nouvelle_donnee_tensor).item()
    classe = int(proba > 0.5)

print(f"\nProbabilité d'être malin : {proba:.4f}")
print("Classe prédite :", "Malin (1)" if classe==1 else "Bénin (0)")

# ---------------------------
# 9) Prédiction interactive
# ---------------------------
valeurs = []
for i in range(X_train.shape[1]):
    val = float(input(f"Entrer la valeur de la feature {i+1}: "))
    valeurs.append(val)

nouvelle_donnee_user = np.array(valeurs).reshape(1, -1)
nouvelle_donnee_user_scaled = scaler.transform(nouvelle_donnee_user)
nouvelle_donnee_user_tensor = torch.FloatTensor(nouvelle_donnee_user_scaled)

with torch.no_grad():
    #Envoyez la donnée sur GPU
    nouvelle_donnee_user_tensor = nouvelle_donnee_user_tensor.to(device)
    proba_user = model(nouvelle_donnee_user_tensor).item()
    classe_user = int(proba_user > 0.5)

print(f"\nProbabilité d'être malin : {proba_user:.4f}")
print("Classe prédite :", "Malin (1)" if classe_user==1 else "Bénin (0)")
