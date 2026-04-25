# ---------------------------
# 1) Imports et configuration
# ---------------------------
import torch  # framework PyTorch pour deep learning
import torch.nn as nn  # modules pour construire les réseaux
import torch.optim as optim  # optimisateurs
from torch.utils.data import DataLoader, TensorDataset  # pour gérer les batches
from sklearn.datasets import load_iris  # dataset Iris
from sklearn.model_selection import train_test_split  # pour séparer train/test
from sklearn.preprocessing import StandardScaler  # normalisation des features
from sklearn.metrics import classification_report, confusion_matrix  # métriques
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
data = load_iris()  # Charger le dataset Iris
X, y = data.data, data.target  # X: features, y: labels

print("Shape X :", X.shape)   # (150,4) → 150 échantillons, 4 features
print("Shape y :", y.shape)   # (150,) → 150 labels
print("Classes :", np.unique(y))  # [0,1,2] → trois classes

# Split train/test avec stratification pour conserver la proportion des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation des features (mean=0, std=1) pour accélérer convergence du modèle
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit sur train, transforme train
X_test = scaler.transform(X_test)        # transforme test avec le même scaler

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)  # labels pour CrossEntropyLoss doivent être LongTensor
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

#envoyer le tenseur au GPU si dispo
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# Création d'un DataLoader pour itérer par batches lors de l'entraînement
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # batch_size petit pour régularisation

# ---------------------------
# 3) Construction du modèle MLP
# ---------------------------
class MLP_Iris(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_Iris, self).__init__()
        # Sequential: permet d'empiler les couches
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),  # couche dense de 16 neurones
            nn.ReLU(),                  # activation ReLU pour non-linéarité
            nn.Dropout(0.2),            # dropout pour réduire overfitting
            nn.Linear(16, 32),          # deuxième couche dense
            nn.ReLU(),                  # activation ReLU pour non-linéarité
            nn.Dropout(0.2),            # dropout pour réduire overfitting
            nn.Linear(32, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 62),   
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(62, num_classes)  # dernière couche → logits pour chaque classe
            # Pas de softmax ici ! CrossEntropyLoss le fera automatiquement
        )
    
    def forward(self, x):
        return self.model(x)  # propagation avant

# Instanciation du modèle
model = MLP_Iris(input_dim=X_train.shape[1], num_classes=3).to(device)

# ---------------------------
# 4) Loss et optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()  # pour classification multiclass
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam adapte le learning rate

# ---------------------------
# 5) Entraînement
# ---------------------------
num_epochs = 50
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(num_epochs):
    model.train()  # mode entraînement (active dropout)
    batch_losses = []
    correct, total = 0, 0  # pour calculer l'accuracy
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()          # reset gradients à chaque batch
        outputs = model(batch_X)       # propagation avant
        loss = criterion(outputs, batch_y)  # calcul de la perte
        loss.backward()                # rétropropagation
        optimizer.step()               # mise à jour des poids
        batch_losses.append(loss.item())
        
        # Calcul de l'accuracy
        _, predicted = torch.max(outputs, 1)  # max sur les logits → classe prédite
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    train_loss = np.mean(batch_losses)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation sur le jeu de test
    model.eval()  # mode évaluation (désactive dropout)
    with torch.no_grad():  # pas besoin de calculer les gradients
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        _, val_pred = torch.max(val_outputs, 1)
        val_acc = (val_pred == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ---------------------------
# 6) Évaluation finale
# ---------------------------
model.eval()
with torch.no_grad():  # pas de calcul de gradients
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs, 1)  # logits → classe prédite
    y_pred = y_pred.cpu().numpy()  # envoyer vers CPU si nécessaire

# Rapport classification et matrice de confusion
print("\nClassification Report :\n", classification_report(y_test, y_pred, target_names=data.target_names))
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :\n", cm)

# ---------------------------
# 7) Courbes d'apprentissage
# ---------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("CrossEntropy Loss")
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
# 8) Prédiction sur nouvelles données
# ---------------------------
nouvelle_donnee = np.array([[5.1, 3.5, 1.4, 0.2]])  # exemple
nouvelle_donnee_scaled = scaler.transform(nouvelle_donnee)  # normalisation
nouvelle_donnee_tensor = torch.FloatTensor(nouvelle_donnee_scaled).to(device)  # envoyer vers GPU si nécessaire

model.eval()
with torch.no_grad():
    logits = model(nouvelle_donnee_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()  # softmax pour obtenir probabilités
    classe_predite = np.argmax(probs)
    nom_classe = data.target_names[classe_predite]

print("\nProbabilités :", probs)
print("Classe prédite :", classe_predite, "-", nom_classe)

# ---------------------------
# 9) Prédiction interactive
# ---------------------------
long_sep = float(input("Longueur sépale : "))
lar_sep  = float(input("Largeur sépale : "))
long_pet = float(input("Longueur pétale : "))
lar_pet  = float(input("Largeur pétale : "))

nouvelle_donnee_user = np.array([[long_sep, lar_sep, long_pet, lar_pet]])
nouvelle_donnee_user_scaled = scaler.transform(nouvelle_donnee_user)
nouvelle_donnee_user_tensor = torch.FloatTensor(nouvelle_donnee_user_scaled)

with torch.no_grad():
    logits_user = model(nouvelle_donnee_user_tensor)
    probs_user = torch.softmax(logits_user, dim=1).numpy()
    classe_predite_user = np.argmax(probs_user)
    nom_classe_user = data.target_names[classe_predite_user]

print("\nProbabilités :", probs_user)
print("Classe prédite :", classe_predite_user, "-", nom_classe_user)
