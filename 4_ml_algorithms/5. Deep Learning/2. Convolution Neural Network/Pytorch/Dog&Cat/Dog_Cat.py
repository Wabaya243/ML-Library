# %% [markdown]
# # Reconnaissance Chien vs Chat avec PyTorch
# Implémentation "from scratch" d'un CNN sur un dataset d'images organisé en sous-dossiers.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import os

# %%
# 1. Paramètres généraux
BATCH_SIZE = 32
IMG_SIZE = (64, 64)   # redimensionner les images
EPOCHS = 20
SEED = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

# %%
# 2. Chargement des données
# Suppose que les données sont dans un dossier avec sous-dossiers 'cats' et 'dogs'
# Exemple : data/train/cats, data/train/dogs, data/test/cats, data/test/dogs

train_dir = "data/train"
validation_dir = "data/test"

# Transforms = prétraitement
transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),   # data augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # normalisation [-1,1]
])

transform_val = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=validation_dir, transform=transform_val)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# 3. Construction du modèle CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # Entrée : 3 canaux (RGB), sortie : 32 canaux, filtre 3x3, padding=1 (pour garder taille)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # réduit H et W de moitié (64→32)
            
            # Entrée : 32 canaux, sortie : 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 32→16
            
            # Entrée : 64 → sortie : 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 16→8
            
            # Encore 128 → 128
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 8→4
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # transformer (128,4,4) en vecteur 128*4*4
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),  # dépend de IMG_SIZE
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Créer le modèle
model = CNNModel().to(DEVICE)
print(model)

# %%
# 4. Compilation (optimiseur + fonction de perte)
criterion = nn.BCELoss()  # Binary CrossEntropy
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %%
# 5. Boucle d'entraînement et validation
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    # Pour stocker l'évolution des performances
    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    # Boucle sur les époques
    for epoch in range(epochs):
        # === Phase d'entraînement ===
        model.train()  # mettre le modèle en mode entraînement
        running_loss, correct, total = 0.0, 0, 0  # accumulateurs

        # Boucle sur les batchs d'entraînement
        for images, labels in train_loader:
            # Déplacer les données vers CPU ou GPU
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)  
            # `.unsqueeze(1)` ajoute une dimension pour que labels = (batch_size, 1)

            # 1) Initialiser les gradients à zéro
            optimizer.zero_grad()

            # 2) Forward : calculer les prédictions
            outputs = model(images)

            # 3) Calculer la perte
            loss = criterion(outputs, labels)

            # 4) Backward : calculer les gradients
            loss.backward()

            # 5) Mettre à jour les poids
            optimizer.step()

            # Mise à jour des métriques
            running_loss += loss.item() * images.size(0)  # perte totale
            preds = (outputs > 0.5).int()  # seuil à 0.5 pour binaire
            correct += (preds == labels.int()).sum().item()  # prédictions correctes
            total += labels.size(0)  # nombre d'images vues

        # Moyenne de la perte et précision sur tout le dataset train
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # === Phase de validation ===
        model.eval()  # mode évaluation (désactive Dropout, BatchNorm fixe)
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        # Pas de calcul de gradients en validation
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                # Forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumuler les résultats
                val_running_loss += loss.item() * images.size(0)
                preds = (outputs > 0.5).int()
                val_correct += (preds == labels.int()).sum().item()
                val_total += labels.size(0)

        # Moyenne des métriques validation
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        # Afficher le résumé de l'époque
        print(f"Epoch {epoch+1}/{epochs} "
              f"- Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} "
              f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

# Lancer l'entraînement
train_loss, train_acc, val_loss, val_acc = train_model(
    model, train_loader, val_loader, criterion, optimizer
)

# %%
# 6. Visualisation des courbes d'apprentissage
epochs_range = range(len(train_acc))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, train_acc, label="Train Acc")
plt.plot(epochs_range, val_acc, label="Val Acc")
plt.legend(loc="lower right")
plt.title("Training vs Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(epochs_range, train_loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.legend(loc="upper right")
plt.title("Training vs Validation Loss")

plt.show()

# %%
# 7. Évaluation plus poussée : matrice de confusion
y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = (outputs > 0.5).int().cpu().numpy()
        y_pred.extend(preds.flatten())
        y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.colorbar()
tick_marks = np.arange(len(train_dataset.classes))
plt.xticks(tick_marks, train_dataset.classes)
plt.yticks(tick_marks, train_dataset.classes)

# Ajouter les nombres
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# %%
# 8. Tester sur une image externe
from PIL import Image

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = (output > 0.5).int().item()

    print(f"Prédiction : {train_dataset.classes[pred]}")

# Exemple
# predict_image("some_path_to_cat.jpg")
# predict_image("some_path_to_dog.jpg")
