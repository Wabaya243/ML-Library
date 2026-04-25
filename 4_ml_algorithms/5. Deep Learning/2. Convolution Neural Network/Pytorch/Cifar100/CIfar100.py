import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# 🔎 Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# -------------------------
# 🔎 Prétraitement & Data Augmentation
# -------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

nb_classes = 100

# -------------------------
# 🔎 Définition du modèle CNN
# -------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),

            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ELU(),

            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNModel(nb_classes).to(device)
print(model)

# -------------------------
# 🔎 Optimizer & Loss
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

# -------------------------
# 🔎 Fonction d’évaluation
# -------------------------
def evaluer_modele(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_total, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return loss_total / len(loader), correct / total

# -------------------------
# 🔎 Fonction AUC
# -------------------------
def calcul_auc(model, loader):
    model.eval()
    y_true, y_proba = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_proba.append(probs)
            y_true.append(labels.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_proba = np.concatenate(y_proba)
    y_true_onehot = np.eye(nb_classes)[y_true]
    return roc_auc_score(y_true_onehot, y_proba, multi_class='ovr')

# -------------------------
# 🔎 Comparaison & sauvegarde du meilleur modèle
# -------------------------
def charger_et_comparer_models(nouveau_path, ancien_path, testloader, tol=1e-3):
    if not os.path.exists(nouveau_path):
        raise FileNotFoundError("❌ Aucun modèle temporaire trouvé après l'entraînement.")

    nouveau_modele = CNNModel(nb_classes).to(device)
    nouveau_modele.load_state_dict(torch.load(nouveau_path))
    nouveau_modele.eval()

    auc_nouveau = calcul_auc(nouveau_modele, testloader)
    loss_nouveau, acc_nouveau = evaluer_modele(nouveau_modele, testloader)

    if os.path.exists(ancien_path):
        ancien_modele = CNNModel(nb_classes).to(device)
        ancien_modele.load_state_dict(torch.load(ancien_path))
        ancien_modele.eval()

        auc_ancien = calcul_auc(ancien_modele, testloader)
        loss_ancien, acc_ancien = evaluer_modele(ancien_modele, testloader)

        print(f"\n🎯 Ancien AUC : {auc_ancien:.5f} | Ancienne précision : {acc_ancien:.5f}")
        print(f"🚀 Nouveau AUC : {auc_nouveau:.5f} | Nouvelle précision : {acc_nouveau:.5f}")

        if auc_nouveau > auc_ancien + tol:
            print("✅ Nouveau modèle meilleur en AUC → Remplacement effectué.")
            torch.save(nouveau_modele.state_dict(), ancien_path)
            return nouveau_modele
        elif abs(auc_nouveau - auc_ancien) <= tol and acc_nouveau > acc_ancien + tol:
            print("✅ AUC identique mais précision meilleure → Remplacement effectué.")
            torch.save(nouveau_modele.state_dict(), ancien_path)
            return nouveau_modele
        else:
            print("❌ Ancien modèle conservé (aucune amélioration significative).")
            return ancien_modele
    else:
        print("📁 Aucun modèle précédent. Le modèle actuel devient le meilleur.")
        torch.save(nouveau_modele.state_dict(), ancien_path)
        return nouveau_modele

# -------------------------
# 🔎 Boucle d’entraînement
# -------------------------
patience, patience_counter = 8, 0
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(50):  # ⚠️ mets 200 si tu veux, mais lent sur CPU
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    val_loss, val_acc = evaluer_modele(model, testloader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f}, "
          f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

    # Sauvegarde du modèle temporaire
    torch.save(model.state_dict(), "Save/temp_cifar100_cnn_model.pth")

    # Comparaison avec le meilleur modèle sauvegardé
    meilleur_modele = charger_et_comparer_models(
        "Save/temp_cifar100_cnn_model.pth",
        "Save/cifar100_meilleur_model.pth",
        testloader
    )

    # Early stopping
    if val_accs[-1] < max(val_accs):
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping déclenché")
            break
    else:
        patience_counter = 0

# -------------------------
# 🔎 Évaluation finale
# -------------------------
final_loss, final_acc = evaluer_modele(meilleur_modele, testloader)
final_auc = calcul_auc(meilleur_modele, testloader)
print(f"✅ Précision finale : {final_acc:.4f}")
print(f"✅ Perte finale : {final_loss:.4f}")
print(f"✅ AUC final : {final_auc:.4f}")
