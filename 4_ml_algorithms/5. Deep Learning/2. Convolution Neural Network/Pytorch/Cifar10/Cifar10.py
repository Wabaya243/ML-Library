# version PyTorch

import os, sys
# Compatible Spyder, Jupyter, etc.
sys.path.append(os.getcwd())  # ajoute le dossier actuel au PYTHONPATH


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # pour Spyder / Jupyter
from PIL import Image
from dataset_utils import SubsetWithTransform

# chemins de sauvegarde (analogues à ton code TF, extension .pth pour PyTorch)
TEMP_PATH = "Save/temp_cifar10_cnn_model_improved.pth"
BEST_PATH = "Save/cifar10_meilleur_model_improved.pth"
os.makedirs("Save", exist_ok=True)

# -------------------------------
# Configuration générale
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
num_epochs = 80
patience = 15  # early stopping patience (val_accuracy)
initial_lr = 0.01

# -------------------------------
# Prétraitement et augmentation (équivalent ImageDataGenerator)
# -------------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(32, scale=(0.95, 1.0)),  # zoom ~ 0.05
    transforms.ToTensor(),
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
])

# chargement CIFAR-10
trainval_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
# On téléchargera test avec transform_eval (pas d'augmentation)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_eval)

# split train / val = 0.8 / 0.2 
train_size = int(0.8 * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))


# remplacer val_dataset par SubsetWithTransform qui applique transform_eval
val_dataset = SubsetWithTransform(val_dataset, transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# -------------------------------
# Définition du modèle (même architecture que tf.keras.Sequential improved_model)
# -------------------------------
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Bloc 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),
        ) # 224 -> 112 # si image 224x224
        # Bloc 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05),
        ) # 112 -> 56
        # Bloc 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
        ) # 56 -> 28
        # Bloc 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        ) # on a pas appliqué de Maxpooling ici

        # Classifier (Flatten + Dense)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # permet de rendre le reseau independant de la taille d'entrée au lieu de faire 256 * 28 * 28 on fait direct 256 
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


print("l'appareil utilisé est :", device)

model = ImprovedCNN().to(device)

# afficher résumé simple
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("nombre des parametres du model:", num_params)

# -------------------------------
# Optimizer, Loss, Scheduler
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)

# scheduler cosinus sur tout l'entrainement (T_max en itérations d'epoch; on choisit T_max = steps_per_epoch * 50 comme TF)
steps_per_epoch = len(train_loader)
T_max = max(1, steps_per_epoch * num_epochs)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

# ReduceLROnPlateau (moniteur val_accuracy) pour réduire lr si stagnation
reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

# -------------------------------
# Fonctions utilitaires : calcul AUC et comparer modèles
# -------------------------------
def calcul_auc_torch(model, loader):
    """
    Retourne roc_auc_score multi-class (ovr) sur l'ensemble fourni (loader).
    """
    model.eval()
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # probabilités
            probs_list.append(probs)
            labels_list.append(y.numpy())
    probs_all = np.concatenate(probs_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    # convertir labels en one-hot pour roc_auc_score
    labels_onehot = label_binarize(labels_all, classes=np.arange(10))
    try:
        auc = roc_auc_score(labels_onehot, probs_all, average='macro', multi_class='ovr')
    except Exception as e:
        # si erreur (par ex. une classe manquante), retourner NaN
        print("Warning AUC:", e)
        auc = float("nan")
    return auc



def charger_et_comparer_models(nouveau_path, ancien_path, test_loader, tol=1e-5):
    """
    Charge le modèle temporaire, calcule AUC & accuracy sur test_loader,
    compare avec ancien modèle :
    - si AUC nouveau > AUC ancien + tol -> remplacer
    - elif abs(AUC diff) <= tol and acc nouveau > acc ancien + tol -> remplacer
    - else garder ancien
    Retourne l'objet modèle chargé (avec weights du meilleur).
    """
    if not os.path.exists(nouveau_path):
        raise FileNotFoundError(" Aucun modèle temporaire trouvé après l'entraînement.")

    # fonction locale pour évaluer accuracy & auc d'un state_dict path
    def eval_from_path(path):
        m = ImprovedCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        # accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = m(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        auc = calcul_auc_torch(m, test_loader)
        return m, auc, acc

    print("\nChargement du modèle nouvellement entraîné...")
    nouveau_modele, auc_nouveau, acc_nouveau = eval_from_path(nouveau_path)

    if os.path.exists(ancien_path):
        print("Chargement de l'ancien modèle...")
        ancien_modele, auc_ancien, acc_ancien = eval_from_path(ancien_path)

        print(f"\nAncien AUC : {auc_ancien:.5f} | Ancienne précision : {acc_ancien:.5f}")
        print(f"Nouveau AUC : {auc_nouveau:.5f} | Nouvelle précision : {acc_nouveau:.5f}")

        if np.isfinite(auc_nouveau) and (auc_nouveau > auc_ancien + tol):
            print("Nouveau modèle meilleur en AUC → Remplacement effectué.")
            torch.save(nouveau_modele.state_dict(), ancien_path)
            return nouveau_modele
        elif np.isfinite(auc_nouveau) and np.isfinite(auc_ancien) and (abs(auc_nouveau - auc_ancien) <= tol) and (acc_nouveau > acc_ancien + tol):
            print("AUC identique mais précision meilleure → Remplacement effectué.")
            torch.save(nouveau_modele.state_dict(), ancien_path)
            return nouveau_modele
        else:
            print("Ancien modèle conservé (aucune amélioration significative).")
            return ancien_modele
    else:
        print("Aucun modèle précédent. Le modèle actuel devient le meilleur.")
        torch.save(nouveau_modele.state_dict(), ancien_path)
        return nouveau_modele



# -------------------------------
# Entraînement (avec early stopping, checkpoint temporaire)
# -------------------------------
def train_model():
    best_val_acc = 0.0
    wait = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # scheduler cosinus (on le step ici par epoch, comme dans TF on a un decay global)
        cosine_scheduler.step()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total

        # validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total

        # ReduceLROnPlateau sur val_accuracy (mode='max')
        reduce_on_plateau.step(epoch_val_acc)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss: {epoch_train_loss:.4f} train_acc: {epoch_train_acc:.4f} | val_loss: {epoch_val_loss:.4f} val_acc: {epoch_val_acc:.4f}")

        # sauvegarde temporaire si amélioration val_accuracy (checkpoint équivalent ModelCheckpoint monitor='val_accuracy')
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), TEMP_PATH)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping déclenché (patience atteinte).")
                break
        
    # -------------------------------
    # Tracés des courbes d'entraînement (loss & accuracy)
    # -------------------------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot([a * 100 for a in train_accs], label="Training Accuracy (%)")
    plt.plot([a * 100 for a in val_accs], label="Validation Accuracy (%)")
    plt.title('Accuracy Par Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title('Loss Par Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_model()


# -------------------------------
# Après entraînement : charger et comparer modèles comme TF
# -------------------------------
print("\nComparaison et chargement du meilleur modèle...")
meilleur_modele = charger_et_comparer_models(TEMP_PATH, BEST_PATH, test_loader)


# -------------------------------
# Évaluation finale (accuracy, f1)
# -------------------------------
meilleur_modele.eval()
y_true = []
y_pred = []
y_probas = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = meilleur_modele(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y_probas.append(probs)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

y_probas = np.concatenate(y_probas, axis=0)
test_acc = np.mean(np.array(y_true) == np.array(y_pred))
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision du model ameliorer : {test_acc:.4f}")
print(f"F1 Score : {f1:.4f}")

# -------------------------------
# Fonction pour charger une image et prédire (équivalent charger_et_predire_image)
# -------------------------------
cifar10_labels = {
    0: 'Avion',
    1: 'Voiture',
    2: 'Oiseau',
    3: 'Chat',
    4: 'Cerf',
    5: 'Chien',
    6: 'Grenouille',
    7: 'Cheval',
    8: 'Bateau',
    9: 'Camion'
}

def charger_et_predire_image(path_image, modele_path):
    """
    Charge une image locale, prétraite (32x32), prédit la classe et affiche l'image.
    """
    model_local = ImprovedCNN().to(device)
    model_local.load_state_dict(torch.load(modele_path, map_location=device))
    model_local.eval()

    img = Image.open(path_image).convert("RGB").resize((32, 32))
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)  # (1,3,32,32)
    with torch.no_grad():
        logits = model_local(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(logits.argmax(dim=1).cpu().item())

    plt.imshow(img)
    plt.title(f"Classe prédite : {cifar10_labels[pred_class]} (idx {pred_class})")
    plt.axis('off')
    plt.show()

    return pred_class, probs

# chemin_image = "chemin/vers/ton_image.jpg"
# charger_et_predire_image(chemin_image, BEST_PATH)



