import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import os

# -----------------------------
# Labels CIFAR-100 en français
# -----------------------------
cifar100_labels_fr = {
    0: 'pomme', 1: 'poisson d’aquarium', 2: 'bébé', 3: 'ours', 4: 'castor',
    5: 'lit', 6: 'abeille', 7: 'scarabée', 8: 'vélo', 9: 'bouteille',
    10: 'bol', 11: 'garçon', 12: 'pont', 13: 'bus', 14: 'papillon',
    15: 'chameau', 16: 'boîte de conserve', 17: 'château', 18: 'chenille', 19: 'bétail',
    20: 'chaise', 21: 'chimpanzé', 22: 'horloge', 23: 'nuage', 24: 'cafard',
    25: 'canapé', 26: 'crabe', 27: 'crocodile', 28: 'tasse', 29: 'dinosaure',
    30: 'dauphin', 31: 'éléphant', 32: 'poisson plat', 33: 'forêt', 34: 'renard',
    35: 'fille', 36: 'hamster', 37: 'maison', 38: 'kangourou', 39: 'clavier',
    40: 'lampe', 41: 'tondeuse à gazon', 42: 'léopard', 43: 'lion', 44: 'lézard',
    45: 'homard', 46: 'homme', 47: 'érable', 48: 'moto', 49: 'montagne',
    50: 'souris', 51: 'champignon', 52: 'chêne', 53: 'orange', 54: 'orchidée',
    55: 'loutre', 56: 'palmier', 57: 'poire', 58: 'pick-up', 59: 'pin',
    60: 'plaine', 61: 'assiette', 62: 'coquelicot', 63: 'porc-épic', 64: 'opossum',
    65: 'lapin', 66: 'raton laveur', 67: 'raie', 68: 'route', 69: 'fusée',
    70: 'rose', 71: 'mer', 72: 'phoque', 73: 'requin', 74: 'musaraigne',
    75: 'moufette', 76: 'gratte-ciel', 77: 'escargot', 78: 'serpent', 79: 'araignée',
    80: 'écureuil', 81: 'tramway', 82: 'tournesol', 83: 'poivron', 84: 'table',
    85: 'char', 86: 'téléphone', 87: 'télévision', 88: 'tigre', 89: 'tracteur',
    90: 'train', 91: 'truite', 92: 'tulipe', 93: 'tortue', 94: 'armoire',
    95: 'baleine', 96: 'saule', 97: 'loup', 98: 'femme', 99: 'ver'
}

# -----------------------------
# Définition du device : GPU si disponible, sinon CPU
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Préparation des transformations pour l'augmentation des données
# -----------------------------
transform_train = transforms.Compose([
    transforms.Resize((224,224)),                     # Resize pour ResNet50
    transforms.RandomHorizontalFlip(),                # Flip horizontal aléatoire
    transforms.RandomRotation(10),                   # Rotation aléatoire ±10 degrés
    transforms.RandomResizedCrop(32, scale=(0.9,1.0)), # Crop aléatoire
    transforms.ToTensor(),                            # Convertir en Tensor PyTorch
    transforms.Normalize((0.5071, 0.4865, 0.4409),   # Normalisation selon CIFAR-100
                         (0.2673, 0.2564, 0.2761))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),                     # Resize pour ResNet50
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2673, 0.2564, 0.2761))
])

# -----------------------------
# Chargement des datasets CIFAR-100
# -----------------------------
dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Split train/validation (80% / 20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=6)


# -----------------------------
# Modèle ResNet50 pré-entraîné
# -----------------------------
model = models.resnet50(pretrained=True)

# Gel de toutes les couches (freeze) pour ne pas toucher aux poids pré-entraînés
for param in model.parameters():
    param.requires_grad = False

# Remplacement de la tête pour CIFAR-100
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),  # Dense
    nn.ReLU(),                  # Activation
    nn.Dropout(0.3),            # Dropout pour régularisation
    nn.Linear(1024, 512),       # Dense
    nn.ReLU(),                  # Activation
    nn.Dropout(0.3),            # Dropout pour régularisation
    nn.Linear(512, 256),       # Dense
    nn.ReLU(),                  # Activation
    nn.Dropout(0.3),            # Dropout pour régularisation
    nn.Linear(256, 100)         # Sortie 100 classes
)

model = model.to(device)

# -----------------------------
# Définition de la perte, optimiseur et scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss()                     # Perte classification multi-classes
optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=50)    # Scheduler cosinus pour la LR

# -----------------------------
# Fonction d'évaluation
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)  # Classe prédite = max softmax
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# -----------------------------
# Entraînement avec sauvegarde du meilleur modèle
# -----------------------------

os.makedirs("Save", exist_ok=True)
best_acc = 0.0
save_path_head = "Save/best_resnet_cifar100_head.pth"
num_epochs_head = 50

for epoch in range(num_epochs_head):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(model, val_loader)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path_head)
        print(f" [HEAD] Epoch {epoch+1}: meilleur modèle sauvegardé (val_acc={val_acc:.4f})")

    print(f"[HEAD] Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")


# -----------------------------
# Fine-tuning : dégel des dernières couches
# -----------------------------


for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name:
        param.requires_grad = True

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=1e-4, momentum=0.9, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=25)

save_path_fine = "Save/best_resnet_cifar100_fine.pth"
num_epochs_fine = 30

for epoch in range(num_epochs_fine):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(model, val_loader)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path_fine)
        print(f" [FINE] Epoch {epoch+1}: meilleur modèle sauvegardé (val_acc={val_acc:.4f})")

    print(f"[FINE] Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")


# -----------------------------
# Charger le meilleur modèle et évaluer sur le test set
# -----------------------------
model.load_state_dict(torch.load(save_path_fine))
test_acc = evaluate(model, test_loader)
print(f" Test accuracy finale : {test_acc:.4f}")

# -----------------------------
# Fonction pour prédire une image externe
# -----------------------------
def predire_image_externe(path_image):
    img = Image.open(path_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2761))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prédit : {cifar100_labels_fr[pred]}")
    plt.show()
    return cifar100_labels_fr[pred]

# Exemple :
# predire_image_externe("mon_image.jpg")
