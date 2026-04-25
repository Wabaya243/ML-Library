import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1️ Charger le modèle pré-entrainé
model = models.resnet50(pretrained=True)

# 2️ Geler toutes les couches de base
for param in model.parameters():
    param.requires_grad = False

# 3️ Ajouter une tête personnalisée
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 6)
)

model = model.to(device)
print(next(model.parameters()).device)


# 4️ Préparer les données
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder("intel_data/seg_train/", transform=transform)
val_data = datasets.ImageFolder("intel_data/seg_test", transform=transform)

train_loader = DataLoader(train_data, batch_size=128,num_workers=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, num_workers=8 ,shuffle=False)

# 5️ Définir perte et optimiseur (seulement pour la tête)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 6️ Entraîner la tête (5-10 epochs)
for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct_train / total_train
    print(f"[Tete] Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")



# 7️ Dégelons les dernières couches (layer4) pour fine-tuning
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# 8️ Réinitialiser l’optimiseur pour inclure layer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 9️ Fine-tuning complet (head + layer4)
for epoch in range(5):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct_train / total_train
    print(f"[FINE-TUNE] Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

#  Évaluation
model.eval()
correct = 0
total = 0
total_loss = 0.0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_loss = total_loss / len(val_loader.dataset)
val_acc = 100 * correct / total
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")



### test sur image 

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1️ Préparer la transformation (comme pour le training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2️ Charger l'image
img_path = "intel_data/seg_pred/mon_image.jpg"  # chemin de ton image
image = Image.open(img_path).convert("RGB")      # s'assurer que c'est RGB

# 3️ Appliquer la transformation et ajouter la dimension batch
input_tensor = transform(image).unsqueeze(0).to(device)  # shape [1, 3, 224, 224]

# 4️ Mettre le modèle en mode évaluation
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

# 5️ Afficher le résultat
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
pred_class = class_names[predicted.item()]

plt.imshow(image)
plt.title(f"Predicted: {pred_class}")
plt.axis('off')
plt.show()

print(f"Classe prédite : {pred_class}")


# Sauvegarder les poids
torch.save(model.state_dict(), "Save/best_resnet50_poids.pth")

# Recharger les poids
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 6)
)
model.load_state_dict(torch.load("Save/best_resnet50_poids.pth"))
model = model.to(device)
model.eval()


# Sauvegarder le modèle complet
torch.save(model, "Save/best_resnet50_full.pth")

# Recharger le modèle complet
model = torch.load("Save/best_resnet50_full.pth")
model = model.to(device)
model.eval()

