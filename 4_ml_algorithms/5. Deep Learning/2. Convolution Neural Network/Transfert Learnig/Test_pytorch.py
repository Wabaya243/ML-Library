import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ------------------
# 1. Data
# ------------------
transform_train = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
testset  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader  = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# ------------------
# 2. Modèle ResNet50
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = base_model.fc.in_features

# Remplacer la tête par une nouvelle
base_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 100)   # CIFAR-100
)

base_model = base_model.to(device)

# ------------------
# Étape 1 : Entraînement tête seule
# ------------------
# Geler le backbone
for param in base_model.parameters():
    param.requires_grad = False
for param in base_model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.fc.parameters(), lr=5e-4)

print(" Phase 1: Entraînement de la tête seule")

for epoch in range(10):
    base_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = base_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/total:.4f}, Acc: {100.*correct/total:.2f}%")

# ------------------
# Étape 2 : Fine-tuning (dé-gel partiel)
# ------------------
print(" Phase 2: Fine-tuning des dernières couches")

# Dé-geler les 20 dernières couches environ
for name, param in list(base_model.named_parameters())[-40:]:
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

early_stop_patience = 5
best_loss = float("inf")
patience_counter = 0

for epoch in range(20):
    base_model.train()
    train_loss, correct, total = 0.0, 0, 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = base_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = train_loss / total
    acc = 100. * correct / total

    # Validation
    base_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = base_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(targets).sum().item()
            val_total += targets.size(0)

    val_loss /= val_total
    val_acc = 100. * val_correct / val_total
    print(f"Epoch [{epoch+1}/20], Train Loss: {avg_loss:.4f}, Train Acc: {acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(base_model.state_dict(), "resnet50_cifar100_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping déclenché")
            break

# ------------------
# Charger modèle
# ------------------
print("🔹 Évaluation finale")
base_model.load_state_dict(torch.load("resnet50_cifar100_best.pth"))
base_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = base_model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

print(f"Précision finale sur CIFAR-100 : {100.*correct/total:.2f}%")
