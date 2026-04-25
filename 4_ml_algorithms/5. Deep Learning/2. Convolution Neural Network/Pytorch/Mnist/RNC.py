# ================================
# 📦 Imports
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Device:", device)

# ================================
# 🖼️ Fonction centrage + resize
# ================================
def center_and_resize(img, size=28):
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros((size, size), dtype=np.uint8)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    digit = img[y:y+h, x:x+w]
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset, y_offset = (28-20)//2, (28-20)//2
    canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit
    return canvas

# ================================
# 📥 Chargement MNIST
# ================================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

# Convert en numpy pour concat avec images custom
x_train = train_dataset.data.numpy().astype("float32") / 255.0
y_train = train_dataset.targets.numpy()
x_test = test_dataset.data.numpy().astype("float32") / 255.0
y_test = test_dataset.targets.numpy()

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# ================================
# 🖼️ Images locales perso
# ================================
paths = [
    ("images/mon_chiffre_1.png", 1), ("images/mon_chiffre_2.png", 2),
    ("images/mon_chiffre_3.png", 3), ("images/mon_chiffre_4.png", 4),
    ("images/mon_chiffre_5.png", 5), ("images/mon_chiffre_6.jpg", 6),
    ("images/mon_chiffre_7.png", 7), ("images/mon_chiffre_8.png", 8),
    ("images/mon_chiffre_9.png", 9)
]

X_custom, y_custom = [], []
for path, label in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image introuvable: {path}")
    img = center_and_resize(img)
    img = 255 - img
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28)
    X_custom.append(img)
    y_custom.append(label)

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)

# ================================
# 🔄 Data augmentation PyTorch
# ================================
aug_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor()
])

X_aug, y_aug = [], []
for img, label in zip(X_custom, y_custom):
    img_t = torch.tensor(img)
    for _ in range(30):  # 30 augmentations
        aug_img = aug_transform(img_t.squeeze(0))
        X_aug.append(aug_img.numpy())
        y_aug.append(label)

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)

# ================================
# 🔀 Fusion MNIST + custom
# ================================
x_train_combined = np.concatenate([x_train, X_custom, X_aug], axis=0)
y_train_combined = np.concatenate([y_train, y_custom, y_aug], axis=0)

x_train_combined, y_train_combined = shuffle(x_train_combined, y_train_combined, random_state=42)

train_tensor = TensorDataset(torch.tensor(x_train_combined), torch.tensor(y_train_combined))
test_tensor = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

train_loader = DataLoader(train_tensor, batch_size=128, shuffle=True)
test_loader = DataLoader(test_tensor, batch_size=128, shuffle=False)

# ================================
# 🧠 CNN PyTorch
# ================================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(x, 0.2, training=self.training)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout(x, 0.2, training=self.training)
        x = self.pool(F.relu(self.conv3(x)))
        x = F.dropout(x, 0.2, training=self.training)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ================================
# 🚀 Entraînement
# ================================
def train_model(model, train_loader, test_loader, epochs=10, save_path="Save/temp_mnist_cnn_model.pth"):
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
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
        train_acc = correct / total

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = correct / total
        print(f"Epoch {epoch+1}: TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

        if val_acc > best_acc:
            print("🚀 Nouveau meilleur modèle sauvegardé")
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc

train_model(model, train_loader, test_loader)

# ================================
# 📊 AUC + Comparaison ancien/nouveau modèle
# ================================
def calcul_auc(model, loader):
    model.eval()
    y_true, y_proba = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_proba.append(probs)
            y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_proba = np.concatenate(y_proba)
    return roc_auc_score(y_true, y_proba, multi_class='ovr')

def charger_et_comparer_models(temp_path, best_path, test_loader):
    nouveau_modele = CNNModel().to(device)
    nouveau_modele.load_state_dict(torch.load(temp_path))
    auc_nouveau = calcul_auc(nouveau_modele, test_loader)

    if os.path.exists(best_path):
        ancien_modele = CNNModel().to(device)
        ancien_modele.load_state_dict(torch.load(best_path))
        auc_ancien = calcul_auc(ancien_modele, test_loader)
        print(f"🎯 Ancien AUC: {auc_ancien:.3f}")
        print(f"🚀 Nouveau AUC: {auc_nouveau:.3f}")
        if auc_nouveau > auc_ancien:
            torch.save(nouveau_modele.state_dict(), best_path)
            return nouveau_modele
        else:
            return ancien_modele
    else:
        torch.save(nouveau_modele.state_dict(), best_path)
        return nouveau_modele

best_model = charger_et_comparer_models("Save/temp_mnist_cnn_model.pth", "Save/mnist_cnn_model.pth", test_loader)

# ================================
# 🔍 Prédictions locales
# ================================
def predict_image(model, filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = center_and_resize(img)
    img = 255 - img
    img = img.astype("float32") / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        pred = outputs.argmax(1).item()
    return pred

for path, label in paths:
    pred = predict_image(best_model, path)
    print(f"Image {path} | Label attendu: {label} | Prédiction: {pred}")

# ================================
# 🔄 Fine-tuning avec images locales
# ================================
X_custom_tensor = torch.tensor(X_custom, dtype=torch.float32)
y_custom_tensor = torch.tensor(y_custom, dtype=torch.long)
custom_loader = DataLoader(TensorDataset(X_custom_tensor, y_custom_tensor), batch_size=2, shuffle=True)

optimizer = optim.Adam(best_model.parameters(), lr=1e-4)
for epoch in range(5):
    best_model.train()
    for images, labels in custom_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Fine-tuning Epoch {epoch+1}/5 terminé")

torch.save(best_model.state_dict(), "Save/mnist_cnn_model.pth")
print("✅ Modèle final sauvegardé")
