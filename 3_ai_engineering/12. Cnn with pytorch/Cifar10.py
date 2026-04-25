import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch


# ------------------------
# 1. Prétraitement
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),                     # Resize pour ResNet50
    transforms.RandomHorizontalFlip(),                # Flip horizontal aléatoire
    transforms.RandomRotation(10),                   # Rotation aléatoire ±10 degrés
    transforms.RandomResizedCrop(32, scale=(0.9,1.0)), # Crop aléatoire
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------------
# 2. Chargement des données
# ------------------------
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ------------------------
# 3. Visualisation
# ------------------------
fig, axes = plt.subplots(1, 7, figsize=(12, 4))
for i in range(7):
    image, label = train_dataset[i]
    axes[i].imshow(image.permute(1, 2, 0))
    axes[i].axis('off')
    axes[i].set_title(f'label: {label}')
plt.show()

# ------------------------
# 4. Définition du modèle CNN
# ------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self._to_linear = None
        self._get_conv_output((3, 32, 32))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.rand(1, *shape)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = F.relu(self.conv4(x))
            self._to_linear = x.numel() // x.shape[0]

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------------------
# 5. Instanciation du modèle
# ------------------------
model = SimpleCNN()
print(model)

# ------------------------
# 6. Définition perte + optimiseur
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# ------------------------
# 7. Entraînement
# ------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=15):
    model.train()  # mode entraînement
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            train_accuracy = 100 * correct_train / total_train

        print(f"Epoch: {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

train_model(model, train_loader, criterion, optimizer)

# ------------------------
# 8. Évaluation
# ------------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
    print(f"Précision du test: {100 * correct / total:.2f}%")

evaluate_model(model, test_loader)




# pour exporter 

# Passer le modèle en mode éval
model.eval()

# Exemple d'entrée factice avec la bonne forme (batch=1)
example_input = torch.rand(1, 3, 32, 32)

# Tracer le modèle
traced_script_module = torch.jit.trace(model, example_input)

# Sauvegarder
traced_script_module.save("SavePytorch/model_traced.pt")


#### Exporter en version onnx que je peux convertir apres en tensorflow
# Exemple d'entrée factice
example_input = torch.rand(1, 3, 32, 32)

# Exporter le modèle en ONNX
torch.onnx.export(model, example_input, "SavePytorch/model.onnx", 
                  input_names=['input'], output_names=['output'],
                  opset_version=11)







#sauvegarder uniquement les poids (recommandé)
torch.save(model.state_dict(), "SavePytorch/poids_model.pth")


#Sauvegarder les models en entier 
torch.save(model, "SavePytorch/model_complet.pth")


#pour chager les poids 
model = SimpleCNN()           # créer une instance du modèle (même architecture !)
model.load_state_dict(torch.load("SavePytorch/poids_model.pth"))
model.eval()                  # mettre en mode évaluation


# pour charger un model 
model = torch.load("SavePytorch/model_complet.pth")
model.eval()
