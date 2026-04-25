import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

# 1️⃣ Préparer MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Sélectionner 10% des données comme étiquetées
num_labeled = int(0.1 * len(train_full))
labeled_indices = list(range(num_labeled))
unlabeled_indices = list(range(num_labeled, len(train_full)))

labeled_set = Subset(train_full, labeled_indices)
unlabeled_set = Subset(train_full, unlabeled_indices)

labeled_loader = DataLoader(labeled_set, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 2️⃣ Définir un CNN simple
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3️⃣ Boucle de Self-training
num_iterations = 5
num_add = 500  # nombre d'exemples les plus confiants à ajouter

for it in range(num_iterations):
    # 3a) Entraîner sur les données étiquetées
    model.train()
    for X_batch, y_batch in labeled_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    # 3b) Prédire sur les données non-étiquetées
    model.eval()
    probs_list, X_unl_list, idx_list = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in unlabeled_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            probs = nn.functional.softmax(output, dim=1)
            max_conf, pred = torch.max(probs, dim=1)
            probs_list.append(max_conf.cpu())
            X_unl_list.append(X_batch.cpu())
            idx_list.append(pred.cpu())

    # 3c) Sélectionner les plus confiants
    all_conf = torch.cat(probs_list)
    all_preds = torch.cat(idx_list)
    all_X = torch.cat(X_unl_list)
    top_idx = torch.topk(all_conf, num_add).indices

    # 3d) Ajouter au dataset étiqueté
    pseudo_X = all_X[top_idx]
    pseudo_y = all_preds[top_idx]
    labeled_set = ConcatDataset([labeled_set, [(pseudo_X[i], pseudo_y[i]) for i in range(len(top_idx))]])
    labeled_loader = DataLoader(labeled_set, batch_size=64, shuffle=True)

    print(f"It {it+1}: labeled_size={len(labeled_set)}")

# 4️⃣ Évaluer sur le test set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        preds = output.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Accuracy finale du Self-training CNN : {correct/total:.3f}")
