# =============================
# 1️ Librairies et Dataset
# =============================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Vérifier si GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
torch.backends.cudnn.benchmark = True


# Transformation des images (normalisation + conversion en tensor)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# Télécharger CIFAR-10
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Visualisation d'une image exemple
images, labels = next(iter(train_loader))
plt.imshow(np.transpose(images[0].numpy(), (1,2,0)))
plt.title(f"Label: {labels[0]}")
plt.show()

'''
On va découper chaque image en patchs et les transformer en vecteurs linéaires.
'''

# =============================
# 2️ Patch Embedding
# =============================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, embed_dim=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
         # x: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)  # aplatir en (batch_size, embed_dim, n_patches)
        x = x.transpose(1,2)  # (batch_size, n_patches, embed_dim)
        return x

'''
Rôle : découper l’image en patchs et les transformer en vecteurs de dimension embed_dim.
PatchEmbedding utilise une Conv2d avec stride = patch_size → chaque patch devient un vecteur.
Comme dans le Transformer texte, on ajoute un encoding positionnel pour que le modèle sache où se situe chaque patch.
'''    


# =============================
# 3️ Positional Encoding
# =============================
class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))  # +1 pour token CLS

    def forward(self, x):
        return x + self.pos_embedding

#=============================
# 4️ Transformer Block
# =============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (n_patches, batch_size, embed_dim) pour MultiheadAttention
        x2 = x.transpose(0,1)  # transpose pour MultiheadAttention: (seq_len, batch, embed)
        attn_output, _ = self.attn(x2, x2, x2)
        x2 = self.norm1(x2 + self.dropout(attn_output))
        ffn_output = self.ffn(x2)
        x2 = self.norm2(x2 + self.dropout(ffn_output))
        return x2.transpose(0,1)


'''
Partie 5 : Construire le ViT complet
On combine tout : patch embedding → ajouter token CLS → positional encoding → empiler plusieurs blocs Transformer → classification.
'''
# =============================
# 5️ Vision Transformer Complet
# =============================

import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,          # taille des images d'entrée (ex : 32x32 pour CIFAR-10)
        patch_size=4,         # taille de chaque patch (les sous-blocs de l'image)
        in_channels=3,        # nombre de canaux d'entrée (3 pour RGB)
        num_classes=10,       # nombre de classes à prédire
        embed_dim=256,        # dimension des vecteurs d'embedding
        depth=6,              # nombre de blocs Transformer empilés
        num_heads=8,          # nombre de têtes d'attention
        ff_dim=1024,           # dimension du réseau feed-forward interne
        dropout=0.1           # taux de dropout pour régularisation
    ):
        super().__init__()

        # --------------------------------------------------------------
        # 1. Patch embedding : découpe l'image en patchs et les projette
        #    dans un espace vectoriel de dimension embed_dim.
        # --------------------------------------------------------------
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)

        # Calcul du nombre total de patchs par image (ex : 32x32 / 8x8 = 16 patchs)
        n_patches = (img_size // patch_size) ** 2

        # --------------------------------------------------------------
        # 2. Token spécial [CLS] : vecteur apprenable ajouté au début
        #    de la séquence pour la classification finale.
        # --------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --------------------------------------------------------------
        # 3. Encodage de position : ajoute une information spatiale
        #    à chaque patch (car le Transformer n’a pas de notion d’ordre).
        # --------------------------------------------------------------
        self.pos_encoding = PositionalEncoding(n_patches, embed_dim)

        # --------------------------------------------------------------
        # 4. Bloc Transformer empilé "depth" fois :
        #    chaque bloc contient multi-head attention + MLP.
        # --------------------------------------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(depth)
        ])

        # --------------------------------------------------------------
        # 5. Tête de classification finale :
        #    prend le token [CLS] et le passe dans une couche Linéaire
        #    pour prédire la classe de l’image.
        # --------------------------------------------------------------
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    # ==================================================================
    # PHASE AVANT (forward)
    # ==================================================================
    def forward(self, x):
        # --------------------------------------------------------------
        # Étape 1 : convertir l’image en séquence de patchs encodés
        # --------------------------------------------------------------
        x = self.patch_embed(x)               # (batch_size, n_patches, embed_dim)

        # --------------------------------------------------------------
        # Étape 2 : créer un token [CLS] pour chaque image du batch
        # --------------------------------------------------------------
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)

        # --------------------------------------------------------------
        # Étape 3 : concaténer le token [CLS] devant les patchs
        # --------------------------------------------------------------
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 1 + n_patches, embed_dim)

        # --------------------------------------------------------------
        # Étape 4 : ajouter l’encodage de position à chaque patch
        # --------------------------------------------------------------
        x = self.pos_encoding(x)

        # --------------------------------------------------------------
        # Étape 5 : faire passer la séquence dans les blocs Transformer
        # --------------------------------------------------------------
        for blk in self.transformer_blocks:
            x = blk(x)

        # --------------------------------------------------------------
        # Étape 6 : récupérer la sortie du token [CLS] uniquement
        #           (il contient l'information globale de l'image)
        # --------------------------------------------------------------
        cls_output = x[:, 0]   # (batch_size, embed_dim)

        # --------------------------------------------------------------
        # Étape 7 : classification finale via la tête MLP
        # --------------------------------------------------------------
        out = self.mlp_head(cls_output)  # (batch_size, num_classes)

        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
model = ViT().to(device)

print(model)



# =============================
# Optimiseur et loss
# =============================
from torch.optim.lr_scheduler import CosineAnnealingLR
num_epochs = 40

# CrossEntropyLoss est standard pour la classification multi-classes
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)


# Adam est adapté aux Transformers car il ajuste automatiquement les learning rates
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)


# =============================
# Fonction d'entraînement pour une époque
# =============================
def train_epoch(model, dataloader, criterion, optimizer,scheduler ,device):
    model.train()  # mode entraînement (Dropout actif)
    running_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs} : "):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()          # Reset des gradients
        outputs = model(images)        # Forward pass
        
        loss = criterion(outputs, labels)
        loss.backward()                # Backpropagation
        optimizer.step()               # Mise à jour des poids
        

        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)  # Classe prédite = argmax
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
# =============================
# Fonction de validation
# =============================
def evaluate(model, dataloader, criterion, device):
    model.eval()  # mode évaluation (Dropout désactivé)
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Pas de calcul de gradients
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# =============================
# Boucle principale
# =============================

train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,scheduler, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# =============================
# Visualisation des losses et accuracy
# =============================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Loss par époque")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title("Accuracy par époque")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =============================
# Tester quelques images
# =============================
def visualize_predictions(model, dataloader, device, n=5):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images[:n].to(device), labels[:n].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)
    
    images = images.cpu().numpy()
    
    for i in range(n):
        plt.imshow(np.transpose(images[i], (1,2,0)))
        plt.title(f"Vrai: {labels[i].item()} | Prédit: {preds[i].item()}")
        plt.axis('off')
        plt.show()

visualize_predictions(model, test_loader, device, n=10)

from PIL import Image
from torchvision import transforms

# Charger l'image
img_path = "mon_image.jpg"
img = Image.open(img_path).convert('RGB')  # convertir en RGB si pas déjà

# Transformer pour le ViT
transform = transforms.Compose([
    transforms.Resize((32,32)),        # CIFAR-10 taille = 32x32
    transforms.ToTensor(),             # convertir en tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # même normalisation que l'entraînement
])

img_tensor = transform(img).unsqueeze(0).to(device)  # ajouter dimension batch

model.eval()  # mode évaluation
with torch.no_grad():
    output = model(img_tensor)
    predicted_class = output.argmax(1).item()

# Les classes CIFAR-10
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("Classe prédite :", classes[predicted_class])
