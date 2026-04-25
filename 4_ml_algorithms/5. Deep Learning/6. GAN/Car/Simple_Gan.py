# ==========================================
#  GAN avec PyTorch
# Inspiré du code TensorFlow fourni
# ==========================================

import os
import random
import glob
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ==========================================================
# ⚙️ 1. Paramètres principaux
# ==========================================================
latent_dim = 100     # dimension du bruit latent (input du générateur)
height = 32
width = 32
channels = 3
batch_size = 32
LR = 0.0002          # taux d'apprentissage
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
#  2. Préparation du Dataset
# ==========================================================

class FaceDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = [f for f in glob.glob(folder + "/*.jpg", recursive=True)]
        random.shuffle(self.files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Transformation (redimension + tensor + normalisation)
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # valeurs centrées autour de 0
])

# Exemple d'utilisation
dataset = FaceDataset("Data/train", transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==========================================================
#  3. Définition du Générateur
# ==========================================================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Dense -> reshape en carte 16x16 avec 128 canaux
            nn.Linear(latent_dim, 128 * 16 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, 16, 16)),

            # Bloc convolutionnel
            nn.Conv2d(128, 256, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, channels, 7, padding=3),
            nn.Tanh()  # sortie entre -1 et 1
        )

    def forward(self, z):
        return self.model(z)

# ==========================================================
#  4. Définition du Discriminateur
# ==========================================================

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128*2*2, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(10, 1),
            nn.Sigmoid()  # sortie = probabilité que l’image soit réelle
        )

    def forward(self, img):
        return self.model(img)

# ==========================================================
#  5. Initialisation des modèles et optimisateurs
# ==========================================================

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss

D_optimizer = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
G_optimizer = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))

# ==========================================================
#  6. Fonction pour visualiser des images
# ==========================================================

def show_images(images, n=16):
    images = images.detach().cpu().numpy()
    images = (images * 0.5) + 0.5  # re-normalisation [0,1]
    plt.figure(figsize=(6, 6))
    for i in range(n):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ==========================================================
#  7. Entraînement du GAN
# ==========================================================

for epoch in range(EPOCHS):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels vrais et faux
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # ==========================
        #  1. Entraînement Discriminateur
        # ==========================
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())

        D_loss_real = criterion(D_real, real_labels)
        D_loss_fake = criterion(D_fake, fake_labels)
        D_loss = (D_loss_real + D_loss_fake) / 2

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # ==========================
        #  2. Entraînement Générateur
        # ==========================
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        D_output = D(fake_imgs)
        G_loss = criterion(D_output, real_labels)  # le générateur veut que D pense que c’est réel

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")

    # afficher quelques images toutes les 5 époques
    if (epoch+1) % 5 == 0:
        show_images(fake_imgs[:16])
