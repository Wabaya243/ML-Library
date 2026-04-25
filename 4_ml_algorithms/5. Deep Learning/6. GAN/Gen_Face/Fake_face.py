# ========================
# IMPORTS
# ========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========================
# CHARGEMENT ET PRÉTRAITEMENT DU DATASET
# ========================

# Fonction pour trier les noms de fichiers avec chiffres
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

# Taille des images
SIZE = 128
_img = []

# Chemin du dataset
path = 'Data/without_mask'
files = os.listdir(path)
files = sorted_alphanumeric(files)

# Chargement des images et normalisation dans [-1, 1]
for i in tqdm(files):
    if i == 'seed9090.png':  # ignorer cette image
        break
    else:
        img = cv2.imread(os.path.join(path, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        img = (img - 127.5) / 127.5
        _img.append(img)

# Conversion en tenseur PyTorch
_img = torch.tensor(np.array(_img), dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)

# Création du DataLoader
batch_size = 64
dataset = TensorDataset(_img)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ========================
# VISUALISATION DES IMAGES RÉELLES
# ========================
def plot_images(sqr=5):
    imgs = _img[:sqr*sqr]
    grid = make_grid(imgs, nrow=sqr, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.title("Real Images", fontsize=20)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

plot_images(6)

# ========================
# GÉNÉRATEUR
# ========================
latent_dim = 100

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Dense layer pour projeter le vecteur latent
            nn.Linear(latent_dim, 512*8*8),
            nn.BatchNorm1d(512*8*8),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape pour image
            nn.Unflatten(1, (512, 8, 8)),

            # Bloc de convolution transposée (upsampling)
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 128x128
            nn.Tanh()  # sortie entre [-1,1]
        )

    def forward(self, z):
        return self.model(z)

generator = Generator(latent_dim).to("cuda" if torch.cuda.is_available() else "cpu")

print(generator)

# ========================
# DISCRIMINATEUR
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(512*8*8, 1),
            nn.Sigmoid()  # probabilité que l’image soit réelle
        )

    def forward(self, img):
        return self.model(img)

discriminator = Discriminator().to("cuda" if torch.cuda.is_available() else "cpu")
print(discriminator)

# ========================
# OPTIMISATEURS ET PERTE
# ========================
criterion = nn.BCELoss()
lr = 2e-4

# Adam avec betas ajustés pour stabiliser l'entraînement du GAN
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# FONCTIONS D’AFFICHAGE
# ========================
def plot_generated_images(generator, epoch, n=5):
    generator.eval()
    noise = torch.randn(n*n, latent_dim, device=device)
    with torch.no_grad():
        fake_imgs = generator(noise).cpu()
    grid = make_grid(fake_imgs, nrow=n, normalize=True)
    plt.figure(figsize=(8,8))
    plt.title(f"Generated Images - Epoch {epoch}", fontsize=16)
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.show()
    generator.train()

# ========================
# ENTRAÎNEMENT DU GAN
# ========================
epochs = 250

for epoch in range(epochs):
    gen_loss_avg = 0
    disc_loss_avg = 0

    for real_imgs, in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        real_imgs = real_imgs.to(device)

        # --- Entraînement du Discriminateur ---
        noise = torch.randn(real_imgs.size(0), latent_dim, device=device)
        fake_imgs = generator(noise).detach()  # pas de gradient vers G

        real_labels = torch.full((real_imgs.size(0), 1), 0.9, device=device)  # label smoothing
        fake_labels = torch.zeros((real_imgs.size(0), 1), device=device)

        disc_real = discriminator(real_imgs)
        disc_fake = discriminator(fake_imgs)

        loss_real = criterion(disc_real, real_labels)
        loss_fake = criterion(disc_fake, fake_labels)
        disc_loss = (loss_real + loss_fake) / 2

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        
        for _ in range(2):
            # --- Entraînement du Générateur ---
            noise = torch.randn(real_imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(noise)
            output = discriminator(fake_imgs)

            gen_loss = criterion(output, real_labels)  # le générateur veut tromper D
    
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

        gen_loss_avg += gen_loss.item()
        disc_loss_avg += disc_loss.item()

    print(f"\nEpoch [{epoch+1}/{epochs}]  "
          f"Gen Loss: {gen_loss_avg/len(dataloader):.4f}  "
          f"Disc Loss: {disc_loss_avg/len(dataloader):.4f}")
    
    plot_generated_images(generator, epoch+1, n=4)

# ========================
# SAUVEGARDE DES MODÈLES
# ========================
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

