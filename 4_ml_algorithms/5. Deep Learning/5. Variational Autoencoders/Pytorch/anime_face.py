"""
VAE convolutionnel PyTorch pour dataset d'images (anime faces)
Remplace l'implémentation TensorFlow fournie par une version PyTorch complète avec sampling et KL-loss.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils

# -------------------------
# Réglages / seed / device
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Chemin vers les images
# -------------------------
# Le dossier doit être structuré pour ImageFolder, par exemple:
# /kaggle/input/animefacedataset/images/<image_files...>
DATA_DIR = "/kaggle/input/animefacedataset"  # change si besoin
IMAGES_SUBFOLDER = "images"  # le dossier contenant réellement les images
ROOT_DIR = os.path.join(DATA_DIR, IMAGES_SUBFOLDER)

# -------------------------
# Hyper-paramètres
# -------------------------
IMAGE_SIZE = 64            # 64x64 comme dans ton code TF
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 128           # latent_dim = 128 dans ton TF
LEARNING_RATE = 1e-3
PATIENCE = 5               # early stopping patience
VALID_SPLIT = 0.1          # fraction du dataset utilisé en validation

# -------------------------
# Transformations / Dataset
# -------------------------
# On met ToTensor() et on garde valeurs [0,1] (comme ton code TF qui divisait par 255)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),    # resize
    transforms.ToTensor(),                          # range [0,1], CxHxW
])

# ImageFolder lira tous les fichiers dans ROOT_DIR/* (sous-dossiers comme classes)
dataset = datasets.ImageFolder(root=ROOT_DIR, transform=transform)
print(f"Dataset size: {len(dataset)} images")

# Split train / val
val_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(SEED))
print(f"Train size: {train_size}, Val size: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------
# Modèle VAE conv
# -------------------------
# Encoder -> z_mean, z_logvar -> reparam -> Decoder (ConvTranspose)
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder conv ---
        # Input shape (B, 3, 64, 64)
        self.enc_conv = nn.Sequential(
            # Conv block 1: 3 -> 32, output 32x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv block 2: 32 -> 64, output 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv block 3: 64 -> 128, output 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # On garde la dimension spatiale pour le décodeur
        # Après convs: tensor shape (B, 128, 8, 8)
        self.flatten_dim = 128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)  # 128 * 8 * 8 = 8192

        # MLPs pour mu et logvar
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Décodeur : projection puis convtranspose
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.dec_conv = nn.Sequential(
            # décoder de (B, 128, 8, 8) -> (B, 128, 8, 8)
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # upsample -> (B, 64, 16, 16)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # upsample -> (B, 32, 32, 32)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # upsample -> (B, 3, 64, 64)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # output in [0,1] to match training data
        )

    def encode(self, x):
        """
        Encode l'image en représentation aplatie, puis retourne mu et logvar.
        """
        h = self.enc_conv(x)             # (B, 128, 8, 8)
        h = h.view(h.size(0), -1)        # (B, flatten_dim)
        mu = self.fc_mu(h)               # (B, latent_dim)
        logvar = self.fc_logvar(h)       # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick : z = mu + sigma * eps
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mu.device)
        return mu + eps * std

    def decode(self, z):
        """
        Décode un vecteur latent z en image (B, 3, 64, 64)
        """
        h = self.fc_dec(z)                       # (B, flatten_dim)
        h = h.view(-1, 128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)  # (B, 128, 8, 8)
        x_recon = self.dec_conv(h)               # (B, 3, 64, 64)
        return x_recon

    def forward(self, x):
        """
        Forward complet : retourne reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Instantiate model
model = ConvVAE(latent_dim=LATENT_DIM).to(device)
print(model)
# -------------------------
# Optimizer / Loss helpers
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def loss_function(recon_x, x, mu, logvar):
    """
    VAE loss = reconstruction_loss + KL divergence
    - reconstruction_loss : MSE summed over pixels (like your TF mse)
      we average per batch to keep scale stable
    - KL divergence :  -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    # Reconstruction: MSE per sample (sum over pixels), then mean over batch
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    # KL divergence per batch (sum over latent dims, mean over batch)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + kl_loss, recon_loss.detach(), kl_loss.detach()

# -------------------------
# Utilities : plotting & generation
# -------------------------
def show_images_grid(images, title=None, nrow=8):
    """
    Affiche un batch d'images (tensor CPU float [0,1]) en grille.
    images : tensor (B, C, H, W)
    """
    images = images.cpu()
    grid = utils.make_grid(images, nrow=nrow, padding=2)
    npimg = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(12, 6))
    plt.imshow(npimg)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def generate_random_images(model, num_images=10):
    """
    Génère `num_images` à partir de z ~ N(0,1) et renvoie un tensor (num_images, 3, H, W).
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, model.latent_dim).to(device)
        gen = model.decode(z)  # sortie déjà sigmoïde en [0,1]
    return gen.cpu()

def visualize_original_vs_generated(orig_batch, gen_batch, n=8):
    """
    Montre n images originales vs générées côte à côte.
    orig_batch, gen_batch : tensors (B, C, H, W) values in [0,1]
    """
    B = min(n, orig_batch.size(0), gen_batch.size(0))
    combined = torch.cat([orig_batch[:B], gen_batch[:B]], dim=0)
    show_images_grid(combined, title="Top row: original | Bottom row: generated", nrow=B)

# -------------------------
# Training loop with EarlyStopping and callback generation
# -------------------------
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_recon = 0.0
    train_kl = 0.0
    for batch_idx, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)  # imgs shape: (B, 3, 64, 64)
        optimizer.zero_grad()

        recon, mu, logvar = model(imgs)
        loss, recon_loss_val, kl_loss_val = loss_function(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon += recon_loss_val.item()
        train_kl += kl_loss_val.item()

    train_loss /= len(train_loader)
    train_recon /= len(train_loader)
    train_kl /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_recon = 0.0
    val_kl = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss, recon_loss_val, kl_loss_val = loss_function(recon, imgs, mu, logvar)
            val_loss += loss.item()
            val_recon += recon_loss_val.item()
            val_kl += kl_loss_val.item()

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)

    print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) "
          f"| Val loss: {val_loss:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f})")

    # Générer des images aléatoires et les afficher (callback analog)
    gen_imgs = generate_random_images(model, num_images=8)
    show_images_grid(gen_imgs, title=f"Generated images at epoch {epoch}", nrow=8)

    # Early stopping sur la validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # sauvegarder le meilleur modèle
        torch.save(model.state_dict(), "best_conv_vae.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered (no improvement for {PATIENCE} epochs).")
            break

# -------------------------
# Après entraînement : charger le meilleur modèle et visualiser reconstructions
# -------------------------
model.load_state_dict(torch.load("best_conv_vae.pth", map_location=device))
model.eval()

# Prendre un batch d'images du loader de validation pour comparer
imgs, _ = next(iter(val_loader))
imgs = imgs.to(device)
with torch.no_grad():
    recon, _, _ = model(imgs)

# Affichage : top row original, bottom row reconstructions
visualize_original_vs_generated(imgs.cpu(), recon.cpu(), n=8)

# Générer une grille 10x10 à partir du latent space (utile visuellement)
def generate_manifold_grid(model, grid_n=10, grid_range=4.0):
    """
    Génère une grille d'images en balayant latents sur [-grid_range, grid_range].
    Note: latent_dim >> 2 => on ne peut pas faire un vrai 2D manifold.
    Ici on fixe toutes les dimensions sauf 2 premières pour visualiser.
    """
    model.eval()
    with torch.no_grad():
        # on fixe z dims > 2 à 0 et on balaye les 2 premières dims
        grid_x = np.linspace(-grid_range, grid_range, grid_n)
        grid_y = np.linspace(-grid_range, grid_range, grid_n)[::-1]
        figure = np.zeros((IMAGE_SIZE * grid_n, IMAGE_SIZE * grid_n, 3), dtype=np.float32)

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = np.zeros((1, model.latent_dim), dtype=np.float32)
                z[0, 0] = xi
                z[0, 1] = yi
                z_t = torch.tensor(z).to(device)
                x_dec = model.decode(z_t).cpu().squeeze(0).permute(1, 2, 0).numpy()  # HWC
                i0 = i * IMAGE_SIZE
                j0 = j * IMAGE_SIZE
                figure[i0:i0 + IMAGE_SIZE, j0:j0 + IMAGE_SIZE, :] = x_dec

        plt.figure(figsize=(10, 10))
        plt.imshow(np.clip(figure, 0, 1))
        plt.axis("off")
        plt.show()

# Attention : generate_manifold_grid fonctionne "visuellement" même si latent_dim > 2
generate_manifold_grid(model, grid_n=8)

print("Training done. Best model saved as best_conv_vae.pth")
