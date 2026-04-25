import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ====================================================
# Hyperparamètres
# ====================================================
NOISE_DIM = 96          # dimension du vecteur latent (bruit)
BATCH_SIZE = 128        # taille du batch
EPOCHS = 50             # nombre d'epochs
LR = 1e-4               # learning rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU si disponible

# ====================================================
# Utilitaires pour visualiser et traiter les images
# ====================================================
def show_images(images):
    """
    Affiche un tableau d'images. 
    images: tensor ou numpy array, shape [N, 784] ou [N, H, W]
    """
    images = images.detach().cpu().numpy() if torch.is_tensor(images) else np.array(images)
    if images.ndim == 2:  # images aplaties (N, 784)
        sqrtimg = int(np.sqrt(images.shape[1]))
        imgs = images.reshape((-1, sqrtimg, sqrtimg))
    else:
        imgs = images
        sqrtimg = images.shape[1]

    n = imgs.shape[0]
    sqrtn = int(math.ceil(np.sqrt(n)))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    for i in range(n):
        ax = plt.subplot(sqrtn, sqrtn, i+1)
        plt.axis('off')
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


def preprocess_img(x):
    """
    Convertit les images de [0,1] -> [-1,1]
    """
    return 2.0 * x - 1.0


def deprocess_img(x):
    """
    Convertit les images de [-1,1] -> [0,1]
    """
    return (x + 1.0) / 2.0


# ====================================================
# Dataset MNIST
# ====================================================
transform = transforms.Compose([
    transforms.ToTensor(),  # shape (H,W,C) -> (C,H,W) et [0,1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ====================================================
# Discriminateur fully-connected
# ====================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),      # couche fully connected: 784 -> 256
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 128),      # 256 -> 128
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, 1)         # sortie logits (pas de sigmoid)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # aplatissement (B, 784)
        return self.model(x)


# ====================================================
# Générateur fully-connected
# ====================================================
class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()  # sortie dans [-1,1] pour correspondre au preprocess
        )
    
    def forward(self, z):
        return self.model(z)


# ====================================================
# Fonctions de perte
# ====================================================
criterion = nn.BCEWithLogitsLoss()  # combine sigmoid + BCE

def discriminator_loss(logits_real, logits_fake):
    """
    Loss du discriminateur: -[log(D(x)) + log(1 - D(G(z)))]
    """
    real_loss = criterion(logits_real, torch.ones_like(logits_real))
    fake_loss = criterion(logits_fake, torch.zeros_like(logits_fake))
    return real_loss + fake_loss

def generator_loss(logits_fake):
    """
    Loss du générateur: -log(D(G(z))) -> on veut que D dise 1
    """
    return criterion(logits_fake, torch.ones_like(logits_fake))


# ====================================================
# Initialisation des modèles et optimizers
# ====================================================
D = Discriminator().to(DEVICE)
G = Generator(NOISE_DIM).to(DEVICE)

D_optimizer = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))
G_optimizer = optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))


# ====================================================
# Echantillonnage du bruit uniforme [-1,1]
# ====================================================
def sample_noise(batch_size, dim):
    """
    Génère un tenseur de bruit uniforme dans [-1,1]
    """
    return torch.rand(batch_size, dim, device=DEVICE) * 2 - 1


# ====================================================
# Boucle d'entraînement
# ====================================================
fixed_noise = sample_noise(16, NOISE_DIM)  # bruit fixe pour visualisation

for epoch in range(EPOCHS):
    for i, (real_images, _) in tqdm(enumerate(train_loader), desc=f"{epoch+1}/{EPOCHS}"):
        real_images = preprocess_img(real_images).to(DEVICE)  # [-1,1]
        batch_size_curr = real_images.size(0)

        # -----------------------------
        # Update Discriminateur
        # -----------------------------
        D_optimizer.zero_grad()
        z = sample_noise(batch_size_curr, NOISE_DIM)
        fake_images = G(z)

        logits_real = D(real_images)
        logits_fake = D(fake_images.detach())  # detach pour ne pas rétroprop sur G
        d_loss = discriminator_loss(logits_real, logits_fake)
        d_loss.backward()
        D_optimizer.step()

        # -----------------------------
        # Update Générateur
        # -----------------------------
        G_optimizer.zero_grad()
        logits_fake = D(fake_images)  # évalue les fake images
        g_loss = generator_loss(logits_fake)
        g_loss.backward()
        G_optimizer.step()

        # -----------------------------
        # Logging / visualisation
        # -----------------------------
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
            sample_fake = deprocess_img(fake_images[:16])
            show_images(sample_fake.view(-1,28,28))


# ====================================================
# Images finales après entraînement
# ====================================================
with torch.no_grad():
    final_samples = G(fixed_noise)
    final_samples = deprocess_img(final_samples)
    print("Images finales générées:")
    show_images(final_samples.view(-1,28,28))
