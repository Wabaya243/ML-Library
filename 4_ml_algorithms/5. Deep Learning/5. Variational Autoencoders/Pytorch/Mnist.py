# ===============================
# 1️ Importations
# ===============================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ===============================
# 2️ Paramètres globaux
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = 784       # 28x28
batch_size = 128
epochs = 50
hidden_1 = 129
hidden_2 = 64
latent_dim = 2

# ===============================
# 3️ Chargement et préparation des données
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader pour batcher les données
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# ===============================
# 4️ AutoEncoder simple
# ===============================

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # === Encodeur ===
        # But : compresser l'image (784 → latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_1),  # couche d'entrée : de 784 à 128 neurones
            nn.Sigmoid(),                       # activation non linéaire
            nn.Linear(hidden_1, hidden_2),       # couche goulot : de 128 à 64 neurones
            nn.Sigmoid()                        # activation -> valeurs entre 0 et 1
        )

        # === Décodeur ===
        # But : reconstruire l'image originale à partir du latent vector
        self.decoder = nn.Sequential(
            nn.Linear(hidden_2, hidden_1),       # de 64 à 128 neurones
            nn.Sigmoid(),
            nn.Linear(hidden_1, num_features),   # de 128 à 784 neurones (taille de sortie)
            nn.Sigmoid()                         # dernière Sigmoid -> pixels [0, 1]
        )

    def forward(self, x):
        # Aplatit l'image (B, 1, 28, 28) → (B, 784)
        # num_features = 784 ici (ou autre selon la taille de ton image)
        x = x.view(-1, num_features)  

        # Encodage (compression en latent space)
        encoded = self.encoder(x)

        # Décodage (reconstruction de l'image)
        decoded = self.decoder(encoded)

        return decoded


ae = AutoEncoder().to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# === Entraînement AE ===
for epoch in range(epochs):
    ae.train()
    total_loss = 0
    for imgs, _ in tqdm(train_loader, desc=f"{epoch+1}/{epochs}"):
        imgs = imgs.to(device)
        outputs = ae(imgs)
        loss = criterion(outputs, imgs.view(-1, num_features))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] - AE Loss: {total_loss/len(train_loader):.4f}")

# === Visualisation AE ===
def plot_reconstructions(model, loader):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recon = model(imgs).cpu().view(-1, 1, 28, 28)
    fig, axes = plt.subplots(2, 10, figsize=(10, 3))
    for i in range(10):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    plt.show()

plot_reconstructions(ae, test_loader)

# ===============================
# 5️ VAE Dense
# ==============================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------------------------
        #  ENCODEUR
        # ---------------------------
        # On part de l’image aplatie (num_features = par ex. 784 pour 28x28)
        # et on la projette dans un espace latent de taille "latent_dim".
        self.fc1 = nn.Linear(num_features, 512)   # couche cachée de l’encodeur
        self.fc_mu = nn.Linear(512, latent_dim)   # calcule la moyenne μ du latent
        self.fc_logvar = nn.Linear(512, latent_dim) # calcule log(variance) du latent

        # ---------------------------
        #  DÉCODEUR
        # ---------------------------
        # Reconstruit l’image à partir du vecteur latent z
        self.fc_dec1 = nn.Linear(latent_dim, 512)  # couche cachée du décodeur
        self.fc_out = nn.Linear(512, num_features) # sortie → pixels reconstruits

    # ---------------------------
    # Étape 1 : Encoder
    # ---------------------------
    def encode(self, x):
        # x : image aplatie (batch_size, num_features)
        h = F.relu(self.fc1(x))         # activation non linéaire
        # Retourne les deux paramètres statistiques du latent
        return self.fc_mu(h), self.fc_logvar(h)

    # ---------------------------
    # Étape 2 : Reparamétrisation
    # ---------------------------
    def reparameterize(self, mu, logvar):
        # logvar = log(σ²) → donc std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # échantillon aléatoire standard gaussien
        eps = torch.randn_like(std)
        # z = μ + σ * ε  (trick de reparamétrisation)
        # permet de rendre le sampling différentiable
        return mu + eps * std

    # ---------------------------
    # Étape 3 : Décoder
    # ---------------------------
    def decode(self, z):
        # z : vecteur latent (batch_size, latent_dim)
        h = F.relu(self.fc_dec1(z))
        # On applique une sigmoïde pour forcer les valeurs entre 0 et 1 (pixels)
        return torch.sigmoid(self.fc_out(h))

    # ---------------------------
    # Étape 4 : Forward global
    # ---------------------------
    def forward(self, x):
        # Aplatir l’image en un vecteur
        x = x.view(-1, num_features)
        # Encoder pour obtenir μ et log(σ²)
        mu, logvar = self.encode(x)
        # Tirer un échantillon latent z = μ + σ·ε
        z = self.reparameterize(mu, logvar)
        # Reconstruire l’image à partir du latent
        return self.decode(z), mu, logvar


vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(recon_x, x, mu, logvar):
    # ---------------------------
    # 1️ Reconstruction loss
    # ---------------------------
    # On mesure l'erreur entre image originale et reconstruite.
    # On utilise MSE ici, mais souvent BCE est utilisée pour des pixels [0,1].
    recon_loss = F.mse_loss(recon_x, x.view(-1, num_features), reduction="sum")

    # ---------------------------
    # 2️ KL Divergence
    # ---------------------------
    # Encourage la distribution latente q(z|x) à se rapprocher de N(0, I)
    # Formule :  D_KL = -0.5 * Σ(1 + logσ² - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # ---------------------------
    # 3️ Somme des deux pertes
    # ---------------------------
    # On divise par la taille du batch pour normaliser
    return (recon_loss + kl) / x.size(0)


# === Entraînement VAE ===
epochs_vae = 150
for epoch in range(epochs_vae):
    vae.train()
    total_loss = 0
    for imgs, _ in tqdm(train_loader, desc=f"{epoch+1}/{epochs_vae}"):
        imgs = imgs.to(device)
        recon, mu, logvar = vae(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs_vae}] - VAE Loss: {total_loss/len(train_loader):.4f}")

# === Visualisation VAE ===
def plot_reconstructionsVAE(model, loader):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recon, _, _ = model(imgs)  #  on récupère seulement la reconstruction
        recon = recon.cpu().view(-1, 1, 28, 28)
    fig, axes = plt.subplots(2, 10, figsize=(10, 3))
    for i in range(10):
        axes[0, i].imshow(imgs[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    plt.show()

plot_reconstructionsVAE(vae, test_loader)

# Génération du manifold latent (visualisation des échantillons)
def plot_latent_space(model, n=15):
    """
    Génère et affiche une grille d’images décodées à partir de l’espace latent du VAE.
    """
    digit_size = 28
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    figure = np.zeros((digit_size * n, digit_size * n))

    model.eval()  #  le modèle complet, pas juste la fonction decode
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # point dans l’espace latent 2D
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                # génération de l’image à partir du décodeur du VAE
                x_decoded = model.decode(z).cpu().view(28, 28)
                figure[i * digit_size:(i + 1) * digit_size,
                       j * digit_size:(j + 1) * digit_size] = x_decoded

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.title("Manifold latent du VAE (z1, z2)")
    plt.axis("off")
    plt.show()

plot_latent_space(vae)

# ===============================
# 6️ ConvVAE (Convolutionnel)
# ===============================
class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # -----------------------------
        #  ENCODEUR CONVOLUTIONNEL
        # -----------------------------
        # Objectif : réduire progressivement la taille spatiale (28×28 → 7×7)
        # tout en augmentant le nombre de canaux (1 → 64)
        self.enc_conv = nn.Sequential(
            # 1 canal (image MNIST) → 32 filtres
            # stride=2 => réduit la taille 28x28 → 14x14
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            # 32 → 64 filtres
            # stride=2 => 14x14 → 7x7
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # On a maintenant un tenseur de taille [batch, 64, 7, 7]
        self.flatten = nn.Flatten()  # pour passer dans des couches linéaires

        # Deux couches linéaires : elles produisent les paramètres du latent space
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)      # moyenne (μ)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)  # log-variance (log(σ²))

        # -----------------------------
        #  DÉCODEUR
        # -----------------------------
        # But : reconstruire l’image à partir du vecteur latent z
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)  # remonte en espace 7×7×64

        # Reconstruction inverse des convolutions
        self.dec_conv = nn.Sequential(
            # 64 → 64 filtres, upsampling 7x7 → 14x14
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # 64 → 32 filtres, upsampling 14x14 → 28x28
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # Dernière couche : 32 → 1 canal (image finale)
            # Pas de stride, on garde 28x28
            nn.Conv2d(32, 1, 3, padding=1),

            # Sigmoïde pour ramener les pixels entre [0,1]
            nn.Sigmoid()
        )

    # -----------------------------
    #  Étape 1 : Encode
    # -----------------------------
    def encode(self, x):
        # Passe l’image à travers les couches convolutionnelles
        h = self.enc_conv(x)   # → [B, 64, 7, 7]
        h = self.flatten(h)    # → [B, 64*7*7]
        # Produit μ et log(σ²)
        return self.fc_mu(h), self.fc_logvar(h)

    # -----------------------------
    #  Étape 2 : Reparamétrisation
    # -----------------------------
    def reparameterize(self, mu, logvar):
        # Convertit log(σ²) en écart-type : σ = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        # Échantillon aléatoire standard ε ~ N(0, I)
        eps = torch.randn_like(std)
        # Trick de reparamétrisation : z = μ + σ * ε
        # Rend le sampling différentiable
        return mu + eps * std

    # -----------------------------
    #  Étape 3 : Decode
    # -----------------------------
    def decode(self, z):
        # Étend le vecteur latent vers une carte 7x7x64
        h = F.relu(self.fc_dec(z))
        h = h.view(-1, 64, 7, 7)
        # Applique les convolutions transposées pour reconstruire l’image
        return self.dec_conv(h)

    # -----------------------------
    #  Étape 4 : Forward global
    # -----------------------------
    def forward(self, x):
        # Encode l’image en μ et log(σ²)
        mu, logvar = self.encode(x)
        # Échantillonne un vecteur latent z
        z = self.reparameterize(mu, logvar)
        # Décode z pour reconstruire l’image
        return self.decode(z), mu, logvar


conv_vae = ConvVAE().to(device)
optimizer = torch.optim.Adam(conv_vae.parameters(), lr=1e-3)

# === Entraînement ConvVAE ===
for epoch in range(epochs_vae):
    conv_vae.train()
    total_loss = 0
    for imgs, _ in  tqdm(train_loader,desc=f"{epoch+1}/{epochs_vae}"):
        imgs = imgs.to(device)
        recon, mu, logvar = conv_vae(imgs)
        recon_loss = F.mse_loss(recon, imgs, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + kl_loss) / imgs.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs_vae}] - ConvVAE Loss: {total_loss/len(train_loader):.4f}")

plot_reconstructionsVAE(conv_vae, test_loader)
plot_latent_space(conv_vae)
