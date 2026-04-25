# ============================================================
# IMPORTS
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook as tqdm

# ============================================================
# DÉFINITION DU GÉNÉRATEUR
# ============================================================
class Generator(nn.Module):
    """
    Générateur du GAN : prend un vecteur latent (bruit)
    et produit une image synthétique (ex. 3 canaux RGB).
    """
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.channels = channels

        # Fonction utilitaire : bloc convolution transposée + BatchNorm + ReLU
        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            return [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True)
            ]

        # Architecture principale du générateur
        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0),  # Bruit → features
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh()  # sortie normalisée entre [-1, 1]
        )

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        return self.model(z)

# ============================================================
# DÉFINITION DU DISCRIMINATEUR
# ============================================================
class Discriminator(nn.Module):
    """
    Discriminateur : reçoit une image (réelle ou générée)
    et renvoie une probabilité d'authenticité.
    """
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        # Fonction utilitaire : bloc convolution + BatchNorm (optionnel) + LeakyReLU
        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        # Architecture principale du discriminateur
        self.model = nn.Sequential(
            *convlayer(channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )

    def forward(self, imgs):
        logits = self.model(imgs)
        out = torch.sigmoid(logits)
        return out.view(-1, 1)

# ============================================================
# HYPERPARAMÈTRES
# ============================================================
batch_size = 32
LR_G = 0.0005          # Taux d'apprentissage du générateur
LR_D = 0.0001          # Taux d'apprentissage du discriminateur
beta1 = 0.5            # Paramètre β1 de l’optimiseur Adam
epochs = 100           # Nombre total d’époques
real_label = 0.9       # Label pour les images réelles (smoothing)
fake_label = 0         # Label pour les images fausses
nz = 128               # Taille du vecteur latent

# Détection automatique du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ============================================================
# INITIALISATION DES RÉSEAUX ET OPTIMISEURS
# ============================================================
netG = Generator(nz).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(beta1, 0.999))

# Bruit fixe pour observer la progression du générateur
fixed_noise = torch.randn(25, nz, 1, 1, device=device)

# ============================================================
# OUTILS DE SUIVI
# ============================================================
G_losses = []
D_losses = []
epoch_time = []

def plot_loss(G_losses, D_losses, epoch):
    """Affiche les courbes de perte du générateur et du discriminateur."""
    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator Loss - Epoch {epoch}")
    plt.plot(G_losses, label="Generator (G)")
    plt.plot(D_losses, label="Discriminator (D)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_generated_img(n_images=5):
    """Affiche un ensemble d’images synthétiques générées par G."""
    samples = []
    for _ in range(n_images):
        noise = torch.randn(1, nz, 1, 1, device=device)
        gen_image = netG(noise).detach().cpu().squeeze(0)
        gen_image = gen_image.numpy().transpose(1, 2, 0)
        samples.append(gen_image)
    
    fig, axes = plt.subplots(1, n_images, figsize=(15, 15))
    for idx, ax in enumerate(axes):
        ax.axis("off")
        ax.imshow((samples[idx] + 1) / 2)  # remise dans [0,1] pour affichage
    plt.show()


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # pour sortir dans [-1, 1]
])

# Chargement du dataset (ex: dossier 'without_mask')
train_dataset = datasets.ImageFolder(root="Data/without_mask", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ============================================================
# BOUCLE D’ENTRAÎNEMENT
# ============================================================
for epoch in range(epochs):
    start = time.time()
    
    for ii, (real_images, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # ------------------------------------------------------------
        # (1) Mise à jour du Discriminateur : maximise log(D(x)) + log(1 - D(G(z)))
        # ------------------------------------------------------------
        netD.zero_grad()
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Pertes sur vraies images
        labels = torch.full((batch_size, 1), real_label, device=device)
        output = netD(real_images)
        errD_real = criterion(output, labels)
        errD_real.backward()
        D_x = output.mean().item()

        # Pertes sur fausses images générées
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Mise à jour de D
        errD = errD_real + errD_fake
        optimizerD.step()

        # ------------------------------------------------------------
        # (2) Mise à jour du Générateur : maximise log(D(G(z)))
        # ------------------------------------------------------------
        netG.zero_grad()
        labels.fill_(real_label)  # le générateur veut que D croie que ses images sont réelles
        output = netD(fake)
        errG = criterion(output, labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Sauvegarde des pertes
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Log toutes les demi-époques
        if (ii + 1) % (len(train_loader) // 2) == 0:
            print(f"[{epoch+1}/{epochs}] Step {ii+1}/{len(train_loader)} "
                  f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} "
                  f"| D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

    # Fin d’époque : affichage et suivi
    plot_loss(G_losses, D_losses, epoch)
    G_losses.clear()
    D_losses.clear()

    if epoch % 10 == 0:
        show_generated_img()

    epoch_time.append(time.time() - start)

print("Entraînement terminé.")
