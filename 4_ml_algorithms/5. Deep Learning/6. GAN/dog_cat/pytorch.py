# dcgan_clean.py
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

# ---------------------------
# Configuration / hyperparams
# ---------------------------
DATA_PATH = '../input/all-dogs/all-dogs/'  # <-- mets ici ton chemin réel
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 5        # ajuste pour entraînement réel
NZ = 128              # dimension du bruit latent
NGF = 64              # taille de base du générateur (channels factor)
NDF = 64              # taille de base du discriminateur (channels factor)
LR_G = 2e-4
LR_D = 2e-4
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REAL_LABEL = 0.9      # label smoothing pour les vrais exemples
FAKE_LABEL = 0.0

# ---------------------------
# Transform & DataLoader
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
])

dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

print(f"Dataset loaded: {len(dataset)} images. Device: {DEVICE}")

# ---------------------------
# Initialisation des poids
# ---------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# ---------------------------
# Generator (DCGAN style)
# input: (batch, NZ, 1, 1) -> output: (batch, 3, 64, 64)
# ---------------------------
class Generator(nn.Module):
    def __init__(self, nz=NZ, ngf=NGF, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),      # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, channels, kernel_size=4, stride=2, padding=1, bias=False),     # 64x64
            nn.Tanh()  # sortie dans [-1,1]
        )

    def forward(self, z):
        return self.model(z)


# ---------------------------
# Discriminator (DCGAN style)
# input: (batch, 3, 64, 64) -> output: logits (batch, 1)
# NOTE: on retourne les logits (pas de sigmoid) et on utilisera
#       BCEWithLogitsLoss qui intègre la sigmoid de façon stable.
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, ndf=NDF, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (channels) x 64 x 64
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),   # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     # 16x16
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), # 4x4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)       # 1x1
            # logits output, shape (batch,1,1,1)
        )

    def forward(self, img):
        out = self.model(img)
        return out.view(-1, 1)  # (batch,1)


# ---------------------------
# Initialisation & optimizers & loss
# ---------------------------
netG = Generator(nz=NZ, ngf=NGF, channels=3).to(DEVICE)
netD = Discriminator(ndf=NDF, channels=3).to(DEVICE)

netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

criterion = nn.BCEWithLogitsLoss()  # stable (logits in input)

optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))

fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)  # pour visualiser à chaque epoch

print("Params D:", sum(p.numel() for p in netD.parameters()))
print("Params G:", sum(p.numel() for p in netG.parameters()))

# ---------------------------
# Training loop
# ---------------------------
iters = 0
SAVE_EVERY = max(1, len(dataloader) // 4)  # fréquence de sauvegarde d'images

G_losses = []
D_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    for i, (real_images, _) in enumerate(loop):
        netD.zero_grad()
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        # --- Train Discriminator on real ---
        labels_real = torch.full((batch_size, 1), REAL_LABEL, device=DEVICE)
        output_real = netD(real_images)                     # logits
        lossD_real = criterion(output_real, labels_real)

        # --- Train Discriminator on fake ---
        noise = torch.randn(batch_size, NZ, 1, 1, device=DEVICE)
        fake_images = netG(noise)
        labels_fake = torch.full((batch_size, 1), FAKE_LABEL, device=DEVICE)
        output_fake = netD(fake_images.detach())            # detach pour ne pas backprop dans G
        lossD_fake = criterion(output_fake, labels_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # --- Train Generator ---
        netG.zero_grad()
        # On veut que le discriminateur classe les fake comme réels
        labels_for_G = torch.full((batch_size, 1), REAL_LABEL, device=DEVICE)
        output_fake_forG = netD(fake_images)   # logits (cela backprop dans G)
        lossG = criterion(output_fake_forG, labels_for_G)
        lossG.backward()
        optimizerG.step()

        # logging
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())
        iters += 1

        if iters % 50 == 0:
            loop.set_postfix({'lossD': lossD.item(), 'lossG': lossG.item()})

        # sauvegarder quelques images
        if iters % SAVE_EVERY == 0:
            with torch.no_grad():
                fake_grid = netG(fixed_noise).detach().cpu()
                # dé-normaliser de [-1,1] -> [0,1]
                img_grid = (fake_grid + 1.0) / 2.0
                save_image(img_grid, os.path.join(OUTPUT_DIR, f"fake_samples_iter_{iters:06d}.png"), nrow=8, normalize=False)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} done in {epoch_time:.1f}s | lossD: {np.mean(D_losses[-len(dataloader):]):.4f} lossG: {np.mean(G_losses[-len(dataloader):]):.4f}")

    # sauvegarde modèle par epoch
    torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, f'netG_epoch_{epoch+1}.pth'))
    torch.save(netD.state_dict(), os.path.join(OUTPUT_DIR, f'netD_epoch_{epoch+1}.pth'))

# ---------------------------
# Sauvegarde finale et affichage pertes
# ---------------------------
torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, 'generator_final.pth'))
torch.save(netD.state_dict(), os.path.join(OUTPUT_DIR, 'discriminator_final.pth'))

# plot losses
plt.figure(figsize=(10,5))
plt.plot(G_losses, label='G loss')
plt.plot(D_losses, label='D loss')
plt.legend()
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('Training losses')
plt.show()


