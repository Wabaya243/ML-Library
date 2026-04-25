import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==========================================
# Normalisation
# ==========================================

def normalize(x):
    if x.max() > 1.5:
        x = x / 255.0
    return x * 2 - 1     # [-1,1]

def denormalize(x):
    return (x + 1) / 2


# ==========================================
# Dataset simple
# ==========================================

class ImageFolderSimple(Dataset):
    def __init__(self, root, size=128):
        self.paths = glob.glob(os.path.join(root, "**/*"), recursive=True)
        self.paths = [p for p in self.paths if p.lower().endswith(("jpg","png","jpeg","webp"))]
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BOX)
        x = torch.from_numpy(np.array(img)).permute(2,0,1).float()
        x = normalize(x)
        return x


# ==========================================
# VectorQuantizer
# ==========================================

class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, dim, beta=0.25):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.beta = beta

        self.embed = nn.Embedding(n_embed, dim)
        self.embed.weight.data.uniform_(-1/n_embed, 1/n_embed)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous()     # BCHW -> BHWC
        z_flat = z.view(-1, C)

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )

        indices = torch.argmin(dist, dim=1)
        z_q = self.embed(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        commit_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        loss = codebook_loss + self.beta * commit_loss

        # Straight-through trick
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss, indices.view(B, H, W)


# ==========================================
# Encoder / Decoder
# ==========================================

class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=128, z_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1), nn.ReLU(),   # 64×64
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU(),  # 32×32
            nn.Conv2d(hidden, z_ch, 3, 1, 1)                # latent
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=3, hidden=128, z_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_ch, hidden, 4, 2, 1), nn.ReLU(),   # 64×64
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.ReLU(), # 128×128
            nn.Conv2d(hidden, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# ==========================================
# VQ-VAE complet
# ==========================================

class VQVAE(nn.Module):
    def __init__(self, z_ch=4, hidden=256, n_embed=1024):
        super().__init__()
        self.encoder = Encoder(3, hidden, z_ch)
        self.quant = VectorQuantizer(n_embed, z_ch)
        self.decoder = Decoder(3, hidden, z_ch)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.quant(z_e)
        x_rec = self.decoder(z_q)
        recon_loss = F.mse_loss(x_rec, x)
        return x_rec, recon_loss + vq_loss, z_q


# ==========================================
# Entraînement
# ==========================================

def train_vqvae(
    data_path="Data",
    size=128,
    batch=52,
    epochs=50,
    lr=2e-4,
    device="cuda"
):
    ds = ImageFolderSimple(data_path, size)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    model = VQVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")

        for img in pbar:
            img = img.to(device)

            x_rec, loss, _ = model(img)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        if (epoch + 1) % 10 == 0:
            # sauvegarde
            os.makedirs("vqvae_ckpt", exist_ok=True)
            torch.save(model.state_dict(), f"vqvae_ckpt/epoch_{epoch+1}.pth")

    print("✔ Entraînement terminé.")
    return model


if __name__ == "__main__":
    train_vqvae(
        data_path="Data/without_mask/",   # dossier contenant tes images
        size=128,
        batch=128,
        epochs=200,         # tu peux monter à 150 si t’es chaud
        lr=2.5e-4
    )
