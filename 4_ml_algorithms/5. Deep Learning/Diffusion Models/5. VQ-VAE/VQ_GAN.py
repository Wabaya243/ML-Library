import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# LPIPS perceptual loss (important pour VQ-GAN)
# pip install lpips
import lpips
lpips_fn = lpips.LPIPS(net="vgg").cuda()


# ============================================================
# Normalisation
# ============================================================

def normalize(x):
    if x.max() > 1.5:
        x = x / 255.0
    return x * 2 - 1

def denormalize(x):
    return (x + 1) / 2


# ============================================================
# Dataset simple
# ============================================================

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
        return normalize(x)


# ============================================================
# Vector Quantizer
# ============================================================

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
        z = z_e.permute(0,2,3,1).contiguous()
        z_flat = z.view(-1, C)

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )

        indices = torch.argmin(dist, dim=1)
        z_q = self.embed(indices).view(B, H, W, C).permute(0,3,1,2)

        commit = F.mse_loss(z_q.detach(), z_e)
        codebook = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook + self.beta * commit

        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss, indices.view(B,H,W)


# ============================================================
# Encoder / Decoder (identiques au VQ-VAE)
# ============================================================

class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=128, z_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, z_ch, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=3, hidden=128, z_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_ch, hidden, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# ============================================================
# PatchGAN Discriminator (petit, efficace)
# ============================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch*4, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# VQ-GAN (encode → quantize → decode + GAN losses)
# ============================================================

class VQGAN(nn.Module):
    def __init__(self, z_ch=4, hidden=256, n_embed=1024):
        super().__init__()
        self.encoder = Encoder(3, hidden, z_ch)
        self.quant = VectorQuantizer(n_embed, z_ch)
        self.decoder = Decoder(3, hidden, z_ch)
        self.disc = PatchDiscriminator()

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.quant(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss


# ============================================================
# Entraînement VQ-GAN
# ============================================================

def train_vqgan(
    data_path="Data",
    size=128,
    batch=32,
    epochs=100,
    lr=2e-4,
    device="cuda"
):

    ds = ImageFolderSimple(data_path, size)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    model = VQGAN().to(device)
    disc = model.disc

    opt_g = torch.optim.Adam(list(model.encoder.parameters())
                             + list(model.quant.parameters())
                             + list(model.decoder.parameters()), lr=lr)

    opt_d = torch.optim.Adam(disc.parameters(), lr=lr)

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")

        for img in pbar:
            img = img.to(device)

            # ------------------------------
            #   1) GENERATOR STEP
            # ------------------------------
            x_rec, vq_loss = model(img)

            # reconstruction L1
            recon_loss = F.l1_loss(x_rec, img)

            # perceptual loss LPIPS
            perc_loss = lpips_fn(x_rec, img).mean()

            # adversarial generator loss
            g_adv = -disc(x_rec).mean()

            loss_g = recon_loss + 0.1 * perc_loss + 0.1 * g_adv + vq_loss

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # ------------------------------
            #   2) DISCRIMINATOR STEP
            # ------------------------------
            real_logits = disc(img)
            fake_logits = disc(x_rec.detach())

            d_loss = (
                F.relu(1 - real_logits).mean() +
                F.relu(1 + fake_logits).mean()
            )

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            pbar.set_postfix({
                "G_loss": f"{loss_g.item():.4f}",
                "D_loss": f"{d_loss.item():.4f}"
            })

        # save
        if (epoch + 1) % 10 == 0:
            os.makedirs("vqgan_ckpt", exist_ok=True)
            torch.save(model.state_dict(), f"vqgan_ckpt/epoch_{epoch+1}.pth")

    print("✔ Entraînement terminé.")
    return model


if __name__ == "__main__":
    train_vqgan(
        data_path="Data/without_mask/",
        size=128,
        batch=32,
        epochs=150,
        lr=2e-4
    )
