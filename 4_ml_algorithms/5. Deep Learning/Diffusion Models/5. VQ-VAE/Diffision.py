# ============================================================
# Imports
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import os, math, glob
from dataclasses import dataclass
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Normalisation / dénormalisation
# ============================================================

def normalize(img):
    # img attendu en torch.Tensor [C,H,W] avec valeurs [0..255] ou [0..1]
    if img.max() > 1.5:
        img = img / 255.0
    img = img * 2.0 - 1.0
    return img.float()

def denormalize(img):
    # retourne dans [0,1]
    return (img + 1.0) / 2.0


# ============================================================
# Dataset d’images (lecture simple par glob)
# ============================================================

class ImageFolderSimple(Dataset):
    def __init__(self, root_dir, image_size=128, exts=("jpg","jpeg","png","webp")):
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root_dir, f"**/*.{e}"), recursive=True)
        self.paths.sort()
        self.size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BOX)
        x = torch.from_numpy(np.array(img)).permute(2,0,1)  # [C,H,W] uint8
        x = normalize(x)                                    # [-1,1]
        return x


# ============================================================
# VQ-VAE (encoder → quantizer → decoder)
# ============================================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim         # dimension des vecteurs du codebook
        self.num_embeddings = num_embeddings       # nb d’entrées dans le codebook
        self.beta = beta                           # poids de la "commitment loss"

        self.embedding = nn.Embedding(num_embeddings, embedding_dim) # on crées un codebook = une matrice (num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings) # nitialisation des vecteurs du codebook dans une petite plage autour de zéro.

    def _flatten(self, z):
        # (B,C,H,W) -> (B*H*W, C)
        z = z.permute(0,2,3,1).contiguous()
        return z.view(-1, z.size(-1))
    #Ça devient [B*32*32, 4]
    #Soit un tableau géant de vecteurs 4D à quantifier.

    def quantize(self, z_e):
        # sélectionne l’entrée la plus proche dans le codebook pour chaque vecteur
        flat = self._flatten(z_e)                  # (N, C)
        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        indices = torch.argmin(d, dim=1)           # l’index du codebook le plus proche.
        z_q = self.embedding(indices).view(*z_e.permute(0,2,3,1).shape)
        z_q = z_q.permute(0,3,1,2).contiguous()    # (B,C,H,W)
        return z_q, indices

    def forward(self, z_e):
        # z_e : latent continu (B,C,H,W)
        z_q, indices = self.quantize(z_e)

        # Pertes VQ-VAE (codebook + commitment)
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach()) # Le codebook doit se rapprocher de z_e.
        loss = codebook_loss + self.beta * commitment_loss #Le latent doit s’engager (commit) à rester proche du codebook.

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        # indices reshaped en (B,H,W)
        B, C, H, W = z_e.shape
        return z_q, loss, indices.view(B, H, W)

    @torch.no_grad()
    def quantize_only(self, z_e):
        # Utilitaire: quantification sans pertes (pour le sampling)
        z_q, _ = self.quantize(z_e)
        return z_q
    
    """
    prend un latent continu [B,C,H,W]
    
    le découpe en H×W vecteurs
    
    trouve le vecteur le plus proche dans un codebook
    
    remplace chaque vecteur par celui du codebook
    
    calcule les pertes :
    
    codebook loss (vers codebook)
    
    commitment loss (vers latent)
    
    utilise un straight-through trick pour laisser passer le gradient
    
    renvoie :
    
    z_q : latent quantifié
    
    loss
    
    indices : latents discrets (B, H, W)
        """


class Encoder(nn.Module):
    # encode 128x128x3 -> 32x32x4
    def __init__(self, in_ch=3, hidden=128, latent_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1), nn.ReLU(inplace=True),    # 64x64
            nn.Conv2d(hidden, hidden, 4, 2, 1), nn.ReLU(inplace=True),   # 32x32
            nn.Conv2d(hidden, latent_ch, 3, 1, 1)                         # 32x32x4
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    # decode 32x32x4 -> 128x128x3
    def __init__(self, out_ch=3, hidden=128, latent_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, hidden, 4, 2, 1), nn.ReLU(inplace=True),  # 64x64
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.ReLU(inplace=True),     # 128x128
            nn.Conv2d(hidden, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(self, in_ch=3, hidden=128, latent_ch=4, n_embed=512, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_ch, hidden, latent_ch)
        self.vq = VectorQuantizer(n_embed, latent_ch, beta)
        self.decoder = Decoder(in_ch, hidden, latent_ch)

    def forward(self, x):
        # x : image [-1,1]
        z_e = self.encoder(x)                   # latent continu
        z_q, vq_loss, _ = self.vq(z_e)          # quantifié
        x_rec = self.decoder(z_q)               # reconstruction
        return x_rec, vq_loss, z_q

    @torch.no_grad()
    def encode_continuous(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def decode_from_continuous(self, z_e):
        # décodeur après quantification (comme TF: vq_layer(z_e) puis decoder)
        z_q = self.vq.quantize_only(z_e)
        return self.decoder(z_q)


# ============================================================
# Plan de bruit (DDPM)
# ============================================================

def linear_beta_schedule(timesteps):
    # Valeur initiale du bruit ajouté à chaque étape & valeur finale
    beta_start = 1e-4 #au début : on ajoute presque rien (β = 0.0001)
    beta_end = 2e-2 # à la fin : on ajoute beaucoup (β = 0.02)
    return torch.linspace(beta_start, beta_end, timesteps)

# Définir les valeurs de β (taux de bruit)
def make_ddpm_constants(T, device):
    betas = linear_beta_schedule(T).to(device) #On récupère les β_t.
    alphas = 1. - betas # α_t = quantité d'image qui reste après injection du bruit.
    alphas_cumprod = torch.cumprod(alphas, dim=0)   #combien d’image originale reste après t étapes de bruit ?
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) #Elles servent à sampler le bruit proprement :
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0) #Pour le t=0, on met 1.
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    #on obtiens un gros dictionnaire avec toutes les constantes que DDPM utilise.
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "posterior_variance": posterior_variance
    }

def extract(a, t, x_shape):
    # Sert à extraire, pour un lot (batch) d’images, les bons coefficients a_t correspondant à chaque pas t
    # C’est une opération de broadcasting qui rend le calcul compatible avec les tenseurs d’images.
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))

def q_sample(x_start, t, noise, consts):
    # En gros → elle prend une image propre et la bruite progressivement selon l’étape t.
    sqrt_alpha = extract(consts["sqrt_alphas_cumprod"], t, x_start.shape)
    sqrt_one_minus = extract(consts["sqrt_one_minus_alphas_cumprod"], t, x_start.shape)
    return sqrt_alpha * x_start + sqrt_one_minus * noise


# ============================================================
# Embedding temporel sinusoïdal + blocs U-Net (attention incluse)
# ============================================================

class PositionalEmbeddings(nn.Module):
    """
    Donner au modèle une notion du temps t dans la diffusion.

    Le modèle doit savoir si on est :

    au début (image très bruitée)

    ou à la fin (image presque propre)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        #C’est la même idée que Transformers, mais appliquée au temps.

        # un tenseur (batch, dim) que le U-Net peut utiliser pour "savoir" à quelle étape il travaille
        # Inspiré de Fairseq. Crée des encodages sinusoïdaux pour représenter la position (ou le temps).
        half = self.dim // 2
        emb = math.log(10000.) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

class ResBlock(nn.Module):
    # C'est un bloc ResNet modifié :
    # il intègre une information temporelle (temb) à chaque étape,
    # utilise la Group Normalization (meilleure pour petits batchs),
    # et garde une connexion résiduelle pour stabiliser l'apprentissage.
    def __init__(self, in_ch, out_ch, n_groups, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.gn = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.temb = nn.Linear(t_dim, out_ch)    # injecte le time embedding dans le bloc, en le transformant pour qu’il ait la même dimension que les canaux.
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(x)
        h = h + self.temb(F.silu(temb))[:, :, None, None] #C’est crucial pour que le modèle sache comment débruiter diféremment selon l’étape t.
        h = F.silu(self.gn(h))
        h = self.conv2(h)
        return h + self.skip(x)

class SelfAttention2D(nn.Module):
    # l’auto-attention appliquée à des images
    # Parce que le U-Net doit comprendre
    # quelles parties de l’image dépendent des autres (forme globale, symétrie, structure)

    def __init__(self, ch, heads=4, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Conv2d(ch, inner * 3, 1, 1, 0)
        self.proj = nn.Conv2d(inner, ch, 1, 1, 0)
        self.norm = nn.GroupNorm(1, ch)
        self.scale = dim_head ** -0.5

    def forward(self, x):
        
        b, c, h, w = x.shape
        x_ = self.norm(x)
        qkv = self.to_qkv(x_).chunk(3, dim=1)
        q, k, v = [t.view(b, self.heads, -1, h*w) for t in qkv]  # # On reshape pour que chaque pixel puisse "regarder" les autres
        attn = (q.transpose(-2,-1) @ k) * self.scale             # (b,h,hw,hw)
        attn = attn.softmax(dim=-1) #Chaque pixel apprend où regarder dans l’image.
        out = (attn @ v.transpose(-2,-1)).transpose(-2,-1)       # On applique l’attention à V
        out = out.contiguous().view(b, -1, h, w) #Projection finale + skip
        return self.proj(out) + x #On récupère la même taille qu’en entrée.


class UNet(nn.Module):
    """
    U-Net avec blocs résiduels + embeddings temporels + self-attention.
     Version PLATE (pas de downsample/upsample) pour garder la résolution 32×32 identique
    → indispensable pour que le bruit prédit ait la même forme que le latent VQ-VAE (4×32×32).
    """

    def __init__(self, in_ch=4, base=64, n_groups=8, n_res=2, attn_heads=4, attn_dim=256, H=32):
        super().__init__()

        # EMBEDDING TEMPOREL
        # encode t → vecteur riche (sinus/cosinus + 2 MLP)
        t_dim = H * 16  
        self.time_mlp = nn.Sequential(
            PositionalEmbeddings(t_dim),   
            nn.Linear(t_dim, t_dim),
            nn.GELU(),
            nn.Linear(t_dim, t_dim),
        )

        # Projection initiale du latent dans l’espace du U-Net
        self.in_conv = nn.Conv2d(in_ch, base, 3, 1, 1)

        # Niveaux de canaux à chaque étage du U-Net
        # Aucun changement spatial (on reste en 32×32)
        chs = [base, 2*base, 4*base]  
        in_c = base

        # ENCODER (sans réduction de résolution)
        # Chaque niveau = plusieurs ResBlocks + SelfAttention (sur niveaux profonds)
        self.down_blocks = nn.ModuleList()
        for ci in chs:
            blocks = []
            for _ in range(n_res):
                # Bloc résiduel + injection du time embedding
                blocks.append(ResBlock(in_c, ci, n_groups, t_dim))
                in_c = ci

                # Attention uniquement pour les niveaux riches en canaux
                if ci >= 2*base:
                    blocks.append(SelfAttention2D(ci, heads=attn_heads, dim_head=attn_dim // attn_heads))

            self.down_blocks.append(nn.ModuleList(blocks))

        # BOTTLE NECK
        # Zone centrale : 1 ResBlock + 1 Attention + 1 ResBlock
        self.mid = nn.ModuleList([
            ResBlock(in_c, in_c, n_groups, t_dim),
            SelfAttention2D(in_c, heads=attn_heads, dim_head=attn_dim//attn_heads),
            ResBlock(in_c, in_c, n_groups, t_dim),
        ])

        # DECODER (sans upsample)
        # On concatène avec les skip connections (donc in_c + ci)
        self.up_blocks = nn.ModuleList()
        ups = chs[::-1]  # ordre inverse

        for ci in ups:
            blocks = []
            in_cat = in_c + ci  # concat skip + courant

            for _ in range(n_res):
                blocks.append(ResBlock(in_cat, ci, n_groups, t_dim))
                in_cat = ci

                if ci >= 2*base:
                    blocks.append(SelfAttention2D(ci, heads=attn_heads, dim_head=attn_dim // attn_heads))

            self.up_blocks.append(nn.ModuleList(blocks))
            in_c = ci  # mise à jour du nb de canaux

        # Dernière convolution : prédit le bruit ε au même format que le latent
        self.out_conv = nn.Conv2d(in_c, in_ch, 3, 1, 1)

    # FORWARD
    def forward(self, x, t):
        # Time-embedding
        temb = self.time_mlp(t)

        # Première projection
        h = self.in_conv(x)

        # Skip connections
        skips = []

        # ENCODER
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, temb)
                else:
                    h = layer(h)
            skips.append(h)

        # Bottleneck
        for layer in self.mid:
            if isinstance(layer, ResBlock):
                h = layer(h, temb)
            else:
                h = layer(h)

        # DECODER
        for block in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)  # concat skip

            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, temb)
                else:
                    h = layer(h)

        # Prédiction finale du bruit ε̂
        return self.out_conv(h)



# ============================================================
# DDPM / DDIM sampling
# ============================================================

@torch.no_grad()
def p_sample(model, x, t, consts):
    # C’est la version “classique” du sampling selon le papier DDPM.
    # Elle est stochastique (elle rajoute du bruit à chaque étape).
    # Extraction des coefficients (spécifiques à chaque t)
    betas_t = extract(consts["betas"], t, x.shape)
    sqrt_one_minus = extract(consts["sqrt_one_minus_alphas_cumprod"], t, x.shape)
    sqrt_recip_alpha = extract(consts["sqrt_recip_alphas"], t, x.shape)
    # Calcul de la moyenne prédite (modèle de diffusion DDPM)
    pred_noise = model(x, t)
    model_mean = sqrt_recip_alpha * (x - betas_t * pred_noise / sqrt_one_minus)
    # bruit pour la stochasticité (sauf à t=0)
    noise = torch.randn_like(x)
    nonzero_mask = (t != 0).float().view(-1, *([1]*(x.ndim-1)))
    return model_mean + nonzero_mask * torch.sqrt(extract(consts["posterior_variance"], t, x.shape)) * noise


@torch.no_grad()
def ddim_step(model, x_t, t, t_prev, consts, eta=0.0):
    # DDIM c’est le fast & deterministic sampler.
    # Il génère des images plus vite et sans hasard si eta = 0
    # Extraction des coefficients pour les étapes t et t_prev
    alpha_bar_t = extract(consts["alphas_cumprod"], t, x_t.shape)
    alpha_bar_prev = extract(consts["alphas_cumprod"], t_prev, x_t.shape)

    # Bruit prédit par le modèle
    eps = model(x_t, t)

    # Reconstruction estimée de l’image originale x₀
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

    # Calcul du sigma (écart-type de bruit ajouté, dépend de η)
    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)

    # Moyenne de la distribution pour x_{t-1}
    mean = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps

    # On ajoute du bruit aléatoire si eta > 0
    noise = torch.randn_like(x_t)
    x_prev = mean + sigma_t * noise
    return x_prev



@torch.no_grad()
def sample_and_plot(denoise_model, vqvae, shape, consts, num_train_timesteps=1000, num_inference_steps=20, ddpm=False, eta=0.0, num_checkpoints=8, epoch=0):
    # On part d’un bruit pur
    x = torch.randn(shape, device=next(denoise_model.parameters()).device)

    # Séquence d’étapes de débruitage (du bruit max → image propre)
    timesteps = np.linspace(num_train_timesteps - 1, 0, num_inference_steps, dtype=int)

    # Choix des checkpoints intermédiaires à sauvegarder
    checkpoint_indices = np.linspace(0, len(timesteps)-1, num_checkpoints, dtype=int).tolist()

    latents_ckpts, images_ckpts = [], []

    # Boucle de débruitage
    for i, t_int in enumerate(timesteps):
        t = torch.full((shape[0],), int(t_int), device=x.device, dtype=torch.long)

        # DDPM (aléatoire) ou DDIM (déterministe)
        if ddpm:
            x = p_sample(denoise_model, x, t, consts)
        else:
            t_prev_val = int(timesteps[i+1] if i < len(timesteps)-1 else 0)
            t_prev = torch.full((shape[0],), t_prev_val, device=x.device, dtype=torch.long)
            x = ddim_step(denoise_model, x, t, t_prev, consts, eta=eta)

        # Sauvegarde des latents et images reconstruites à certains intervalles
        if i in checkpoint_indices:
            latents_ckpts.append((t_int, x.detach().cpu()))
            # déquantification + décodeur (VQ-VAE)
            recon = vqvae.decode_from_continuous(x).detach().cpu()
            images_ckpts.append((t_int, recon))

        if t_int % 100 == 0:
            print(f"t={t_int}")

    # --- Visualisation des checkpoints ---
    n_ckpts = len(images_ckpts)
    fig, axes = plt.subplots(2, n_ckpts, figsize=(2.5 * n_ckpts, 6))
    axes = np.atleast_2d(axes)

    # Affichage des latents (pseudo-RGB)
    for col, (t_idx, latent) in enumerate(latents_ckpts):
        # moyenne sur canaux pour visualiser
        latent_map = latent[0].mean(0)
        latent_map = (latent_map - latent_map.min()) / (latent_map.max() - latent_map.min() + 1e-8)
        axes[0, col].imshow(latent_map, cmap='viridis')
        axes[0, col].set_title(f"Latent t={t_idx}")
        axes[0, col].axis("off")

    # Affichage des images décodées
    for col, (t_idx, img) in enumerate(images_ckpts):
        img_display = denormalize(img[0]).clamp(0,1).permute(1,2,0).numpy()
        axes[1, col].imshow(img_display)
        axes[1, col].set_title(f"Decoded t={t_idx}")
        axes[1, col].axis("off")

    plt.suptitle(f"Latent vs Decoded Denoising Checkpoints, Size:{images_ckpts[0][1][0].shape}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"debug_samples/epoch_{epoch}.png")
    plt.close()


# ============================================================
# Entraînement (mono-GPU) avec EMA
# ============================================================

@dataclass
class CFG:
    BATCH_SIZE_PER_REPLICA: int = 64
    TIME_STEPS: int = 1000
    IM_LATENT_SHAPE = (4, 32, 32)   # Forme de l'image latente (C,H,W)
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 400
    EMA: float = 0.999 #Plus EMA est proche de 1 → modèle plus stable, moins réactif.
    IMAGE_SIZE: int = 128
    TRAIN_PATH: str = "Data/without_mask/"        # dossier d’images
    PRECOMPUTE_LATENTS: bool = True # pré-calculer les latents VQ-VAE

os.makedirs("debug_samples", exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CFG()

    # --- Charger / initialiser VQ-VAE ---
    vqvae = VQVAE(in_ch=3, hidden=128, latent_ch=cfg.IM_LATENT_SHAPE[0], n_embed=512, beta=0.25).to(device)
    state = torch.load("vqvae_ckpt/epoch_100.pth", map_location=device)
    new_state = {}
    
    for k, v in state.items():
        k2 = k.replace("quant.", "vq.").replace("embed.", "embedding.")
        new_state[k2] = v
    
    vqvae.load_state_dict(new_state, strict=False)
    vqvae.eval()    # gelé comme dans le TF (tu peux charger un checkpoint si tu en as un)
    for p in vqvae.parameters():
        p.requires_grad = False

    # --- Encoder continu pour extraire les latents (pré-quantification) ---
    @torch.no_grad()
    def encode_batch(x):
        return vqvae.encode_continuous(x)

    # --- Dataset d’images ---
    ds = ImageFolderSimple(cfg.TRAIN_PATH, image_size=cfg.IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE_PER_REPLICA, shuffle=False, num_workers=0, pin_memory=False)

  
    # --- Pré-calcul de tous les latents OU chargement depuis cache ---
    if cfg.PRECOMPUTE_LATENTS:
        if not os.path.exists("latents_cache/latents.pt"):
            print("Pre-computing latents...")
            all_latents = []
            with torch.no_grad():
                for batch in tqdm(dl):
                    batch = batch.to(device)
                    z = encode_batch(batch)
                    all_latents.append(z.cpu())
            all_latents = torch.cat(all_latents, dim=0)
            print(f"Latents computed: {tuple(all_latents.shape)}")

            os.makedirs("latents_cache", exist_ok=True)
            torch.save(all_latents, "latents_cache/latents.pt")
            print("Saved latents to latents_cache/latents.pt")
        else:
            print("Loading cached latents...")
            all_latents = torch.load("latents_cache/latents.pt", map_location="cpu")
            print(f"Loaded: {tuple(all_latents.shape)}")

        # Dataset final (latents pré-calculés)
        class LatentDataset(Dataset):
            def __init__(self, z): self.z = z
            def __len__(self): return self.z.size(0)
            def __getitem__(self, i): return self.z[i]

        train_ds = LatentDataset(all_latents)
        train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE_PER_REPLICA,
                            shuffle=True, num_workers=0, pin_memory=False)

    else:
        # encode on-the-fly (rarement conseillé, plus lent)
        class WrappedDS(Dataset):
            def __init__(self, image_ds): self.image_ds = image_ds
            def __len__(self): return len(self.image_ds)
            def __getitem__(self, i):
                x = self.image_ds[i]
                with torch.no_grad():
                    z = encode_batch(x.unsqueeze(0).to(device)).squeeze(0).cpu()
                return z

        train_dl = DataLoader(WrappedDS(ds), batch_size=cfg.BATCH_SIZE_PER_REPLICA,
                            shuffle=True, num_workers=0, pin_memory=False)

                            
    # --- DDPM constantes ---
    consts = make_ddpm_constants(cfg.TIME_STEPS, device)

    # --- Modèle de débruitage + EMA ---
    denoise = UNet(in_ch=cfg.IM_LATENT_SHAPE[0], base=32, n_groups=8, n_res=1,
                   attn_heads=4, attn_dim=128, H=cfg.IM_LATENT_SHAPE[1]).to(device)
    
    ema_denoise = UNet(in_ch=cfg.IM_LATENT_SHAPE[0], base=32, n_groups=8, n_res=1,
                       attn_heads=4, attn_dim=128, H=cfg.IM_LATENT_SHAPE[1]).to(device) # Ça stabilise énormément l’échantillonnage.
    
    ema_denoise.load_state_dict(denoise.state_dict())
    optimizer = torch.optim.Adam(denoise.parameters(), lr=cfg.LEARNING_RATE)

    def custom_loss(denoise_model, z, t):
        # compare le bruit prédit par le modèle au bruit réel injecté (Huber)
        noise = torch.randn_like(z)
        z_noisy = q_sample(z, t, noise, consts)
        pred = denoise_model(z_noisy, t)
        return F.smooth_l1_loss(pred, noise)  # Huber

    scaler = GradScaler() # pour AMP
    update_ema_every = 2
    global_steps = 0

    torch.cuda.empty_cache()

    # --- Entraînement ---
    EPOCHS = cfg.EPOCHS
    os.makedirs("save", exist_ok=True)

    for epoch in range(EPOCHS):
        denoise.train()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for batch in pbar:
            global_steps += 1

            z = batch.to(device)  # (B,4,32,32)
            t = torch.randint(0, cfg.TIME_STEPS, (z.size(0),), device=device).long()

            with autocast():  #  AMP 
                noise = torch.randn_like(z)
                z_noisy = q_sample(z, t, noise, consts)
                pred = denoise(z_noisy, t)
                loss = F.smooth_l1_loss(pred, noise)

            # Backward AMP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- EMA toutes les 2 itérations ---
            # Mise à jour des poids EMA (Exponential Moving Average)
            if global_steps % update_ema_every == 0:
                with torch.no_grad():
                    for p, pe in zip(denoise.parameters(), ema_denoise.parameters()):
                        pe.data.mul_(cfg.EMA).add_(p.data, alpha=1 - cfg.EMA)

            pbar.set_postfix({"Huber_loss": f"{loss.item():.4f}"})
        
        # sauvegarde tous les 20 epoch
        if (epoch + 1) % 25 == 0:
            torch.save(ema_denoise.state_dict(), f"save/ema_denoise_{epoch+1}.pth")
            torch.save(denoise.state_dict(), f"save/denoise_{epoch+1}.pth")


        # Sampling régulier
        if (epoch % 20) == 0:
            sample_and_plot(
                ema_denoise,              # modèle lissé EMA
                vqvae,                    # décodeur VQ-VAE
                (1, *cfg.IM_LATENT_SHAPE), # En clair : on génère une image en partant d’un bruit latent 4×32×32.
                consts, # Toutes les constantes DDPM pré-calculées :
                num_train_timesteps=cfg.TIME_STEPS,
                num_inference_steps=20,  # Nombre d’étapes DDIM pour générer une image.
                ddpm=False,               # False → DDIM (déterministe si eta=0)
                eta=0.0,
                epoch=epoch,
                num_checkpoints=6 # Pendant le sampling, on sauvegardes 5 images intermédiaires.
            )

    # Échantillonnage final (modèle courant)
    sample_and_plot(
        denoise,
        vqvae,
        (1, *cfg.IM_LATENT_SHAPE),
        consts,
        num_train_timesteps=cfg.TIME_STEPS,
        num_inference_steps=20,
        ddpm=False,
        epoch="final",
        eta=0.0, # eta=0 = DDIM déterministe
        num_checkpoints=6 # Pendant le sampling, on sauvegardes 6 images intermédiaires.
    )

if __name__ == "__main__":
    main()
