from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from torchvision.utils import save_image, make_grid
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()

        # Vérifie si les dimensions d’entrée et de sortie sont identiques
        self.same_channels = in_channels == out_channels

        # Indique si on doit utiliser une connexion résiduelle (skip connection)
        self.is_res = is_res

        # Première couche de convolution : 3x3, stride=1, padding=1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),  # Activation non linéaire
        )

        # Deuxième couche de convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si on veut une connexion résiduelle
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            # Si les canaux sont identiques, on additionne directement
            if self.same_channels:
                out = x + x2
            else:
                # Sinon, on utilise une convolution 1x1 pour adapter les dimensions
                shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = shortcut(x) + x2

            # Division par √2 pour stabiliser la variance des activations
            return out / 1.414

        else:
            # Si pas de connexion résiduelle, on applique juste les convolutions
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Récupère le nombre de canaux de sortie
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Modifie le nombre de canaux de sortie
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Liste de couches pour l’upsampling :
        # 1) Transposed convolution (augmente la taille de l’image)
        # 2) Deux blocs convolutionnels résiduels pour affiner les détails
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),  # double la taille
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Création du module séquentiel
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatène le tenseur courant avec le tenseur du "skip connection" venant du bas du U-Net
        x = torch.cat((x, skip), 1)
        # Applique les couches définies
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Bloc de descente :
        # 1) Deux blocs convolutionnels résiduels
        # 2) Un MaxPool pour réduire la taille spatiale de moitié
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        Petit réseau linéaire : il transforme un vecteur d’entrée
        (comme un label, ou un vecteur de temps) en un embedding
        de dimension emb_dim.
        """
        self.input_dim = input_dim
        
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Aplatissement de l’entrée
        x = x.view(-1, self.input_dim)
        # Passage dans le réseau
        return self.model(x)


def unorm(x):
    """Normalisation unité : met les valeurs dans [0, 1]"""
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return (x - xmin) / (xmax - xmin)

def norm_all(store, n_t, n_s):
    """Applique unorm à chaque image sur tous les pas de temps et échantillons"""
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def norm_torch(x_all):
    """Version PyTorch : normalise un batch d’images dans [0, 1]"""
    x = x_all.cpu().numpy()
    xmax = x.max((2,3))
    xmin = x.min((2,3))
    xmax = np.expand_dims(xmax, (2,3))
    xmin = np.expand_dims(xmin, (2,3))
    nstore = (x - xmin) / (xmax - xmin)
    return torch.from_numpy(nstore)

def gen_tst_context(n_cfeat):
    """
    Génère un ensemble de vecteurs de contexte de test.
    Chaque vecteur représente une catégorie (ex : humain, nourriture, sort...).
    """
    vec = torch.tensor([
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0],
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0],
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0],
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0],
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0],
        [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0]
    ])
    return len(vec), vec

def plot_grid(x, n_sample, n_rows, save_dir, w):
    """Affiche une grille d’images et la sauvegarde"""
    ncols = n_sample // n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)
    save_image(grid, save_dir + f"run_image_w{w}.png")
    print('Image sauvegardée dans ' + save_dir + f"run_image_w{w}.png")
    return grid



class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context=False):
        # Chargement des fichiers numpy (.npy)
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"Dimensions des sprites : {self.sprites.shape}")
        print(f"Dimensions des labels : {self.slabels.shape}")

        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
                
    def __len__(self):
        # Retourne le nombre d’images
        return len(self.sprites)
    
    def __getitem__(self, idx):
        # Récupère une image + son label
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # Retourne les formes des données et des labels
        return self.sprites_shape, self.slabel_shape

transform = transforms.Compose([
    transforms.ToTensor(),                # Convertit [0–255] → [0.0–1.0]
    transforms.Normalize((0.5,), (0.5,))  # Normalise dans [-1, 1]
])


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  
        # in_channels : nombre de canaux de l’image d’entrée (ex : 3 pour RGB)
        # n_feat      : nombre de canaux intermédiaires (taille des feature maps)
        # n_cfeat     : taille du vecteur de contexte (ex : nombre de classes)
        # height      : hauteur (et largeur) des images, supposée divisible par 4
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  # hauteur des images (souvent 28x28, 32x32, etc.)

        # === Bloc initial de convolution (entrée du U-Net)
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # === Chemin de descente (encoder) ===
        self.down1 = UnetDown(n_feat, n_feat)        # Réduction 1 (ex: 28 → 14)
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # Réduction 2 (ex: 14 → 7)
        
        # Moyennage pour transformer les features en un petit vecteur latent
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((4)),  # Réduit fortement la taille spatiale
            nn.GELU()           # Activation
        )

        # === Embedding du temps (t) et du contexte (c)
        # Chaque embed est un petit MLP (réseau fully-connected)
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # === Chemin de remontée (decoder) ===
        # Première étape : upsample à partir du "bottleneck"
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4),  # upsample spatialement
            nn.GroupNorm(8, 2 * n_feat),  # normalisation par groupe
            nn.ReLU(),
        )

        # Étapes suivantes : reconstruction progressive
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # === Couche de sortie finale ===
        # Reprojetter vers le même nombre de canaux que l’entrée
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),  # Sortie avec même nombre de canaux que l’entrée
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, in_channels, h, w)  → image d’entrée (ex: bruitée)
        t : (batch, 1)                  → embedding temporel (étape de diffusion)
        c : (batch, n_cfeat)            → vecteur de contexte (labels)
        """

        # === 1. Encodage de l’image ===
        x = self.init_conv(x)         # Bloc initial
        down1 = self.down1(x)         # Première descente (encodeur niveau 1)
        down2 = self.down2(down1)     # Deuxième descente (encodeur niveau 2)

        # Transformation en vecteur latent (bottleneck)
        hiddenvec = self.to_vec(down2)

        # === 2. Si aucun contexte fourni, on met un vecteur nul ===
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # === 3. Calcul des embeddings ===
        # Chaque embedding est mis en forme comme un tenseur 4D (batch, canaux, 1, 1)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # === 4. Décodage (chemin de montée) ===
        # On injecte les embeddings dans le flux d’upsampling
        up1 = self.up0(hiddenvec)                          # première montée
        up2 = self.up1(cemb1 * up1 + temb1, down2)         # fusion embeddings + skip
        up3 = self.up2(cemb2 * up2 + temb2, down1)         # fusion embeddings + skip

        # === 5. Couche de sortie finale ===
        # On concatène la dernière feature map avec la feature d’entrée initiale
        out = self.out(torch.cat((up3, x), 1))
        return out


# Les Hyperparametres

#parametres des diffusion
timesteps = 1000
beta1 = 1e-4
beta2 = 0.02

#parametres du reseau 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_feat = 128 # 64 canaux intermédiaires
n_cfeat = 5 # 5 classes pour le contexte
height = 16 # 28x28 images
save_dir = "Save/"
os.makedirs(save_dir, exist_ok=True)

#parametres d'entrainement
batch_size = 360
num_epochs = 200
lr = 2.8e-3  # learning rate

# Construire un DPPM bruit 
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1 #crée une suite de 1001 valeurs (pour 1000 pas).
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1

#construction du model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

#chargement du dataset
dataset = CustomDataset("Data/sprites.npy", "Data/sprites_labels.npy", transform, null_context=False)
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
optim = torch.optim.Adam(nn_model.parameters(), lr=lr)

#  FONCTION : Ajout de bruit à une image

def perturb_input(x, t, noise):
    """
    Ajoute du bruit à une image x en fonction du pas de temps t.
    C’est la formule standard des modèles de diffusion :
    
        x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * ε
    
    où :
      - x_t  : image bruitée à l’étape t
      - x_0  : image propre originale
      - ε    : bruit aléatoire (gaussien)
      - ab_t : coefficient de bruit cumulé (valeurs entre 0 et 1)
    """
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise


# PHASE D'ENTRAÎNEMENT DU MODÈLE

# On met le modèle en mode entraînement (active dropout, batchnorm, etc.)
nn_model.train()

# Boucle principale sur le nombre d’époques
for ep in range(num_epochs):
    print(f' Époque {ep + 1}/{num_epochs}')
    
    #  Diminution linéaire du taux d’apprentissage au fil des époques
    optim.param_groups[0]['lr'] = lr * (1 - ep / num_epochs)
    
    # tqdm permet d’avoir une barre de progression pendant l’entraînement
    pbar = tqdm(dataloader, mininterval=2)
    
    # Boucle sur chaque batch d’images du dataset
    for x, _ in pbar:
        # Remise à zéro des gradients accumulés
        optim.zero_grad()
        
        # Envoi des données sur le GPU (ou CPU selon device)
        x = x.to(device)
        
        # 1️ GÉNÉRATION DU BRUIT ET DE L'ÉTAPE TEMPORELLE
        
        # Génère un bruit aléatoire de la même taille que l’image
        noise = torch.randn_like(x)
        
        # Choisit un timestep aléatoire pour chaque image du batch
        # t ∈ [1, timesteps]
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        
        # 2️ CRÉATION DE L’IMAGE BRUITÉE
        
        # On perturbe l’image originale avec le bruit et le coefficient ab_t
        x_pert = perturb_input(x, t, noise)
        
        # 3️ PRÉDICTION DU BRUIT PAR LE MODÈLE
        
        # Le modèle reçoit :
        #  - l’image bruitée x_pert
        #  - le timestep normalisé t/timesteps (valeur entre 0 et 1)
        # Il doit prédire le bruit qui a été ajouté.
        pred_noise = nn_model(x_pert, t / timesteps)
        
        # 4️ CALCUL DE LA PERTE (MSE)
        
        # La perte est la différence entre le bruit prédit et le vrai bruit
        # => le modèle apprend à "dénoyer" l’image
        loss = F.mse_loss(pred_noise, noise)
        
        
        # 5️ RÉTROPROPAGATION ET MISE À JOUR DES POIDS
        
        # Calcul du gradient
        loss.backward()
        
        # Mise à jour des poids du modèle avec l’optimiseur
        optim.step()

        # Affiche la valeur de la perte dans la barre de progression
        pbar.set_description(f"loss: {loss.item():.4f}")

    
    # 6️ SAUVEGARDE DU MODÈLE APRÈS CHAQUE ÉPOQUE
    if (ep + 1) % 20 == 0:
        torch.save(nn_model.state_dict(), f"{save_dir}model_epoch{ep+1}.pth")
        print(f"Modèle sauvegardé : {save_dir}model_epoch{ep+1}.pth")

    


# FIN DE L’ENTRAÎNEMENT

print(' Entraînement terminé et modèle sauvegardé !')


def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # 1️ Initialisation : bruit pur N(0, 1)
    samples = torch.randn(n_sample, 3, height, height).to(device)

    intermediate = []  # pour sauvegarder les étapes intermédiaires (visualisation)
    
    # 2️ Boucle de débruitage : on va de t=T à t=1
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # Prépare le tenseur du temps normalisé
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # Crée un bruit aléatoire z sauf pour la dernière étape (i=1)
        z = torch.randn_like(samples) if i > 1 else 0

        # 3️ Le modèle prédit le bruit à cette étape
        eps = nn_model(samples, t)

        # 4️ On retire le bruit prédit et on ajoute un petit bruit z
        samples = denoise_add_noise(samples, i, eps, z)

        # 5️ Sauvegarde des étapes intermédiaires pour animation
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    # Regroupe toutes les étapes en un tableau numpy
    intermediate = np.stack(intermediate)
    return samples, intermediate


def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn, w=None, save=True, show=True):
    """
    Crée et affiche une animation propre des étapes de génération.
    Sauvegarde aussi un GIF utilisable directement.
    """

    ncols = n_sample // nrows

    # Vérifie la dimension et remet les axes au bon ordre
    # Entrée: (frames, samples, 3, H, W)
    sx_gen_store = np.moveaxis(x_gen_store, 2, 4)  # -> (frames, samples, H, W, 3)
    sx_gen_store = np.clip((sx_gen_store - sx_gen_store.min()) / (sx_gen_store.max() - sx_gen_store.min()), 0, 1)

    frames, samples, H, W, C = sx_gen_store.shape
    print(f"[INFO] Animation: {frames} étapes, {samples} images, taille {H}x{W}")

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    axs = axs.flatten() if nrows * ncols > 1 else [axs]

    def animate(i):
        fig.suptitle(f"ÉTAPE {i+1}/{frames}", fontsize=10)
        for idx in range(n_sample):
            axs[idx].imshow(sx_gen_store[i, idx])
            axs[idx].set_xticks([])
            axs[idx].set_yticks([])
        return axs

    ani = FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)

    gif_path = os.path.join(save_dir, f"{fn}_w{w or 'none'}.gif")

    if save:
        ani.save(gif_path, dpi=100, writer=PillowWriter(fps=5))
        print(f" GIF sauvegardé : {gif_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani

# Charger le modèle entraîné (si ce n'est pas déjà fait)
nn_model.load_state_dict(torch.load(f"{save_dir}/model_epoch200.pth", map_location=device))
nn_model.eval()
print(" Modèle chargé avec succès")

# Génération d’un batch de 32 images
samples, intermediate_ddpm = sample_ddpm(32)

# Créer et afficher l’animation proprement
plot_sample(intermediate_ddpm, n_sample=32, nrows=4, save_dir=save_dir, fn="ani_run", w=None, save=True, show=True)
