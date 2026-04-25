# ===========================================================
# DIFFUSION CONDITIONNELLE SUR LES CLASSES (fleurs-102)
# ===========================================================

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from datasets import load_dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
import pandas as pd
from datasets import Dataset
from diffusers import DDPMScheduler, UNet2DConditionModel, DDPMPipeline
from torch.utils.data import Dataset as TorchDataset
from diffusers.utils import make_image_grid

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ===========================================================
# Chargement du dataset
# ===========================================================
# chemins
ROOT = "data"
IMG_DIR = os.path.join(ROOT, "102flowers")
LABEL_FILE = os.path.join(ROOT, "imagelabels.mat")

# Charger les labels
mat = scipy.io.loadmat(LABEL_FILE)
labels = mat["labels"][0] - 1  # on passe de 1–102 à 0–101

#construire une liste d'image
paths = [
    os.path.join(IMG_DIR, f"image_{i:05d}.jpg") 
    for i in range(1, len(labels) + 1)
] 

#verification assez rapide
assert len(paths) == len(labels), "Mismatching Label/images"

#creation du datafram
df = pd.DataFrame({
    "image" : paths,
    "label" : labels,
})

# === split 50% ===
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

image_size=64

class FlowersDataset(TorchDataset):
    def __init__(self, df, image_size=64):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image"]
        label = row["label"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "images": img,
            "labels": torch.tensor(label, dtype=torch.long)
        }


dataset = FlowersDataset(df)

print(dataset)
print(dataset[0])


batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================
# Scheduler & U-Net conditionnel
# ===========================================================
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)

model = UNet2DConditionModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    cross_attention_dim=128,               # taille des embeddings de classe
    # Cette ligne active le “conditionnement”.
    # Le modèle devient capable de recevoir un vecteur externe (embedding) à chaque étape du débruitage.
    attention_head_dim=8, #éfinit la taille des “têtes d’attention” internes du modèle.
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
).to(device)

model.enable_xformers_memory_efficient_attention()

# Embedding pour 102 classes
num_classes = 102
class_embed = torch.nn.Embedding(num_classes, 128).to(device)
# On ajoute le vecteur embedding à chaque étape de débruitage.
# Chaque “fleur” a donc sa représentation vectorielle apprise pendant l’entraînement.
# C’est comme si le modèle apprend à “conditionner” son débruitage sur la classe de l’image.

optimizer = torch.optim.AdamW(list(model.parameters()) + list(class_embed.parameters()), lr=1e-4)
#Tu dis à l’optimiseur d’entraîner à la fois : les poids du U-Net (model.parameters()), et les vecteurs de classe (class_embed.parameters()).
scaler = torch.cuda.amp.GradScaler()

# ===========================================================
# Entraînement
# ===========================================================
num_epochs = 200

for epoch in range(num_epochs):
    model.train()                                     
    total_loss = 0                                     # reset de la loss cumulée pour l'affichage

    for step, batch in enumerate(tqdm(train_loader, desc=f"Époque {epoch+1}/{num_epochs}")):
        clean_images = batch["images"].to(device)      # batch d’images propres sur GPU
        labels = batch["labels"].to(device)            # labels (0–101) sur GPU

        noise = torch.randn_like(clean_images)         # bruit gaussien de même shape que les images
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (clean_images.shape[0],), device=device).long()  
                                                       # tirage aléatoire d’un timestep pour chaque image

        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                                                       # ajoute le bruit selon la courbe du scheduler (forward diffusion)

        # Embedding de classe (condition)
        cond_emb = class_embed(labels)                 # vecteur [B, 128] : embedding de classe
        cond_emb = cond_emb.unsqueeze(1)               # reshape -> [B, 1, 128] pour Cross-Attention

        # forward pass en float16 (AMP)
        with torch.cuda.amp.autocast():
            pred = model(
                noisy_images,                          # entrée bruitée
                timesteps,                             # étape t
                encoder_hidden_states=cond_emb         # embedding de classe injecté via cross-attention
            ).sample                                   # prédiction du bruit
            loss = torch.nn.functional.mse_loss(pred, noise)
                                                       # objectif DDPM : prédire le bruit injecté

        optimizer.zero_grad()                          # reset gradient
        scaler.scale(loss).backward()                  # backward en AMP (scaling pour éviter underflow)
        scaler.step(optimizer)                         # met à jour les poids scaled
        scaler.update()                                # update du scaler AMP
        total_loss += loss.item()                      # cumule la loss pour l'affichage

    print(f"→ Époque {epoch+1}/{num_epochs} | Loss moyenne: {total_loss/len(train_loader):.6f}")

    # sauvegarde tous les 10 epochs
    if (epoch + 1) % 10 == 0:
        os.makedirs("Save", exist_ok=True)             # crée le dossier si inexistant
        torch.save({
            "model": model.state_dict(),               # poids du U-Net
            "embed": class_embed.state_dict(),         # poids du layer d’embedding
        }, f"Save/model_cond_epoch{epoch+1}.pth")      # checkpoint




torch.save({
    "model": model.state_dict(),
    "embed": class_embed.state_dict(),
}, "Save/model_cond_final.pth")

flower_classes = [
    "primevère rose", # 0
    "orchidée à feuilles dures", #1
    "campanule des jardins", #2
    "pois de senteur",  #3
    "souci des jardins",    #4
    "lis tigré",    #5
    "orchidée de la lune", #6
    "oiseau de paradis",    #7
    "aconit",   #8
    "chardon globe",    #9
    "muflier",  #10
    "tussilage", #11
    "protéa royale", #12
    "chardon lancéolé", #13
    "iris jaune",   #14
    "trolle d’Europe",  #15
    "échinacée pourpre", #16
    "lis du Pérou", #17
    "platycodon (fleur ballon)",    #18
    "arum blanc géant", #19
    "lis de feu",   #20
    "scabieuse (fleur coussinet)",  #21
    "fritillaire",  #22
    "gingembre rouge",  #23
    "muscari (jacinthe à grappes)", #24
    "coquelicot",   #25
    "plume du Pays de Galles",  #26
    "gentiane sans tige",   #27
    "artichaut",    #28
    "œillet de poète",  #29
    "œillet",   #30
    "phlox des jardins",    #31
    "nigelle de Damas", #32
    "pavot islandais",  #33
    "violette africaine",   #34
    "lys calla",    #35
    "dahlia",   #36
    "campanule à feuilles de pêcher",   #37
    "gerbera",  #38
    "lis blanc",    #39
    "fleur de la passion",  #40
    "iris bleu",
    "anémone du Japon",
    "hibiscus",
    "glaïeul",
    "coquelourde",  #45
    "tournesol",
    "pélargonium",
    "daisy (marguerite)",
    "rose trémière",
    "pavot de Californie",  #50
    "bougainvillée",
    "camélia",
    "mauve",
    "pétunia mexicain",
    "broméliacée",  #55
    "gaillarde (fleur couverture)",    #56
    "bignone",
    "rosier de Chine",
    "hibiscus rose",
    "lis d’eau",    #60
    "lotus sacré",  #61
    "pissenlit",
    "marguerite du Cap",
    "anémone couronnée",
    "pivoine",  #65
    "chrysanthème",
    "marguerite africaine",
    "jonquille",
    "lys asiatique",
    "orchidée tigrée",  #70
    "fleur de cerisier",    #71
    "coquelicot bleu",
    "narcisse",
    "camomille",
    "amaryllis",    #75
    "lis calla jaune",  #76
    "lys oriental",
    "iris violet",
    "géranium vivace",
    "primevère rouge",  #80
    "pavot d’Islande blanc",
    "hibiscus jaune",
    "pavot de l’Himalaya",
    "zinnia",
    "hortensia",    #85
    "jasmin",
    "fleur de lotus rose",
    "lys de la Madone",
    "coquelicot rose",
    "rose rouge",
    "rose blanche",
    "rose jaune",
    "rose rose",
    "rose orangée",
    "rose violette",
    "rose bleue",
    "rose multicolore",
    "lavande",
    "lys tacheté",
    "fleur de lys du Canada",
    "iris japonais",
    "fleur de pommier",
    "glycine",
    "lys du Canada",
    "pavot oriental",
    "marguerite commune"
]

# ===========================================================
# Échantillonnage conditionnel
# ===========================================================
@torch.no_grad()
def eval_model(epoch_model, inference_steps=1000, class_id=0):
    ckpt = torch.load(f"Save/model_cond_epoch{epoch_model}.pth", map_location=device)

    model.load_state_dict(ckpt["model"])
    class_embed.load_state_dict(ckpt["embed"])
    model.eval()

    B = 16
    img = torch.randn((B, 3, image_size, image_size), device=device)

    cond = class_embed(torch.tensor([class_id] * B, device=device)).unsqueeze(1)

    for t in reversed(range(inference_steps)):
        ts = torch.tensor([t] * B, device=device, dtype=torch.long)
        with torch.amp.autocast("cuda"):
            eps = model(img, ts, encoder_hidden_states=cond).sample
        
        img = scheduler.step(eps, t, img).prev_sample

    # convert to PIL
    img = ((img.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    imgs = [Image.fromarray(img[i].permute(1, 2, 0).cpu().numpy()) for i in range(B)]

    grid = make_image_grid(imgs, rows=4, cols=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.title(f"Classe {flower_classes[class_id]} — modèle epoch {epoch_model}")
    plt.axis("off")
    plt.show()
# ===========================================================
# Exemple d’évaluation
# ===========================================================
# Génère des fleurs de classes spécifiques
for flower_class in [5, 40, 80, 100]:
    eval_model(200, inference_steps=1000, class_id=flower_class)
