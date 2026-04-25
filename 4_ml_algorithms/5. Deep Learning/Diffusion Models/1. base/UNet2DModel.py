
# librairies de base Deep Learning
import torch
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

# dataset
from datasets import load_dataset
from tqdm import tqdm
import os

# modèles de diffusion (librairie HuggingFace diffusers)
from diffusers import DDPMScheduler     # planification du bruit (scheduler)
from diffusers import UNet2DModel       # U-Net pour diffusion
from diffusers import DDPMPipeline      # pipeline entraînement/inférence
from diffusers.utils import make_image_grid

# visualisation
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# =====================================================================
#  Chargement du dataset
# =====================================================================

# Dataset contenant des images de fleurs
# Ici on ne prend qu'une petite fraction (10%) pour l'exemple
dataset = load_dataset("huggan/flowers-102-categories", split='train[:10%]')

# =====================================================================
#  Pré-traitement des données (Data Augmentation + Normalisation)
# =====================================================================
image_size = 64   # redimensionner toutes les images en 64x64
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),   # resize
    # transforms.RandomHorizontalFlip(),           # on peut activer du flip aléatoire
    transforms.ToTensor(),                         # convertir en tenseur PyTorch
    transforms.Normalize([0.5], [0.5]),            # normalisation entre [-1,1]
])

# appliquer le pré-traitement au dataset
def transform(examples):
    # convertir les images en RGB puis appliquer la pipeline de prétraitement
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# choisir GPU si dispo sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Batch size: {batch_size}")
print(f"Dataset size (nb batches): {len(train_loader)}")
print(f"Device found: {device}")

# =====================================================================
#  Fonction pour afficher un batch d’images
# =====================================================================

def show_images(image):
    # denormalisation pour revenir en [0,1]
    plt.imshow(make_grid(image * 0.5 + 0.5).permute(1, 2, 0))
    plt.show()

batch = next(iter(train_loader))
plt.figure(figsize=(15, 6))
show_images(batch["images"][:8])


# =====================================================================
#  Définition du Scheduler et du Modèle (U-Net)
# =====================================================================

# Scheduler = définit comment on ajoute du bruit aux images pendant l’entraînement
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)

# U-Net adapté à la génération d’images par diffusion
model = UNet2DModel(
    sample_size=image_size,            # résolution cible (64x64)
    in_channels=3,                     # canaux d’entrée (3 pour RGB)
    out_channels=3,                    # canaux de sortie (prédiction bruit sur RGB)
    layers_per_block=2,                # nb de couches ResNet par bloc U-Net
    block_out_channels=(128, 128, 256, 256, 512, 512),  # largeur des blocs
    down_block_types=(
        "DownBlock2D", 
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",   # bloc avec self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D", 
        "AttnUpBlock2D",     # bloc avec self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


print(model)

# =====================================================================
#  Entraînement
# =====================================================================
num_epochs = 200
lr = 1e-4  # learning rate
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

losses = []
scaler = torch.cuda.amp.GradScaler()

# --- Entraînement ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"{epoch+1}/{num_epochs}")):
        clean_images = batch["images"].to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

        with torch.cuda.amp.autocast():
            pred = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} | Mean Loss: {total_loss/len(train_loader):.6f}")

    if (epoch + 1) % 5 == 0:
        os.makedirs("Save", exist_ok=True)
        torch.save(model.state_dict(), f"Save/model_{epoch+1}.pth")
        
        
torch.save(model.state_dict(), "Save/model_final.pth")



# =====================================================================
#  Fonction d’échantillonnage (génération d’images depuis du bruit)
# =====================================================================

def eval_model(epoch_model, inference_steps=1000):
    # recréer le même modèle qu’avant
    nn_model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    
    # charger les poids sauvegardés
    nn_model.load_state_dict(torch.load(f"Save/model_{epoch_model}.pth", map_location=device))
    nn_model.eval()
    print(" Model chargé")

    # pipeline Diffusion = modèle U-Net + scheduler
    pipe = DDPMPipeline(unet=nn_model, scheduler=scheduler)
    pipe.to(device)

    # génération d’images aléatoires depuis du bruit pur
    generated_image = pipe(batch_size=128, num_inference_steps=inference_steps, output_type='pil', return_dict=True)
    images = generated_image[0][:16]

    # affichage en grille
    image_grid = make_image_grid(images, rows=4, cols=5)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_grid)
    ax.axis('off')
    ax.set_title(f"Images générées a avec le model a {epoch_model} epoch et steps {inference_steps}", fontsize=16)
    plt.show()


# =====================================================================
#  Évaluation du modèle
# =====================================================================

# on compare la qualité des images générées selon :
# - l’époque d’entraînement (5, 50, 200)
# - le nombre d’étapes de diffusion (500 vs 1000)

eval_model(5, 500)      # modèle peu entraîné + peu d’étapes
eval_model(5, 1000)     # idem mais plus d’étapes
eval_model(50, 1000)    # modèle entraîné plus longtemps
eval_model(100, 1000)    # modèle entraîné plus longtemps
eval_model(150, 1000)    # modèle entraîné plus longtemps
eval_model(200, 1000)   # modèle final
for steps in [100, 250, 500, 1000]:
    eval_model(200, inference_steps=steps)

