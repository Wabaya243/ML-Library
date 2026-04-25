import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ===============================
# Hyperparamètres
# ===============================
NOISE_DIM = 100  # dimension du vecteur latent (bruit)
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Prétraitement MNIST [-1,1]
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),  # (H,W,C) -> (C,H,W) et normalise en [0,1]
    transforms.Normalize((0.5,), (0.5,))  # [0,1] -> [-1,1]
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# Générateur
# ===============================
class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 7*7*128, bias=False),  # couche fully connected -> 7x7x128
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),  # reshape en (128,7,7)
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),  # 28x28x1
            nn.Tanh()  # sortie dans [-1,1]
        )
    
    def forward(self, x):
        return self.model(x)

# ===============================
# Discriminateur
# ===============================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # 28x28x1 -> 14x14x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 14x14x64 -> 7x7x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(7*7*128, 1)  # sortie score sans sigmoid (from_logits=True)
        )
    
    def forward(self, x):
        return self.model(x)

# ===============================
# Fonctions de perte
# ===============================
criterion = nn.BCEWithLogitsLoss()  # combine sigmoid + BCE

def discriminator_loss(real_output, fake_output):
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss

def generator_loss(fake_output):
    return criterion(fake_output, torch.ones_like(fake_output))  # veut que D dise 1

# ===============================
# Initialisation des modèles et optimizers
# ===============================
generator_model = Generator(NOISE_DIM).to(DEVICE)
discriminator_model = Discriminator().to(DEVICE)

generator_optimizer = optim.Adam(generator_model.parameters(), lr=LR, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=LR, betas=(0.5, 0.999))

# ===============================
# Fonction pour afficher les images générées
# ===============================
def generate_and_show_images(model, test_input):
    with torch.no_grad():
        predictions = model(test_input).cpu()
    # [-1,1] -> [0,1]
    predictions = (predictions + 1.0) / 2.0
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,0,:,:], cmap='gray')
        plt.axis('off')
    plt.show()

# ===============================
# Boucle d'entraînement
# ===============================
fixed_noise = torch.randn(16, NOISE_DIM, device=DEVICE)  # bruit fixe pour visualiser

for epoch in range(EPOCHS):
    for batch in tqdm(train_loader, desc=f"{epoch+1}/{EPOCHS}"):
        real_images, _ = batch
        real_images = real_images.to(DEVICE)

        # --- Update discriminateur ---
        noise = torch.randn(real_images.size(0), NOISE_DIM, device=DEVICE)
        fake_images = generator_model(noise)

        discriminator_optimizer.zero_grad()
        real_output = discriminator_model(real_images)
        fake_output = discriminator_model(fake_images.detach())
        d_loss = discriminator_loss(real_output, fake_output)
        d_loss.backward()
        discriminator_optimizer.step()

        # --- Update générateur ---
        generator_optimizer.zero_grad()
        fake_output = discriminator_model(fake_images)
        g_loss = generator_loss(fake_output)
        g_loss.backward()
        generator_optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
    generate_and_show_images(generator_model, fixed_noise)
