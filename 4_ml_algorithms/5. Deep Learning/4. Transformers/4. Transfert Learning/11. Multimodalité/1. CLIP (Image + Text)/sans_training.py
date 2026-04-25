import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from PIL import Image
import requests
from tqdm import tqdm
import os

# Vérifier si GPU est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 2️⃣ Créer un petit dataset personnalisé
image_urls = [
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131",  # chat
    "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",  # chien
    "https://images.unsplash.com/photo-1472214103451-9374bd1c798e",  # montagne
]

captions = [
    "un chat mignon",
    "un chien adorable",
    "une montagne enneigée"
]

# Téléchargement et sauvegarde locale
os.makedirs("images", exist_ok=True)
image_paths = []
for i, url in enumerate(image_urls):
    path = f"images/img_{i}.jpg"
    if not os.path.exists(path):
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        img.save(path)
    image_paths.append(path)

# 🔹 Charger les images avec PIL
images = [Image.open(p).convert("RGB") for p in image_paths]

# 4️⃣ Charger le modèle et le processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)

# Préparer les inputs
inputs = processor(text=captions, images=images, return_tensors='pt', padding=True).to(device)

# 8️⃣ Tester le modèle fine-tuné
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

# Normaliser les vecteurs
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Similarité cosinus texte ↔ image
similarity = text_embeds @ image_embeds.T
print("Matrice de similarité :\n", similarity)

# Trouver la meilleure correspondance pour chaque texte
for i, text in enumerate(captions):
    best_idx = similarity[i].argmax().item()
    print(f"Texte : '{text}' -> Image la plus proche : {image_urls[best_idx]}")
