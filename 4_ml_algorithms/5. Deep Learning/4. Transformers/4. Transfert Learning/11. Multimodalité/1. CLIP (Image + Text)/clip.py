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


# 2️ Créer un petit dataset personnalisé
# Ici on prend des images en ligne (exemple : chat, chien)
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

# Téléchargement et sauvegarde locale pour simplifier
os.makedirs("images", exist_ok=True)
image_paths = []
for i, url in enumerate(image_urls):
    path = f"images/img_{i}.jpg"
    if not os.path.exists(path):
        img = Image.open(requests.get(url, stream=True).raw)
        img.save(path)
    image_paths.append(path)

# 3️ Créer un Dataset PyTorch pour CLIP
class CLIPDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        caption = self.captions[idx]
        # Préparer les inputs pour CLIP
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
        return {key: val.squeeze(0) for key, val in inputs.items()}  # enlever la dimension batch


# 4️ Charger le modèle et le processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model = model.to(device)

# 5️ Créer DataLoader
dataset = CLIPDataset(image_paths, captions, processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 6️ Définir optimiseur
optimizer = AdamW(model.parameters(), lr=5e-6)
#LR tres petit pour evité de casser le poids du model pre_entrainé

# 7️ Boucle de fine-tuning (très simple, 3 epochs pour démo)
model.train()
epochs = 3

for epoch in epochs(epochs):
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        #deplcaer les tenseur sur GPU
        batch = {k: v.to(device) for k,v in batch.items()}

        outputs = model(**batch)
        #CLIP fourni la loss cross entropy interne
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss.set_description(f"epoch : {epoch+1}")
        loss.set_postfix(loss=loss.item())

#### 8️ Tester le modèle fine-tuné

model.eval()
with torch.no_grad():
    test_caption = "un chat mignon"
    test_image = Image.open(image_paths[0]).convert("RGB")
    
    inputs = processor(text=test_caption, images=test_image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    similarity = (text_embeds @ image_embeds.T).item()
    print("Similarité texte ↔ image :", similarity)













