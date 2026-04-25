import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 1️⃣ Vérifier si on utilise GPU ou CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device utilisé :", device)

# 2️⃣ Charger BLIP-2
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)

# 3️⃣ Charger plusieurs images locales
image_paths = [
    "images/img_0.jpg",  # chat
    "images/img_1.jpg",  # un homme
    "images/img_2.jpg",  # montagne
]

images = [Image.open(p).convert("RGB") for p in image_paths]

# 4️⃣ Prétraitement
inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)

# 5️⃣ Génération automatique des légendes
generated_ids = model.generate(**inputs, max_new_tokens=50)  
captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Afficher les résultats
for path, caption in zip(image_paths, captions):
    print(f"🖼️ Image : {path}")
    print(f"📝 Légende générée : {caption}\n")

# 6️⃣ Évaluation BLEU / METEOR (optionnelle)
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

# Références factices (normalement tu mets tes vraies légendes)
references = [
    ["a cute cat sitting on a bed"],    # pour img_0
    ["a man standing outdoors"],        # pour img_1
    ["a snowy mountain landscape"]      # pour img_2
]

print("🔎 Évaluation des légendes :")
for ref, cand in zip(references, captions):
    bleu = sentence_bleu([ref], cand.split())
    meteor = meteor_score([ref], cand)
    print(f"Référence : {ref[0]}")
    print(f"Candidat  : {cand}")
    print(f"➡️ BLEU = {bleu:.4f}, METEOR = {meteor:.4f}\n")
