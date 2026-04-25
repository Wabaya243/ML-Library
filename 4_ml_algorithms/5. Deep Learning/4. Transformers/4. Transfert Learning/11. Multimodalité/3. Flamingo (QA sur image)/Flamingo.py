import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AdamW, get_scheduler
from PIL import Image
from tqdm import tqdm

# ⚡ Détection GPU (ou CPU si pas dispo)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📦 On charge le modèle Flamingo pré-entraîné (OpenFlamingo 3B)
# "AutoProcessor" gère à la fois les images et le texte
model_name = "openflamingo/OpenFlamingo-3B-vitl-mpt1b"
processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForVision2Seq.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, # FP16 = moins de mémoire
    device_map="auto"          # met automatiquement sur GPU/CPU dispo
)

# 🟢 === Pourquoi LoRA ? ===
# Fine-tuner un modèle 3B complet est beaucoup trop coûteux (VRAM énorme).
# LoRA (Low-Rank Adaptation) permet de n’ajouter QUE quelques matrices 
# (de petit rang) aux poids existants, au lieu d’entraîner tout le modèle.
# → Résultat : on n’entraîne que 1-2% des paramètres, donc c’est + rapide et + léger.

# Configuration LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # type de tâche : génération texte à partir d’images+texte
    r=8,               # rang faible : taille des matrices LoRA (plus petit = plus léger)
    lora_alpha=16,     # facteur de mise à l’échelle
    lora_dropout=0.1,  # dropout pour éviter overfitting
    bias="none"        # on ne touche pas aux biais du modèle
)

# On applique LoRA au modèle Flamingo
model = get_peft_model(model, lora_config)

# === Jeu de données jouet ===
# Chaque élément = une image, une question et la réponse attendue
data = [
    {"image": "images/cat.jpg", "question": "Quel animal est sur l'image ?", "answer": "Un chat"},
    {"image": "images/car.jpg", "question": "Quelle est la couleur de la voiture ?", "answer": "Rouge"},
    {"image": "images/mountain.jpg", "question": "Quel type de paysage est montré ?", "answer": "Une montagne"}
]

# Dataset personnalisé
class FlamingoDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        
        # Le processor gère l’encodage image + question
        inputs = self.processor(
            images=image,
            text=item["question"],
            return_tensors="pt",
            padding=True
        )
        
        # La réponse est tokenizée comme "labels"
        labels = self.processor.tokenizer(item["answer"], return_tensors="pt").input_ids
        inputs["labels"] = labels.squeeze(0)
        
        # On retourne un batch prêt pour le modèle
        return {k: v.squeeze(0) for k, v in inputs.items()}

# On crée DataLoader
train_dataset = FlamingoDataset(data, processor)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# === Optimiseur + Scheduler ===
optimizer = AdamW(model.parameters(), lr=1e-4)
num_training_steps = len(train_dataloader) * 3  # 3 epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 🚀 Entraînement
model.train()
epochs = 3

for epoch in range(epochs):
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        # On passe batch → Flamingo
        outputs = model(**batch)
        loss = outputs.loss

        # Backpropagation classique
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Affichage progression
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Sauvegarde du modèle fine-tuné (LoRA)
model.save_pretrained("flamingo-finetuned")
processor.save_pretrained("flamingo-finetuned")

# 🧪 Test : poser une question sur une image
model.eval()
test_image = Image.open("images/car.jpg").convert("RGB")
question = "Quelle est la couleur de la voiture ?"

# Prétraitement entrée
inputs = processor(images=test_image, text=question, return_tensors="pt").to(device)

# Génération réponse
generated_ids = model.generate(**inputs, max_new_tokens=50)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("❓ Question :", question)
print("🤖 Réponse fine-tunée :", answer)
