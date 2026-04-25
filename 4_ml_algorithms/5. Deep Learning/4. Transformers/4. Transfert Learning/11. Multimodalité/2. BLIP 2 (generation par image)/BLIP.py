import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AdamW, get_scheduler
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# --------- 1. Device setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --------- 2. Charger modèle et processor ----------
model_name = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float16
).to(device)

# --------- 3. Geler tout sauf la tête ----------
for name, param in model.named_parameters():
    param.requires_grad = False

# dégeler uniquement la tête de génération (langage)
for name, param in model.named_parameters():
    if "lm_head" in name or "language_model.model.decoder" in name:
        param.requires_grad = True

trainable = sum(p.requires_grad for p in model.parameters())
total = sum(1 for _ in model.parameters())
print(f"Paramètres entraînables : {trainable}/{total}")

# --------- 4. Dataset Flickr8k ----------
dataset = load_dataset("flickr8k", split="train[:500]")  # petit subset

class CaptionDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item["image"]).convert("RGB")
        caption = item["caption"]

        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}

train_dataset = CaptionDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# --------- 5. Optimizer / scheduler ----------
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_training_steps = len(train_dataloader) * 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# --------- 6. Entraînement ----------
model.train()
epochs = 3

for epoch in range(epochs):
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

# --------- 7. Sauvegarde ----------
model.save_pretrained("blip2-finetuned-head")
processor.save_pretrained("blip2-finetuned-head")

# --------- 8. Test ----------
model.eval()
test_image = Image.open(dataset[0]["image"]).convert("RGB")

inputs = processor(images=test_image, return_tensors="pt").to(device)
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n Légende générée après fine-tuning :", caption)
