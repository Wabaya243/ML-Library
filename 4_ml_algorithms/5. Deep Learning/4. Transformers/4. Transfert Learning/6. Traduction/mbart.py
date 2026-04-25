# -------------------------
# Étape 0 : Import des librairies
# -------------------------
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Trainer, TrainingArguments
from datasets import Dataset

# Vérification du device GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Étape 1 : Préparer un dataset Lingala -> Français (augmenté)
# -------------------------
# Ajout de phrases plus variées pour enrichir le dataset
data = {
    "source": [
        "Nalingi yo",                
        "Mbote na yo",               
        "Naboyi koluka",             
        "Tika ngai",                 
        "Tosakoli mbongo",           
        "Ngai nazali na posa ya kolya",  
        "Oza malamu?",               
        "Nazali na posa ya kofanda na ndako",  
        "Tokendaki na ndako ya mobali na ngai",  
        "Bolingo ezali kitoko"
    ],
    "target": [
        "Je t'aime",
        "Bonjour",
        "Je ne veux pas chercher",
        "Laisse-moi",
        "Nous avons dépensé de l'argent",
        "moi J'ai faim",
        "Ça va ?",
        "Je veux rester à la maison",
        "Nous sommes allés chez mon mari",
        "L'amour est beau"
    ]
}

dataset = Dataset.from_dict(data)
print(dataset)

# -------------------------
# Étape 2 : Charger le tokenizer et le modèle mBART50
# -------------------------
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Définir la langue source et cible
tokenizer.src_lang = "ln_CD"  # Lingala
target_lang = "fr_XX"         # Français

# -------------------------
# Étape 3 : Tokenisation
# -------------------------
def preprocess(batch):
    # Tokeniser les phrases source
    inputs = tokenizer(batch["source"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    # Tokeniser les phrases cibles
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels["input_ids"]
    }
    return batch

tokenized_dataset = dataset.map(preprocess, batched=True)

# -------------------------
# Étape 4 : Définir les arguments d'entraînement
# -------------------------
training_args = TrainingArguments(
    output_dir="./mbart_lingala_fr",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=3e-5,
    evaluation_strategy="no",
    remove_unused_columns=False,
)

# -------------------------
# Étape 5 : Définir le Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -------------------------
# Étape 6 : Fine-tuning
# -------------------------
trainer.train()

# -------------------------
# Étape 7 : Tester la traduction
# -------------------------
def translate(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=128
    )
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# Exemples de test
test_phrases = [
    "Nalingi yo",
    "Mbote na yo",
    "Ngai nazali na posa ya kolya",
    "Bolingo ezali kitoko",
    "Tosakoli mbongo"
]

for phrase in test_phrases:
    print(f"Lingala : {phrase} --> Français : {translate(phrase)}")
