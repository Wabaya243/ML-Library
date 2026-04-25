from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset
import torch

#pour forcé la verbosité
import transformers
transformers.utils.logging.set_verbosity_info()

# --- 1️ Charger le dataset ---
dataset = load_dataset("ag_news")

# --- 2️ Tokenizer + modèle Roberta ---
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model_roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

# --- 3️ Tokenisation ---
def tokenize_function(examples):
    return roberta_tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# --- 4️ Entraînement Roberta ---
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,  # mets 3 pour un vrai entraînement
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,

    # Désactive TensorBoard / WandB
    report_to=[],
    disable_tqdm=True,
)

trainer = Trainer(
    model=model_roberta,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=roberta_tokenizer,
)

print("=== Début de l'entraînement Roberta ===")
trainer.train()

# --- 5️ Sauvegarder le modèle ---
trainer.save_model("./roberta_finetuned")
roberta_tokenizer.save_pretrained("./roberta_finetuned")
print(" Roberta sauvegardé dans ./roberta_finetuned")


# --- 6️ Recharger Roberta entraîné ---
label_names = ["World", "Sports", "Business", "Sci/Tech"]
roberta_model = AutoModelForSequenceClassification.from_pretrained("./roberta_finetuned")

# --- 7️ Charger GPT-2 (ou GPT-2 FR) ---
gpt2_model_name = "dbddv01/gpt2-french-small"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

# --- 8️ Prédire la catégorie ---
def predict_category(text):
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    return label_names[predicted_class_id]

# --- 9️ Générer un texte avec GPT-2 ---
def generate_text(category):
    prompt = f"Écris un court article de presse dans le domaine {category.lower()}."
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.7,
        do_sample=True
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---  Démo ---
texte_utilisateur = "Cristiano ronaldo marque un but spectaculaire lors du match."
categorie = predict_category(texte_utilisateur)
print(f"\nCatégorie prédite : {categorie}")

texte_genere = generate_text(categorie)
print("\nTexte généré :\n", texte_genere)
