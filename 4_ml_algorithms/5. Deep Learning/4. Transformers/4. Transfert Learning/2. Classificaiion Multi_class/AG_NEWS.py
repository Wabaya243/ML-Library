# -------------------------
# Étape 0 : Import des librairies
# -------------------------
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset

# -------------------------
# Étape 1 : Charger le dataset AG News
# -------------------------
# AG News : 4 classes (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
datasets = load_dataset("ag_news")
print(datasets)

# -------------------------
# Étape 2 : Charger le tokenizer BERT
# -------------------------
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# -------------------------
# Étape 3 : Tokenisation du texte
# -------------------------
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# -------------------------
# Étape 4 : Préparer les colonnes pour PyTorch
# -------------------------
# Renommer la colonne de labels et formater pour PyTorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# -------------------------
# Étape 5 : Charger le modèle BERT pour classification multi-classes
# -------------------------
num_classes = 4
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ajouter une couche linéaire en dur sur la sortie du classifieur
# Le classifier original : model.classifier (souvent une seule couche)
old_classifier = model.classifier

# On empile tout : sortie BERT -> ancienne tête -> nouvelle couche
model.classifier = nn.Sequential(
    old_classifier,
    nn.ReLU(),
    nn.Linear(model.config.num_labels, 8),  # nouvelle couche intermédiaire
    nn.ReLU(),
    nn.Linear(8, model.config.num_labels)   # sortie finale (même nb de classes)
)

print(model.classifier)

# Geler toutes les couches sauf la tête de classification (pour accélérer l'entraînement)
for param in model.bert.parameters():
    param.requires_grad = False
    
for param in model.classifier.parameters():
    param.requires_grad = True

# -------------------------
# Étape 6 : Définir les arguments d'entraînement
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    fp16=True,  # active le float16, plus rapide sur GPU
)

# -------------------------
# Étape 7 : Définir le Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# -------------------------
# Étape 8 : Fine-tuning du modèle
# -------------------------
trainer.train()


layers = list(model.bert.encoder.layer)

# dégèle les 4 dernières couches
for layer in layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

'''

# -------------------------
# Étape 8b : Sauvegarder le modèle fine-tuné
# -------------------------
trainer.save_model("./bert_sentiment_model")
# Sauvegarde aussi le tokenizer pour réutilisation
tokenizer.save_pretrained("./bert_sentiment_model")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Charger le modèle et le tokenizer sauvegardés
tokenizer = AutoTokenizer.from_pretrained("./bert_sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("./bert_sentiment_model")

# Tester la prédiction
def predict_sentiment(text):
    encodings = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    sentiment = 'positif' if pred == 1 else 'négatif'
    print(f"Texte : {text}\nPrédiction : {sentiment}, Probabilités : {probs.tolist()}")

predict_sentiment("I really enjoyed this movie!")
predict_sentiment("The movie was awful and I hated it")


'''


# -------------------------
# Étape 9 : Évaluer le modèle
# -------------------------
results = trainer.evaluate()
print("Évaluation :", results)

# -------------------------
# Étape 10 : Tester avec un exemple de texte (prédiction multi-classes)
# -------------------------
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def predict_category(text):
    encodings = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    print(f"Texte : {text}\nClasse prédite : {label_map[pred]}, Probabilités : {probs.tolist()}")

# Exemples
predict_category("The stock market crashed due to economic instability.")
predict_category("The football team won the championship last night.")
predict_category("NASA announced a new Mars rover mission.")

# -------------------------
# Étape 11 : Génération de texte avec GPT2
# -------------------------
gpt_model = AutoModelForCausalLM.from_pretrained('gpt2')

def generate_text(prompt, max_len=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = gpt_model.generate(input_ids, max_length=max_len, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_text("Once upon a time in the world of sports"))

# -------------------------
# Étape 12 : Génération de texte guidée par la catégorie détectée
# -------------------------
def category_guided_generation(text):
    # détecter la catégorie avec BERT
    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    output = model(**encodings)
    pred = torch.argmax(torch.softmax(output.logits, dim=1)).item()
    category = label_map[pred]

    # générer du texte avec GPT
    prompt = f"Write a short news article in the {category} category: {text}"
    generated_text = generate_text(prompt)
    
    print("Catégorie détectée :", category)
    print("Texte généré :", generated_text)
    
    return generated_text

# Exemples
category_guided_generation("A new breakthrough in quantum computing has been announced.")
category_guided_generation("The local basketball team won the finals.")


########## Traduction

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Charger le modèle multilingue mBART (gère plusieurs langues)
model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer_trad = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model_trad = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def traduire_texte(texte_fr, src_lang="en_XX", tgt_lang="fr_XX"):
    # Spécifier la langue source pour le tokenizer
    tokenizer_trad.src_lang = src_lang
    
    # Tokeniser le texte français
    inputs = tokenizer_trad(texte_fr, return_tensors="pt")
    
    # Forcer le token de début de phrase pour la langue cible (important !)
    forced_bos_token_id = tokenizer_trad.lang_code_to_id[tgt_lang]
    
    # Générer la traduction
    outputs = model_trad.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )
    
    # Décoder et retourner la traduction
    traduction = tokenizer_trad.decode(outputs[0], skip_special_tokens=True)
    return traduction

# Exemple d'utilisation

# Texte généré par GPT-2
texte_generé = category_guided_generation("A new breakthrough in quantum computing has been announced.")

# Traduire vers le français
traduction = traduire_texte(texte_generé, src_lang="en_XX", tgt_lang="fr_XX")

print("Texte généré :", texte_generé)
print("Traduction :", traduction)



