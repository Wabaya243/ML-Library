# -------------------------
# Étape 0 : Import des librairies
# -------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# -------------------------
# Étape 1 : Charger le dataset IMDB
# -------------------------

datasets = load_dataset("ag_news")
print(datasets)

# -------------------------
# Étape 2 : Charger le tokenizer BERT
# -------------------------
# BERT convertit le texte en tokens et IDs utilisables par le modèle
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# -------------------------
# Étape 3 : Tokenisation du texte
# -------------------------
# Padding → toutes les séquences ont la même longueur
# Truncation → tronque les séquences trop longues
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

#appliquer la tokenisation sur tous les text
tokenized_datasets = datasets.map(tokenize_function, batched=True)


# -------------------------
# Étape 4 : Préparer les colonnes pour PyTorch
# -------------------------
# Le Trainer attend que la colonne de labels s'appelle "labels"
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch')

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# -------------------------
# Étape 5 : Charger le modèle BERT pour classification binaire
# -------------------------

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# Geler toutes les couches sauf la tête de classification
for param in model.bert.parameters():
    param.requires_grad = False

#Liberer les couches des classification seulement
for param in model.classifier.parameters():
    param.requires_grad = True


# -------------------------
# Étape 6 : Définir les arguments d'entraînement
# -------------------------

training_args = TrainingArguments(
    output_dir="./results",          # où stocker les checkpoints
    eval_strategy="steps",           # évaluer à chaque epoch
    eval_steps=50,
    learning_rate=2e-5,              # learning rate classique pour BERT
    per_device_train_batch_size=8,   # au lieu de 16 si max_length=256 ou plus
    per_device_eval_batch_size=64,   # batch size pour évaluation
    num_train_epochs=3,              # nombre d'epochs pour test rapide
    weight_decay=0.01,               # régularisation
    logging_dir="./logs",            # dossier pour logs
    logging_steps=10,
    save_steps=500,
)

# -------------------------
# Étape 7 : Définir le Trainer
# -------------------------

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    tokenizer = tokenizer,  # le tokenizer est nécessaire pour la génération de batch
)

# -------------------------
# Étape 8 : Fine-tuning du modèle
# -------------------------

trainer.train()


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
# Étape 10 : Tester avec un exemple de texte
# -------------------------

def predict_sentiment(text):
    encodings = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    encodings = {k:v for k,v in encodings.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    sentiment = 'positif' if pred == 1 else 'négatif'
    print(f"Texte : {text}\nPrédiction : {sentiment}, Probabilités : {probs.tolist()}")

# Exemple
predict_sentiment("This movie was amazing and I loved it!")
predict_sentiment("I did not like this film at all. It was boring.")
predict_sentiment("The acting was terrible, and the plot was predictable.")

# -------------------------
# Étape 11 : Génération de texte avec GPT2
# -------------------------
from transformers import AutoModelForCausalLM

gpt_model = AutoModelForCausalLM.from_pretrained('gpt2')

input_text = "Once upon a time in china"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

#generation du text
output = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1)
print("text genere :", tokenizer.decode(output[0], skip_special_tokens=True))


#### Bonus predire du text a partir du sentiment predit par le modèle BERT

def sentiment_guided_generation(text):
    # detecter les sentiment avec Bert
    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    output = model(**encodings)
    sentiment = "positive" if torch.argmax(torch.softmax(output.logits, dim=1)) == 1 else "negative"

    #generer le text avec GPT
    prompt = f"Write a short story with a {sentiment} mood: {text}"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated = gpt_model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print("Sentiment détecté :", sentiment)
    print("Texte généré :", generated_text)


sentiment_guided_generation("I had a really bad day today")
sentiment_guided_generation("The movie was amazing and I loved it")



