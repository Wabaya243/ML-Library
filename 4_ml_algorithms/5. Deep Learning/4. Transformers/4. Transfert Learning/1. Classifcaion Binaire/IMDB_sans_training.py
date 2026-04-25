# -------------------------
# Étape 0 : Import des librairies
# -------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# -------------------------
# Étape 1 : Charger le tokenizer et modèle pré-entraîné pour la classification
# -------------------------
# Ici on utilise BERT pré-entraîné sur SST-2 (sentiment)
tokenizer_cls = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model_cls = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# -------------------------
# Étape 2 : Définir une fonction de prédiction de sentiment
# -------------------------
def predict_sentiment(text):
    inputs = tokenizer_cls(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    model_cls.eval()
    with torch.no_grad():
        outputs = model_cls(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    # Ici le modèle a 5 classes de sentiment (1 étoile → négatif, 5 étoiles → positif)
    sentiment = "positif" if pred >= 3 else "négatif"
    print(f"Texte : {text}\nPrédiction : {sentiment}, Probabilités : {probs.tolist()}")

# -------------------------
# Étape 3 : Tester la prédiction
# -------------------------
predict_sentiment("I really enjoyed this movie!")
predict_sentiment("The movie was awful and I hated it")

# -------------------------
# Étape 4 : Charger GPT‑2 pour la génération de texte
# -------------------------
tokenizer_gpt = AutoTokenizer.from_pretrained('gpt2')
model_gpt = AutoModelForCausalLM.from_pretrained('gpt2')
    
# -------------------------
# Étape 5 : Génération de texte guidée par le sentiment
# -------------------------
def sentiment_guided_generation(text):
    # Détecter le sentiment
    inputs = tokenizer_cls(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_cls(**inputs)
        sentiment = "positive" if torch.argmax(torch.softmax(outputs.logits, dim=1)) >= 3 else "negative"

    # Générer le texte avec GPT-2
    prompt = f"Write a short story with a {sentiment} mood: {text}"
    input_ids = tokenizer_gpt.encode(prompt, return_tensors='pt')
    generated = model_gpt.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer_gpt.decode(generated[0], skip_special_tokens=True)
    
    print("Sentiment détecté :", sentiment)
    print("Texte généré :", generated_text)

# -------------------------
# Étape 6 : Tester la génération
# -------------------------
sentiment_guided_generation("I had a really bad day today")
sentiment_guided_generation("The movie was amazing and I loved it")
