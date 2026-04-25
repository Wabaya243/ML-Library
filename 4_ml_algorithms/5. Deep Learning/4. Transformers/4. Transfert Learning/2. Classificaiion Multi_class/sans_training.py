# -------------------------
# Étape 0 : Import des librairies
# -------------------------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# -------------------------
# Étape 1 : Charger BERT pré-entraîné pour classification multi-classes
# -------------------------
# Exemple : "cardiffnlp/twitter-roberta-base-ag-news" est pré-entraîné sur AG News
tokenizer_cls = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-ag-news")
model_cls = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-ag-news")

label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# -------------------------
# Étape 2 : Fonction de prédiction de catégorie
# -------------------------
def predict_category(text):
    inputs = tokenizer_cls(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    model_cls.eval()
    with torch.no_grad():
        outputs = model_cls(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    print(f"Texte : {text}\nCatégorie prédite : {label_map[pred]}, Probabilités : {probs.tolist()}")

# -------------------------
# Étape 3 : Tester la classification
# -------------------------
predict_category("The stock market crashed due to economic instability.")
predict_category("The football team won the championship last night.")
predict_category("NASA announced a new Mars rover mission.")

# -------------------------
# Étape 4 : Charger GPT-2 pour génération de texte
# -------------------------
tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")
model_gpt = AutoModelForCausalLM.from_pretrained("gpt2")

def generate_text(prompt, max_len=100):
    input_ids = tokenizer_gpt.encode(prompt, return_tensors="pt")
    output = model_gpt.generate(input_ids, max_length=max_len, num_return_sequences=1)
    return tokenizer_gpt.decode(output[0], skip_special_tokens=True)

# -------------------------
# Étape 5 : Génération guidée par catégorie détectée
# -------------------------
def category_guided_generation(text):
    # Détecter la catégorie
    inputs = tokenizer_cls(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_cls(**inputs)
        pred = torch.argmax(torch.softmax(outputs.logits, dim=1)).item()
        category = label_map[pred]

    # Générer le texte avec GPT-2
    prompt = f"Write a short news article in the {category} category: {text}"
    generated_text = generate_text(prompt)
    
    print("Catégorie détectée :", category)
    print("Texte généré :", generated_text)

# -------------------------
# Étape 6 : Tester la génération guidée
# -------------------------
category_guided_generation("A new breakthrough in quantum computing has been announced.")
category_guided_generation("The local basketball team won the finals.")
