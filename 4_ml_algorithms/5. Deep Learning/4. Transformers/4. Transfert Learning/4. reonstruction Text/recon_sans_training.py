# =========================
# Reconstruction de mots masqués (MLM) en français
# =========================

from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch

# Charger CamemBERT pré-entraîné
model_name = "camembert-base"  # français
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Exemple de texte avec un token masqué
text = "Le film était <mask> et j'ai beaucoup apprécié la performance des acteurs."

# Tokenization
inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

# Génération des prédictions pour le token masqué
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Récupérer le top 5 des tokens les plus probables
mask_token_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print("Top 5 des prédictions pour <mask> :")
for token in top_5_tokens:
    word = tokenizer.decode([token])
    print(word)

# =========================
# Variante avec pipeline Hugging Face
# =========================
fill_mask = pipeline("fill-mask", model=model_name, tokenizer=tokenizer)

results = fill_mask("Le film était <mask> et j'ai beaucoup apprécié la performance des acteurs.")
print("\nRésultats via pipeline :")
for r in results:
    print(f"{r['sequence']} (score: {r['score']:.4f})")
