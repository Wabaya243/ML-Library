from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Charger le modèle multilingue mBART (gère plusieurs langues)
model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def traduire_texte(texte_fr, src_lang="fr_XX", tgt_lang="en_XX"):
    # Spécifier la langue source pour le tokenizer
    tokenizer.src_lang = src_lang
    
    # Tokeniser le texte français
    inputs = tokenizer(texte_fr, return_tensors="pt")
    
    # Forcer le token de début de phrase pour la langue cible (important !)
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    
    # Générer la traduction
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )
    
    # Décoder et retourner la traduction
    traduction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return traduction

# Exemple d'utilisation
texte_francais = "Yo mon ami tu fais quoi maintenant ?"
traduction_anglais = traduire_texte(texte_francais)
print("Texte original :", texte_francais)
print("Traduction :", traduction_anglais)





# Choisis un dossier local pour sauvegarder
chemin_sauvegarde = "./mbart_traducteur"

# Sauvegarder le tokenizer
tokenizer.save_pretrained(chemin_sauvegarde)

# Sauvegarder le modèle
model.save_pretrained(chemin_sauvegarde)


# Plus tard, recharger depuis local
tokenizer = AutoTokenizer.from_pretrained(chemin_sauvegarde)
model = AutoModelForSeq2SeqLM.from_pretrained(chemin_sauvegarde)