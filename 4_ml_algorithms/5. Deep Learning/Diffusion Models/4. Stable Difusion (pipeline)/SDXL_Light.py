from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionXLLightningPipeline
from diffusers import StableDiffusionXLPipeline
import torch
import xformers

device = "cuda" if torch.cuda.is_available() else "cpu"

def traduire_texte(texte_fr, src_lang="fr_XX", tgt_lang="en_XX"):
    # Spécifier la langue source pour le tokenizer
    translator_tokenizer.src_lang = src_lang
    
    # Tokeniser le texte français
    inputs = translator_tokenizer(texte_fr, return_tensors="pt").to(device)
    
    # Forcer le token de début de phrase pour la langue cible (important !)
    forced_bos_token_id = translator_tokenizer.lang_code_to_id[tgt_lang]
    
    # Générer la traduction
    outputs = translator_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )
    
    # Décoder et retourner la traduction
    traduction = translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return traduction

# Charger le modèle multilingue mBART (gère plusieurs langues)
model_name = "facebook/mbart-large-50-many-to-many-mmt"

translator_tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
translator_model  = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)



# Charger le modèle complet Stable Diffusion v1-4
pipe = StableDiffusionXLLightningPipeline.from_pretrained(
    "ByteDance/SDXL-Lightning",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

pipe.enable_xformers_memory_efficient_attention()

# Exemple d'utilisation
prompt_en = traduire_texte("image d'une femme nue entrain de se mettre le doigts dans la chatte et gemir u, détaillé, réaliste, 4k")


# Génération
prompt = prompt_en

image = pipe(
    prompt,
    num_inference_steps=4,    # obligatoire Lightning
).images[0]

# Sauvegarder / afficher
image.save("output_cat_hat.png")
image.show()
