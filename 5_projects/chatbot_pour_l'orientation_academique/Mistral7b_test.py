# Test d’un modèle LoRA fine-tuné sur Mistral-7B-Instruct-v0.3
# Objectif : conseiller académique UNIKIN (RTX 4070 Laptop 8GB)

import torch, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

# 1. Configuration de base

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./mistral-7B-unikin-lora"

# Quantization 4-bit optimisée pour éviter les OOM (Out Of Memory) sur 8Go
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 2. Tokenizer (Chargé depuis l'adaptateur pour conserver les modifs)


tok = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)

# Alignement strict avec votre script d'entraînement Mistral
if tok.pad_token is None:
    tok.pad_token = tok.unk_token  # unk_token = <unk> (id=0)

tok.padding_side = "right"

print("Tokenizer chargé :", tok.name_or_path)
print("EOS token :", tok.eos_token, " (id:", tok.eos_token_id, ")")
print("PAD token :", tok.pad_token, " (id:", tok.pad_token_id, ")")


# 3. Chargement du modèle + LoRA

mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_cfg,
    device_map="auto",
    use_safetensors=True
)

# Forcer l'implémentation SDPA (Scaled Dot Product Attention) pour la rapidité et l'économie de VRAM
mdl.config.attn_implementation = "sdpa"

# Pas de redimensionnement de vocabulaire requis pour Mistral v0.3 standard
mdl = PeftModel.from_pretrained(mdl, ADAPTER_PATH)
mdl.eval()

print("Adaptateurs actifs :", mdl.active_adapters)

# 4. Gestion de la conversation (Chat-template Mistral v0.3)


streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

history = [
    {
        "role": "system",
        "content": (
            "Tu es un conseiller d'orientation académique expérimenté. "
            "Réponds clairement, contextualise pour l'UNIKIN et propose des démarches concrètes. "
            "Tu t'exprimes dans un français fluide et bienveillant. "
            "Tes réponses doivent être claires, longues, et expliquer le raisonnement."
        )
    }
]

MAX_PAIRS = 7

def reset_history():
    global history
    history = [history[0]]
    print("Historique réinitialisé.")

def _trim(hist, max_pairs=MAX_PAIRS):
    sys_msg = hist[0]
    pairs = []
    dialog = hist[1:]
    i = 0
    while i + 1 < len(dialog):
        if dialog[i]["role"] == "user" and dialog[i + 1]["role"] == "assistant":
            pairs.append((dialog[i], dialog[i + 1]))
        i += 1
    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]
    new_hist = [sys_msg]
    for u, a in pairs:
        new_hist += [u, a]
    if len(dialog) >= 1 and dialog[-1]["role"] == "user":
        if new_hist[-1]["role"] != "user":
            new_hist.append(dialog[-1])
    return new_hist

def ask(question: str, max_new_tokens=450):
    global history
    history.append({"role": "user", "content": question})
    
    # Génération du prompt selon le format natif de Mistral v0.3
    chat_str = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
    enc = tok(chat_str, return_tensors="pt").to(mdl.device)

    print("\n=== PROMPT ENVOYÉ ===\n")
    print(chat_str)
    print("=====================\n")

    with torch.no_grad():
        output = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # Légère créativité pour des explications fluides et longues
            temperature=0.7,
            top_p=0.90,
            repetition_penalty=1.12,  # Évite les boucles répétitives de phrases
            eos_token_id=tok.eos_token_id, # Jeton d'arrêt natif de Mistral </s>
            pad_token_id=tok.pad_token_id,
            streamer=streamer
        )

    gen = output[0][enc["input_ids"].shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": resp})
    history = _trim(history, MAX_PAIRS)
    print(f"\n=== RÉPONSE NETTOYÉE ===\n{resp}\n")

# 5. Démonstration (Séquence de test)

reset_history()

ask("Bonjour, je viens d’obtenir mon diplôme d’État et j’aimerais comprendre les différentes étapes pour m’inscrire à l’Université de Kinshasa, notamment les documents à préparer, les délais et les erreurs à éviter pour ne pas rater la rentrée.")
ask("Je suis passionné par la biologie mais j’ai peur de ne pas avoir le niveau scientifique nécessaire. Est-ce que je peux quand même réussir si je travaille dur, et quelles sont les meilleures stratégies à adopter dès la première année ?")
ask("Peux-tu me donner un exemple de plan d’étude équilibré pour un étudiant en sciences humaines qui veut aussi apprendre à programmer ?")
ask("Comment puis-je gérer mon temps entre les cours, le travail et la vie personnelle sans tomber dans la fatigue ou la procrastination ?")
ask("J’aimerais que tu m’expliques comment un étudiant timide peut améliorer sa communication orale et participer davantage aux cours magistraux.")
ask("refflechis et dis moi c'est quoi la premiere la question que je t'ai posé")

# 6. Sauvegarde de l'historique


with open("Data/history_mistral.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print("\n Historique Mistral sauvegardé avec succès.")