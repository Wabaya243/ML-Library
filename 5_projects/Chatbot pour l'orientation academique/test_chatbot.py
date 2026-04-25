# ==========================================================
# Test d’un modèle LoRA fine-tuné sur Mistral-7B-Instruct
# ==========================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
import json

# ----------------------------------------------------------
# 1. Configuration de base
# ----------------------------------------------------------

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./mistral-unikin-lora-final2"   # ton dossier LoRA fine-tuné

# Configuration de quantisation 4-bit (optimisée RTX 4070 8 Go)
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ----------------------------------------------------------
# 2. Tokenizer : charge celui du LoRA (important)
# ----------------------------------------------------------

tok = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Vérifie que le token <END> existe
end_id = tok.convert_tokens_to_ids("<END>")
if end_id is None or end_id == 0:
    print("  Le token <END> est introuvable dans le tokenizer.")
else:
    print(f" Token <END> trouvé : id = {end_id}")

# ----------------------------------------------------------
# 3. Chargement du modèle + LoRA
# ----------------------------------------------------------

mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_cfg,
    device_map="auto"
)

# Ajuste la taille du vocabulaire avant de charger le LoRA
base_vocab = mdl.get_input_embeddings().weight.shape[0]
if len(tok) != base_vocab:
    print(f" Ajustement vocabulaire: {base_vocab} → {len(tok)}")
    mdl.resize_token_embeddings(len(tok))

mdl = PeftModel.from_pretrained(mdl, ADAPTER_PATH)
mdl.eval()

print("Adaptateurs actifs :", mdl.active_adapters)
print("Clés PEFT :", getattr(mdl, "peft_config", {}).keys())

import torch

layer = mdl.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"]
print("Moyenne :", torch.mean(layer.weight).item())
print("Écart-type :", torch.std(layer.weight).item())
print("Max :", torch.max(layer.weight).item())
print("Min :", torch.min(layer.weight).item())


# ----------------------------------------------------------
# 4. Utilitaires de conversation
# ----------------------------------------------------------

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

history = [
    {
        "role": "system",
        "content": (
            "Tu es un conseiller d’orientation académique expérimenté.  "
            "Réponds toujours à la dernière question de l'utilisateur "
            "en tenant compte du contexte précédent, sans inventer de phrases hors sujet, Réponds de façon claire, contextualisée et responsable et explicatif"
        )
    }
]


MAX_PAIRS = 7  # = 6 tours user+assistant (12 messages hors system)

def reset_history():
    """Réinitialise proprement: 1 message system, rien d'autre."""
    global history
    history = [{
        "role": "system",
        "content": ("Tu es un conseiller d’orientation académique expérimenté. "
                    "Réponds toujours à la dernière question de l'utilisateur "
                    "en tenant compte du contexte précédent, sans inventer, Réponds de façon claire, contextualisée et responsable et explicatif")
    }]

def _ensure_single_system(hist):
    # garde le premier 'system', supprime les autres
    sys_msg = next((m for m in hist if m["role"] == "system"), None)
    if sys_msg is None:
        sys_msg = {"role": "system", "content": ("Tu es un conseiller d’orientation académique expérimenté. "
                    "Réponds toujours à la dernière question de l'utilisateur "
                    "en tenant compte du contexte précédent, sans inventer, Réponds de façon claire, contextualisée et responsable et explicatif")
                   }
        
    # on ne garde que les messages non-system puis on re-préfixe
    non_sys = [m for m in hist if m["role"] != "system"]
    return [sys_msg] + non_sys

def _enforce_alternation(hist):
    """Après le (optionnel) system, force l’alternance user/assistant/user/assistant..."""
    hist = _ensure_single_system(hist)
    cleaned = [hist[0]]  # system
    for m in hist[1:]:
        if cleaned[-1]["role"] == m["role"]:
            # Si doublon de rôle, on remplace le dernier par le nouveau (on garde le plus récent)
            cleaned[-1] = m
        else:
            cleaned.append(m)
    # Si ça commence par assistant après system, on drop jusqu’à tomber sur user
    while len(cleaned) >= 2 and cleaned[1]["role"] == "assistant":
        cleaned.pop(1)
    return cleaned

def _trim_to_max_pairs(hist, max_pairs=MAX_PAIRS):
    """Garde au plus max_pairs paires user/assistant après le system, en retirant la plus ancienne paire."""
    hist = _enforce_alternation(hist)
    sys_msg = hist[0]
    dialog = hist[1:]

    # On veut compter des paires complètes (user, assistant)
    # Si le dernier est 'user' (pas encore de réponse), on ne le compte pas comme paire complète
    pairs = []
    i = 0
    while i + 1 < len(dialog):
        if dialog[i]["role"] == "user" and dialog[i+1]["role"] == "assistant":
            pairs.append((dialog[i], dialog[i+1]))
            i += 2
        else:
            # Si on tombe sur une rupture, on tente de resynchroniser (retire l’élément courant)
            i += 1

    # Fenêtrage
    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]

    # Reconstruit l’historique
    new_hist = [sys_msg]
    for u, a in pairs:
        new_hist.append(u)
        new_hist.append(a)

    # Si le tout dernier message d’origine était un user non répondu, on peut le ré-attacher
    if len(dialog) >= 1 and dialog[-1]["role"] == "user":
        # uniquement si le dernier stocké n’est pas déjà un user
        if new_hist[-1]["role"] != "user":
            new_hist.append(dialog[-1])

    return new_hist

def ask(question: str):
    """Ajoute la question, prépare le prompt propre, génère, puis tronque à 6 paires."""
    global history
    # Ajout du user
    history.append({"role": "user", "content": question})
    # Nettoyage & alternance
    history = _enforce_alternation(history)

    # Construction du prompt (toujours finir par un user pour add_generation_prompt=True)
    # Si le dernier n’est pas 'user', on ajoute un user vide pour respecter le template
    if history[-1]["role"] != "user":
        # petite sécurité: on n’ajoute pas de faux user, on retire plutôt le dernier assistant orphelin
        if history[-1]["role"] == "assistant":
            history.pop()

    chat_str = tok.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=False
    )

    enc = tok(chat_str, return_tensors="pt").to(mdl.device)

    print("\n=== PROMPT ENVOYÉ ===\n")
    print(chat_str)
    print("=====================\n")

    with torch.no_grad():
        output = mdl.generate(
            **enc,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.70,
            top_p=0.8,
            repetition_penalty=1.5,
            eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<END>")],
            streamer=streamer
        )

    gen = output[0][enc["input_ids"].shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True).split("<END>")[0].strip()

    # Push assistant + trim à 6 paires
    history.append({"role": "assistant", "content": resp})
    history = _trim_to_max_pairs(history, MAX_PAIRS)

    print(f"\nAssistant : {resp}\n")



# ----------------------------------------------------------
# 5. Démonstration
# ----------------------------------------------------------

reset_history()

ask("bot mbote boni ?")
ask("yo bot quoi de neuf ?")
ask("Salut, j’hésite entre 1. informatique et 2. géographie.")
ask("parfois j'ai envie de faire la geoscience pour observer et parfois l'informatique pour etre derriere la machine")
ask("donc je dois chosisr quoi ?")
ask("Et si je n’aime pas trop les maths, que devrais-je choisir entre le 1 et le 2 ?")
ask("Comment savoir si je suis fort en quelque chose ?")
ask("Est-ce que tu te souviens de ce que je viens de demander ?")
ask("Peux-tu me rappeler mon premier choix entre informatique et géographie ?")
ask("Je suis en L1, j’hésite entre Info et Géo. J’aime Python, niveau maths moyen.")
ask("toi tu peux me dire quelle sont les points fort pour choisir l'un des ces deux matiere ?")
ask("comment ça ?")
ask("t'es sur ?")
ask("bon essaie un peu d'etre plus reflechis cette fois")
ask("tu fais quoi comme ça?")
ask("t'as commencé a ecrire en anglais merde")
ask("alors qui avait ecrit cette ligne ? ?[/INST] I’m still training, but my current goal is: Help students understand their potential, make informed decisions about future studies, and provide guidance on choosing electives that align with personal interests and career goals.</s>[INST]")
ask("t'es bete")
ask("tu oses repondres alors que tu dis des betises ?")

# ----------------------------------------------------------
# 6. Sauvegarde et rechargement de l’historique
# ----------------------------------------------------------

with open("Data/history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

with open("Data/history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

print("\n💾 Historique sauvegardé et rechargé avec succès.")
