# ==========================================================
# Test d’un modèle LoRA fine-tuné sur LLaMA 3.1 8B Instruct
# ==========================================================

import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

# ----------------------------------------------------------
# 1. Configuration de base
# ----------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./llama3.1-8b-unikin-lora-final"   # dossier LoRA fine-tuné

# Quantization 4-bit (optimisée RTX 4070 8 Go)
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ----------------------------------------------------------
# 2. Tokenizer (celui du LoRA)
# ----------------------------------------------------------

tok = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Vérifie la présence du token de fin de tour officiel LLaMA
eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
print(f" Token <|eot_id|> trouvé : id = {eot_id}")

# ----------------------------------------------------------
# 3. Chargement du modèle + LoRA
# ----------------------------------------------------------

mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_cfg,
    device_map="auto"
)

# Ajuste la taille du vocabulaire si besoin
base_vocab = mdl.get_input_embeddings().weight.shape[0]
if len(tok) != base_vocab:
    print(f"Ajustement vocabulaire: {base_vocab} → {len(tok)}")
    mdl.resize_token_embeddings(len(tok))

mdl = PeftModel.from_pretrained(mdl, ADAPTER_PATH)
mdl.eval()

print("Adaptateurs actifs :", mdl.active_adapters)
print("PEFT keys :", getattr(mdl, "peft_config", {}).keys())

# ----------------------------------------------------------
# 4. Outils de conversation
# ----------------------------------------------------------

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

history = [
    {
        "role": "system",
        "content": (
            "Tu es un conseiller académique de l’Université de Kinshasa. "
            "Réponds avec bienveillance et précision, en proposant des pistes concrètes et adaptées."
        )
    }
]

MAX_PAIRS = 6  # nb max de paires user/assistant

def reset_history():
    global history
    history = [{
        "role": "system",
        "content": (
            "Tu es un conseiller académique bienveillant et explicatif."
        )
    }]
    print("Historique réinitialisé.")

def _enforce_alternation(hist):
    """Force alternance user/assistant."""
    sys_msg = hist[0]
    dialog = [m for m in hist[1:] if m["role"] in ("user","assistant")]
    cleaned = [sys_msg]
    for m in dialog:
        if cleaned[-1]["role"] == m["role"]:
            cleaned[-1] = m
        else:
            cleaned.append(m)
    # drop leading assistant après system
    while len(cleaned) >= 2 and cleaned[1]["role"] == "assistant":
        cleaned.pop(1)
    return cleaned

def _trim(hist, max_pairs=MAX_PAIRS):
    """Garde max_pairs paires user/assistant."""
    hist = _enforce_alternation(hist)
    sys_msg = hist[0]
    dialog = hist[1:]
    pairs = []
    i = 0
    while i + 1 < len(dialog):
        if dialog[i]["role"] == "user" and dialog[i+1]["role"] == "assistant":
            pairs.append((dialog[i], dialog[i+1]))
            i += 2
        else:
            i += 1
    if len(pairs) > max_pairs:
        pairs = pairs[-max_pairs:]
    new_hist = [sys_msg]
    for u,a in pairs:
        new_hist += [u,a]
    if len(dialog) >= 1 and dialog[-1]["role"] == "user":
        if new_hist[-1]["role"] != "user":
            new_hist.append(dialog[-1])
    return new_hist

def ask(question: str):
    """Ajoute la question, génère une réponse et met à jour l’historique."""
    global history
    history.append({"role": "user", "content": question})
    history = _enforce_alternation(history)

    # Construction du prompt LLaMA
    chat_str = tok.apply_chat_template(
        history, add_generation_prompt=True, tokenize=False
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
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tok.convert_tokens_to_ids("<|eot_id|>"),
            streamer=streamer
        )

    gen = output[0][enc["input_ids"].shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": resp})
    history = _trim(history, MAX_PAIRS)
    print(f"\nAssistant : {resp}\n")

# ----------------------------------------------------------
# 5. Démonstration
# ----------------------------------------------------------

reset_history()
ask("Mbote, comment choisir une filière à l’UNIKIN ?")
ask("Si j’aime l’informatique mais j’ai peur des maths, c’est grave ?")
ask("Tu peux me rappeler mes deux choix de filière ?")
ask("Quel conseil donnerais-tu à un étudiant timide qui veut réussir à l’université ?")
ask("Et comment savoir si je suis fait pour la géographie ?")

# ----------------------------------------------------------
# 6. Sauvegarde et rechargement de l’historique
# ----------------------------------------------------------

with open("Data/history_llama.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

with open("Data/history_llama.json", "r", encoding="utf-8") as f:
    history = json.load(f)

print("\n💾 Historique sauvegardé et rechargé avec succès.")
