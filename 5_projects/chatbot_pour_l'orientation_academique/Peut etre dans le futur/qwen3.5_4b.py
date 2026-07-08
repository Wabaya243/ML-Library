# ========
# LoRA 4-bit — Qwen3-8B (UNIKIN)
# RTX 4070 Laptop 8GB — rapide et stable
# ========

import os, warnings, gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["HF_TOKEN"] = "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"

warnings.filterwarnings("ignore", message=".*h5py is running against HDF5.*")

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorWithPadding, TextStreamer, TrainerCallback
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training, PeftModel
)
from trl import SFTTrainer, SFTConfig
import re, sys, transformers
import torch
import time
import psutil
from dataclasses import dataclass


# ---------------- Collator : n'entraîner QUE la zone assistant ----------------
# Qwen3 garde le même chat template que Qwen2.5 : <|im_start|> / <|im_end|>
# CORRECTION : on boucle sur TOUS les tours assistant (pas seulement le dernier)
# + fix du masque sur end_matches (taille L-pat_len+1, pas L)
@dataclass
class DataCollatorAssistantOnly:
    tokenizer: object

    def __call__(self, features):
        pad = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            return_tensors="pt"
        )

        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # masque tout par défaut
        labels[:] = -100

        # tokens spéciaux Qwen
        im_start = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end   = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        B, L = input_ids.shape

        for b in range(B):
            ids = input_ids[b]

            # trouver tous les assistant blocks
            i = 0
            while i < L:
                if ids[i].item() == im_start:
                    # vérifier si c'est "assistant"
                    # (assistant header = im_start + "assistant")
                    if i + 1 < L:
                        next_token = self.tokenizer.decode([ids[i+1].item()])
                        if "assistant" in next_token:
                            start = i

                            # chercher fin
                            for j in range(i, L):
                                if ids[j].item() == im_end:
                                    labels[b, start:j+1] = input_ids[b, start:j+1]
                                    i = j
                                    break
                i += 1

        batch["labels"] = labels
        return batch


#  Callbacks 
class PerfAndMem(TrainerCallback):
    def __init__(self):
        self.t0        = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        steps = state.global_step
        dt    = time.time() - self.t0 if steps > self.last_step else 0.0
        spd   = (steps - self.last_step) / max(dt, 1e-6) if steps > self.last_step else 0.0
        self.t0        = time.time()
        self.last_step = steps

        if steps % 50 == 0 and steps > 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        vram_cur  = torch.cuda.memory_allocated()     / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        ram_free  = psutil.virtual_memory().available / 1e9
        cpu       = psutil.cpu_percent(interval=0.0)

        print(f"[step {steps}] {spd:.2f} steps/s | VRAM {vram_cur:.2f}G "
              f"(peak {vram_peak:.2f}G) | RAM free {ram_free:.1f}G | CPU {cpu:.0f}%")


class EpochCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n🧹 [Cleanup] Fin époque {state.epoch:.2f} — libération mémoire GPU...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"   VRAM après nettoyage : {mem:.2f} Go\n")


class SpyderFriendlyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(logs)
        sys.stdout.flush()


#  Optimisations GPU 
torch.backends.cuda.matmul.allow_tf32  = True
torch.backends.cudnn.allow_tf32  = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.cuda.empty_cache()


#  Modèle de base 
model_name = "Qwen/Qwen3.5-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side    = "right"
tokenizer.truncation_side = "right"


#  Dataset 
def safe_str(x):
    if not isinstance(x, str) or x is None or x.strip() == "" or x == "nan":
        return ""
    return x

def truncate_text(text: str, max_len: int = 2000):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
        text = text.rsplit(" ", 1)[0] + "..."
    return text

def strip_end_tag(x: str):
    if not isinstance(x, str):
        return ""
    x = re.sub(r'\s*<END>\s*$', '', x.strip())
    return x

def to_chat(example):
    user_raw  = safe_str(example.get("instruction", ""))
    resp_raw  = safe_str(example.get("response",    ""))

    user      = truncate_text(user_raw,               1000)
    assistant = truncate_text(strip_end_tag(resp_raw), 1500)

    msgs = [
        {"role": "system",
         "content": "Tu es un conseiller d'orientation académique expérimenté. "
                    "Réponds clairement, contextualise pour l'UNIKIN et propose des démarches concrètes."},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant}
    ]

    # enable_thinking=False : réponses directes, pas de CoT pendant le fine-tuning
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False       # ← spécifique Qwen3
    )
    encoded = tokenizer(text, truncation=True, max_length=540)
    return {"text": tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)}


dataset = load_dataset("csv", data_files="Data/unikin_orientation_tagged_FIXED.csv")
dataset = dataset.filter(lambda x: bool(x["instruction"]) and bool(x["response"]))
dataset = dataset["train"].map(to_chat, remove_columns=dataset["train"].column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]


#  Quantization 4-bit 
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

#  Chargement modèle 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    use_safetensors=True
)

print(model.hf_device_map)

model.config.attn_implementation = "sdpa"
print(model.config._attn_implementation)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)


#  LoRA 
# Qwen3 architecture identique à Qwen2.5 → mêmes modules cibles
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()


#  SFTConfig 
cfg_kwargs = dict(
    output_dir="./qwen3.5-4b-unikin",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    eval_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=20,
    num_train_epochs=4,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=100,
    bf16=True,
    optim="paged_adamw_32bit",
    dataset_text_field="text",
    max_grad_norm=0.3,
    label_smoothing_factor=0.05,
    packing=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    gradient_checkpointing=True
)

if hasattr(SFTConfig, "__dataclass_fields__"):
    key = "max_seq_length" if "max_seq_length" in SFTConfig.__dataclass_fields__ else "max_length"
elif hasattr(SFTConfig, "__fields__"):
    key = "max_seq_length" if "max_seq_length" in SFTConfig.__fields__ else "max_length"
else:
    key = "max_length"
cfg_kwargs[key] = 540

cfg = SFTConfig(**cfg_kwargs)

collator = DataCollatorAssistantOnly(tokenizer)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[
        SpyderFriendlyCallback(),
        EpochCleanupCallback(),
        PerfAndMem(),
        transformers.EarlyStoppingCallback(early_stopping_patience=3)
    ]
)

#  Logs VRAM 
print(f"VRAM dispo    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
print(f"VRAM utilisée : {torch.cuda.memory_allocated()/1e9:.2f} Go")
print("pad_token:", tokenizer.pad_token, "id:", tokenizer.pad_token_id)

#  Entraînement 
model.config.use_cache = False
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM libre avant train : {torch.cuda.mem_get_info()[0]/1e9:.2f} Go")


#  VALIDATION DU COLLATOR 
print("\n" + "=" * 70)
print("VALIDATION DU MASQUAGE — vérifie qu'on entraîne bien sur l'assistant")
print("=" * 70)

# Simule ce que SFTTrainer fait : tokenise le texte d'abord
sample_texts = [train_ds[0]["text"], train_ds[1]["text"]]
sample_batch = [
    tokenizer(t, truncation=True, max_length=720) for t in sample_texts
]
test_out = collator(sample_batch)

for i in range(len(sample_batch)):
    ids = test_out["input_ids"][i]
    labels = test_out["labels"][i]

    learned_mask = labels != -100
    learned_text = tokenizer.decode(ids[learned_mask], skip_special_tokens=False)
    masked_text = tokenizer.decode(ids[~learned_mask], skip_special_tokens=False)

    print(f"\n--- Exemple {i} ---")
    print(f" ZONE APPRISE:\n   {learned_text[:300]}")
    print(f"\n ZONE MASQUÉE (fin):\n   {masked_text[-200:]}")
    print(f"\nTokens appris: {learned_mask.sum().item()} / {len(ids)}")

print("=" * 70 + "\n")
input("Entrée pour continuer vers trainer.train()...")

trainer.train()

#  Sauvegarde 
adapter_out = "./qwen3.5-4b-unikin-lora"
trainer.model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)



r'''
# ========
# PHASE DE TEST — Chat interactif avec historique (comme Qwen2.5)
# ========

import json

BASE_MODEL   = model_name
ADAPTER_PATH = adapter_out

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ── Tokenizer depuis l'adapter (contient le chat template sauvegardé) ──
tok = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

print("Tokenizer chargé :", tok.name_or_path)
print("EOS token :", tok.eos_token,  " (id:", tok.eos_token_id, ")")
print("PAD token :", tok.pad_token,  " (id:", tok.pad_token_id, ")")

# Extraction stricte du token de fin de tour Qwen3
try:
    IM_END_TOKEN_ID = tok.convert_tokens_to_ids("<|im_end|>")
    print("Jeton d'arrêt Qwen détecté (<|im_end|>) -> id:", IM_END_TOKEN_ID)
except Exception:
    IM_END_TOKEN_ID = tok.eos_token_id
    print("Attention: <|im_end|> non trouvé, utilisation de eos_token_id:", IM_END_TOKEN_ID)

# ── Rechargement propre du modèle pour l'inférence ──
mdl = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant,
    device_map="auto",
    use_safetensors=True
)
mdl.config.attn_implementation = "sdpa"
mdl = PeftModel.from_pretrained(mdl, ADAPTER_PATH)
mdl.eval()

print("Adaptateurs actifs :", mdl.active_adapters)

# ── Streamer ──
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

# ── Historique multi-tour ──
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
    sys_msg  = hist[0]
    dialog   = hist[1:]
    pairs    = []
    i        = 0
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

def ask(question: str, max_new_tokens: int = 450):
    global history
    history.append({"role": "user", "content": question})

    # enable_thinking=False → réponses directes en inférence aussi
    chat_str = tok.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False       # ← spécifique Qwen3
    )
    enc = tok(chat_str, return_tensors="pt").to(mdl.device)

    print("\n=== PROMPT ENVOYÉ ===\n")
    print(chat_str)
    print("=====================\n")

    with torch.no_grad():
        output = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.90,
            repetition_penalty=1.12,        # bloque les répétitions en français
            eos_token_id=IM_END_TOKEN_ID,   # arrêt propre sans halluciner l'user
            pad_token_id=tok.pad_token_id,
            streamer=streamer
        )

    gen  = output[0][enc["input_ids"].shape[1]:]
    resp = tok.decode(gen, skip_special_tokens=True).strip()

    history.append({"role": "assistant", "content": resp})
    history = _trim(history, MAX_PAIRS)
    print(f"\n=== RÉPONSE NETTOYÉE ===\n{resp}\n")


# ── Démonstration : séquence de test de cohérence ──
reset_history()

ask("Bonjour, je viens d'obtenir mon diplôme d'État et j'aimerais comprendre les différentes étapes "
    "pour m'inscrire à l'Université de Kinshasa, notamment les documents à préparer, les délais "
    "et les erreurs à éviter pour ne pas rater la rentrée.")

ask("Je suis passionné par la biologie mais j'ai peur de ne pas avoir le niveau scientifique nécessaire. "
    "Est-ce que je peux quand même réussir si je travaille dur, et quelles sont les meilleures stratégies "
    "à adopter dès la première année ?")

ask("Peux-tu me donner un exemple de plan d'étude équilibré pour un étudiant en sciences humaines "
    "qui veut aussi apprendre à programmer ?")

ask("Comment puis-je gérer mon temps entre les cours, le travail et la vie personnelle "
    "sans tomber dans la fatigue ou la procrastination ?")

ask("J'aimerais que tu m'expliques comment un étudiant timide peut améliorer sa communication orale "
    "et participer davantage aux cours magistraux.")

ask("refflechis et dis moi c'est quoi la premiere question que je t'ai posé")


# ── Sauvegarde JSON de l'historique ──
with open("Data/history_qwen3.json", "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print("\nHistorique Qwen3-8B sauvegardé avec succès.")

'''