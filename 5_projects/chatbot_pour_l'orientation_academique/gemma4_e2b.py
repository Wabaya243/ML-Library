# LoRA 4-bit — gemma-4-E4B-it-text-only (UNIKIN)
# RTX 4070 Laptop 8GB — adapté pour le checkpoint text-only de principled-intelligence
# Avantage : pas d'encodeurs vision/audio → AutoTokenizer suffit, device_map simplifié

import os, warnings, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


# Chat template Gemma 4 (identique au modèle original) :
# <start_of_turn>user\nmsg<end_of_turn>\n
# <start_of_turn>model\nmsg<end_of_turn>
@dataclass
class DataCollatorAssistantOnly:
    tokenizer: object

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)

        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)
        B, L = input_ids.size()

        # Template réel de ce checkpoint :
        # <|turn> = 105, "model" = 4368, \n = 107
        # Le marqueur assistant est [105, 4368, 107]
        start_tensor = torch.tensor([105, 4368, 107], dtype=torch.long, device=input_ids.device)
        pat_len = 3

        for b in range(B):
            match_idx = -1
            for idx in range(L - pat_len + 1):
                if torch.equal(input_ids[b, idx:idx + pat_len], start_tensor):
                    match_idx = idx  # garde le DERNIER match (multi-turn safe)

            if match_idx == -1:
                continue  # pas de tour assistant trouvé → tout masqué

            # s pointe sur le premier token de la réponse assistant
            s = match_idx + pat_len

            non_pad_indices = torch.where(input_ids[b] != self.tokenizer.pad_token_id)[0]
            e = non_pad_indices[-1].item() + 1 if len(non_pad_indices) > 0 else L

            if s < e:
                labels[b, s:e] = input_ids[b, s:e]

        batch["labels"] = labels
        return batch    


class PerfAndMem(TrainerCallback):
    def __init__(self):
        self.t0 = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        steps = state.global_step
        dt = time.time() - self.t0 if steps > self.last_step else 0.0
        spd = (steps - self.last_step) / max(dt, 1e-6) if steps > self.last_step else 0.0
        self.t0 = time.time()
        self.last_step = steps

        if steps % 50 == 0 and steps > 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        vram_cur  = torch.cuda.memory_allocated() / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        ram_free  = psutil.virtual_memory().available / 1e9
        cpu       = psutil.cpu_percent(interval=0.0)

        print(f"[step {steps}] {spd:.2f} steps/s | VRAM {vram_cur:.2f}G "
              f"(peak {vram_peak:.2f}G) | RAM free {ram_free:.1f}G | CPU {cpu:.0f}%")


class EpochCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n[Cleanup] Fin époque {state.epoch:.2f} — libération mémoire GPU...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM après nettoyage : {mem:.2f} Go\n")


class SpyderFriendlyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(logs)
        sys.stdout.flush()


#  Optimisations GPU ─
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

#  Modèle 
# Checkpoint text-only : pas d'encodeurs vision/audio, AutoTokenizer suffit
model_name = "principled-intelligence/gemma-4-E2B-it-text-only"

# AutoTokenizer au lieu de AutoProcessor (pas de vision à gérer)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side    = "right"
tokenizer.truncation_side = "right"

SYSTEM_PROMPT = (
    "Tu es un conseiller d'orientation académique expérimenté. "
    "Réponds clairement, contextualise pour l'UNIKIN et propose des démarches concrètes."
)

#  Dataset ─
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
    user_raw = safe_str(example.get("instruction", ""))
    resp_raw = safe_str(example.get("response",    ""))

    user      = truncate_text(user_raw,               1000)
    assistant = truncate_text(strip_end_tag(resp_raw), 1500)

    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant}
    ]

    # apply_chat_template via le tokenizer (identique au template original Gemma 4)
    # enable_thinking=False : réponses directes sans chain-of-thought
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    encoded = tokenizer(text, truncation=True, max_length=520)
    return {"text": tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)}


dataset = load_dataset("csv", data_files="Data/unikin_orientation_tagged_FIXED.csv")
dataset = dataset.filter(lambda x: bool(x["instruction"]) and bool(x["response"]))
dataset = dataset["train"].map(to_chat, remove_columns=dataset["train"].column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]

#  Quantization 4-bit 
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

free, total = torch.cuda.mem_get_info()
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM totale : {total/1024**3:.2f} GB | libre : {free/1024**3:.2f} GB")

#  Chargement du modèle 
# text-only → device_map simple, plus besoin de mapper vision/audio sur CPU
# E4B (~7.52B params text-only) tient en 4-bit sur 8GB avec gradient checkpointing
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
     device_map={"": 0},  # GPU 0
    use_safetensors=True,
    attn_implementation="sdpa"  # scaled dot-product attention (efficace)
)

print(f"Implémentation attention : {model.config._attn_implementation}")
model.config.use_cache = False

# Gradient checkpointing pour économiser la VRAM
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# NOTE : unwrap_gemma_for_lora (Gemma4ClippableLinear) N'EST PLUS NÉCESSAIRE
# Le checkpoint text-only ne contient pas les couches vision → pas de ClippableLinear

#  LoRA 
# E4B-it-text-only : language_model avec 35 layers (même architecture que E4B original)
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora)

r'''
print("\n=== Modules LoRA injectés ===")
for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        print(name)
print("=" * 50)
'''

model.print_trainable_parameters()

#  SFTConfig 
cfg_kwargs = dict(
    output_dir="./gemma4-e2b-text-only-unikin",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=20,
    num_train_epochs=4,
    save_steps=100,
    save_total_limit=3,
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
    gradient_checkpointing=True,
)

if hasattr(SFTConfig, "__dataclass_fields__"):
    key = "max_seq_length" if "max_seq_length" in SFTConfig.__dataclass_fields__ else "max_length"
elif hasattr(SFTConfig, "__fields__"):
    key = "max_seq_length" if "max_seq_length" in SFTConfig.__fields__ else "max_length"
else:
    key = "max_length"
cfg_kwargs[key] = 520

cfg = SFTConfig(**cfg_kwargs)

collator = DataCollatorAssistantOnly(tokenizer)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    processing_class=tokenizer,
    callbacks=[
        SpyderFriendlyCallback(),
        EpochCleanupCallback(),
        PerfAndMem(),
        transformers.EarlyStoppingCallback(early_stopping_patience=3)
    ]
)

print(f"\nVRAM dispo    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
print(f"VRAM utilisée : {torch.cuda.memory_allocated()/1e9:.2f} Go")
print(f"pad_token     : {tokenizer.pad_token!r}  id={tokenizer.pad_token_id}")


#  Validation du collator 
print("\n" + "=" * 70)
print("VALIDATION DU MASQUAGE — vérifie qu'on entraîne bien sur l'assistant")
print("=" * 70)

sample_texts = [train_ds[0]["text"], train_ds[1]["text"]]
sample_batch = [
    tokenizer(t, truncation=True, max_length=620) for t in sample_texts
]
test_out = collator(sample_batch)

for i in range(len(sample_batch)):
    ids    = test_out["input_ids"][i]
    labels = test_out["labels"][i]
    learned_mask = labels != -100
    learned_text = tokenizer.decode(ids[learned_mask],  skip_special_tokens=False)
    masked_text  = tokenizer.decode(ids[~learned_mask], skip_special_tokens=False)
    print(f"\n--- Exemple {i} ---")
    print(f"  ZONE APPRISE :\n   {learned_text[:300]}")
    print(f"\n  ZONE MASQUÉE (extrait fin) :\n   {masked_text[-200:]}")
    print(f"\n  Tokens appris : {learned_mask.sum().item()} / {len(ids)}")

print("=" * 70 + "\n")
input("Entrée pour lancer trainer.train()...")

#  Entraînement 
model.config.use_cache = False
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM libre avant train : {torch.cuda.mem_get_info()[0]/1e9:.2f} Go")

trainer.train()

#  Sauvegarde adapter LoRA + tokenizer 
adapter_out = "./gemma4-e2b-text-only-unikin-lora"
trainer.model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)
print(f"\nAdapter sauvegardé dans : {adapter_out}")


#  PHASE DE TEST 
r"""
base    = model_name
adapter = adapter_out

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tok = AutoTokenizer.from_pretrained(adapter)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, adapter)
mdl.eval()

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

SYSTEM_PROMPT = (
    "Tu es un conseiller d'orientation académique expérimenté. "
    "Réponds clairement, contextualise pour l'UNIKIN, propose des démarches concrètes."
)

def build_inputs(question: str):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question}
    ]
    chat_str = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False
    )
    return tok(chat_str, return_tensors="pt", padding=True).to(mdl.device)

enc = build_inputs("Je suis en L1, j'hésite entre Info et Géo. J'aime Python, niveau maths moyen.")
with torch.no_grad():
    output = mdl.generate(
        **enc,
        max_new_tokens=220,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        repetition_penalty=1.05,
        eos_token_id=tok.convert_tokens_to_ids("<end_of_turn>"),
        pad_token_id=tok.pad_token_id,
        streamer=streamer
    )

generated_tokens = output[0][enc["input_ids"].shape[1]:]
response_text = tok.decode(generated_tokens, skip_special_tokens=True).strip()
print("\n=== Réponse (nettoyée) ===\n")
print(response_text)
"""