
# LoRA 4-bit — Lucie-7B-Instruct (OpenLLM-France) — UNIKIN
# RTX 4070 Laptop 8GB — rapide et stable
#
# Modèle francophone, architecture Llama-like standard (PAS de
# sliding window interleavée -> pas de piège de lenteur Gemma/Ministral).
# Chat template identique à Llama 3.1, sauf <|begin_of_text|> -> <s>.
# Pas gated, accès libre sur HuggingFace.


import os, warnings, gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["DISABLE_TELEMETRY"] = "1"

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


#-  Collator: n'entraîner QUE la zone assistant# Lucie-7B-Instruct utilise le format Llama 3.1:
# <s><|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>
# <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>\n\n...<|eot_id|>
@dataclass
class DataCollatorAssistantOnly:
    tokenizer: object
    sample_text_for_detection: str = None

    def __post_init__(self):
        if self.sample_text_for_detection is None:
            raise ValueError("Passe sample_text_for_detection (texte réel apply_chat_template).")

        text = self.sample_text_for_detection

        candidates = [
            "<|start_header_id|> assistant<|end_header_id|>\n\n",  # Version exacte générée par Lucie
            "<|start_header_id|> assistant<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>",
        ]
        marker_found = None
        for cand in candidates:
            if cand in text:
                marker_found = cand
                break

        if marker_found is None:
            raise ValueError(
                f"Aucun marker assistant connu trouvé. Fin du texte: ...{text[-300:]}"
            )

        self.assistant_marker = marker_found
        print(f"[Collator Lucie] Marker détecté: {marker_found!r}")

        marker_pos = text.rfind(marker_found)
        context_start = max(0, marker_pos - 20)
        probe_text = text[context_start:marker_pos] + marker_found

        probe_ids = self.tokenizer.encode(probe_text, add_special_tokens=False)
        context_ids = self.tokenizer.encode(
            text[context_start:marker_pos], add_special_tokens=False
        )
        marker_ids = probe_ids[len(context_ids):]
        if len(marker_ids) == 0:
            raise ValueError("Échec extraction IDs du marker.")

        self._marker_ids = torch.tensor(marker_ids, dtype=torch.long)
        print(f"[Collator Lucie] Marker IDs: {marker_ids} -> "
              f"{self.tokenizer.decode(marker_ids)!r}")

    def _find_last_match(self, seq, pattern):
        pat_len = pattern.size(0)
        if seq.size(0) < pat_len:
            return -1
        windows = seq.unfold(0, pat_len, 1)
        matches = (windows == pattern).all(dim=1)
        idxs = torch.where(matches)[0]
        if len(idxs) == 0:
            return -1
        return idxs[-1].item()

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)

        B = input_ids.size(0)
        pad_id = self.tokenizer.pad_token_id

        for b in range(B):
            seq = input_ids[b]
            match_idx = self._find_last_match(seq, self._marker_ids)
            if match_idx == -1:
                continue
            start_tok_idx = match_idx + self._marker_ids.size(0)
            if start_tok_idx >= seq.size(0):
                continue
            labels[b, start_tok_idx:] = seq[start_tok_idx:]
            pad_mask = seq == pad_id
            labels[b][pad_mask] = -100

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
        print(f"\n [Cleanup] Fin époque {state.epoch:.2f} — libération mémoire GPU...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"   VRAM après nettoyage : {mem:.2f} Go\n")


class SpyderFriendlyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs: print(logs)
        sys.stdout.flush()


#  Optimisations GPU 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


#  Modèle de base 
model_name = "OpenLLM-France/Lucie-7B-Instruct-v1.1"  # pas gated, accès libre

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side   = "right"
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
    return re.sub(r'\s*<END>\s*$', '', x.strip())

def to_chat(example):
    user_raw = safe_str(example.get("instruction", ""))
    resp_raw = safe_str(example.get("response",    ""))

    user      = truncate_text(user_raw,               1000)
    assistant = truncate_text(strip_end_tag(resp_raw), 1500)

    msgs = [
        {"role": "system",
         "content": "Tu es un conseiller d'orientation académique expérimenté. "
                    "Réponds clairement, contextualise pour l'UNIKIN et propose des démarches concrètes."},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant}
    ]

    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(text, truncation=True, max_length=720, add_special_tokens=False)
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
    use_safetensors=True,
    attn_implementation="sdpa"
)

print(model.config._attn_implementation)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)


#  LoRA 
# Architecture Llama-like -> mêmes target_modules que Llama/Mistral/Qwen
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


#  SFTConfig / Trainer 
cfg_kwargs = dict(
    output_dir="./lucie-7b-unikin",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    num_train_epochs=4,
    save_steps=300,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=300,
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
cfg_kwargs[key] = 720

cfg = SFTConfig(**cfg_kwargs)

collator = DataCollatorAssistantOnly(
    tokenizer,
    sample_text_for_detection=train_ds[0]["text"]
)

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

# Entraînement 
model.config.use_cache = False
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM libre avant train : {torch.cuda.mem_get_info()[0]/1e9:.2f} Go")

#  VALIDATION DU COLLATOR 
print("\n" + "=" * 70)
print("VALIDATION DU MASQUAGE — vérifie qu'on entraîne bien sur l'assistant")
print("=" * 70)

sample_texts = [train_ds[0]["text"], train_ds[1]["text"]]
sample_batch = [
    tokenizer(t, truncation=True, max_length=720, add_special_tokens=False)
    for t in sample_texts
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

trainer.train(resume_from_checkpoint=True)
#trainer.train()

# ========================= Sauvegarde =========================
adapter_out = "./lucie-7b-unikin-lora"
trainer.model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)


# ========================= PHASE DE TEST =========================
r"""
base    = model_name
adapter = adapter_out

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, adapter)
mdl.eval()

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

def build_inputs(question: str):
    msgs = [
        {"role": "system",
         "content": "Tu es un conseiller d'orientation académique expérimenté. "
                    "Réponds clairement, contextualise pour l'UNIKIN, propose des démarches concrètes."},
        {"role": "user", "content": question}
    ]
    chat_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(chat_str, return_tensors="pt", padding=True).to(mdl.device)

enc = build_inputs("Je suis en L1, j'hésite entre Info et Géo. J'aime Python, niveau maths moyen.")
with torch.no_grad():
    output = mdl.generate(
        **enc,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=tok.convert_tokens_to_ids("<|eot_id|>"),
        pad_token_id=tok.pad_token_id,
        streamer=streamer
    )

generated_tokens = output[0][enc["input_ids"].shape[1]:]
response_text = tok.decode(generated_tokens, skip_special_tokens=True).strip()
print("\n=== Réponse (nettoyée) ===\n")
print(response_text)
"""