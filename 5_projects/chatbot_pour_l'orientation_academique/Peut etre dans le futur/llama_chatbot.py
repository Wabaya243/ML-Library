# ==========================================================
# Fine-tuning LoRA (4-bit) sur Meta-Llama-3.1-8B-Instruct
# Environnement : RTX 4070 Laptop (8 Go VRAM)
# Objectif : conseiller académique (UNIKIN)
# ==========================================================

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TextStreamer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import torch, sys, re, transformers
from collator import DataCollatorAssistantOnlyLLamma

import os, warnings, gc, time, psutil

# Environnement (tu peux laisser HF en ligne si ton cache local n'a pas le modèle)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TELEMETRY"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings("ignore", message=".*h5py is running against HDF5.*")

# ========================= Callbacks perfs =========================
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
        if steps % 100 == 0 and steps > 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        vram_cur = torch.cuda.memory_allocated() / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        ram_free = psutil.virtual_memory().available / 1e9
        cpu = psutil.cpu_percent(interval=0.0)
        print(f"[step {steps}] {spd:.2f} steps/s | VRAM {vram_cur:.2f}G (peak {vram_peak:.2f}G) | "
              f"RAM free {ram_free:.1f}G | CPU {cpu:.0f}%")

class EpochCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n🧹 [Cleanup] Fin de l’époque {state.epoch:.2f} — libération mémoire GPU...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"   VRAM utilisée après nettoyage : {mem:.2f} Go\n")

class SpyderFriendlyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs: print(logs)
        sys.stdout.flush()

# ========================= Optimisations GPU =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ========================= Modèle de base (LLaMA 3.1) =========================
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# ========================= Nettoyage & Troncature =========================
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
    resp_raw = safe_str(example.get("response", ""))

    user = truncate_text(user_raw, 1500)
    assistant = truncate_text(strip_end_tag(resp_raw), 1800)

    msgs = [
        {"role": "system",
         "content": "Tu es un conseiller d’orientation académique expérimenté. "
                    "Réponds clairement, contextualise pour l’UNIKIN et propose des démarches concrètes."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text, truncation=True, max_length=540)
    return {"text": tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)}

dataset = load_dataset("csv", data_files="Data/unikin_orientation_tagged_FIXED.csv")
dataset = dataset.filter(lambda x: bool(x["instruction"]) and bool(x["response"]))
dataset = dataset["train"].map(to_chat, remove_columns=dataset["train"].column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]

# ========================= Quantization 4-bit =========================
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ========================= Chargement + préparation k-bit =========================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    use_safetensors=True
)

# Implémentation attention (SDPA = safe par défaut, Flash si lib installée)
model.config.attn_implementation = "sdpa"
model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

# ========================= LoRA (attn + MLP) =========================
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",      # Attention LLaMA
        "gate_proj","up_proj","down_proj"         # MLP LLaMA
    ]
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ========================= Trainer TRL (SFT) =========================
cfg_kwargs = dict(
    output_dir="./llama31-8b-unikin",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.05,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=100,
    num_train_epochs=4,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
    optim="paged_adamw_32bit",
    dataset_text_field="text",
    max_grad_norm=0.3,
    label_smoothing_factor=0.03,
    eval_accumulation_steps=8,
    packing=False,
    group_by_length=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

# Compat TRL
if hasattr(SFTConfig, "__dataclass_fields__"):
    if "max_seq_length" in SFTConfig.__dataclass_fields__:
        cfg_kwargs["max_seq_length"] = 540
    else:
        cfg_kwargs["max_length"] = 540
elif hasattr(SFTConfig, "__fields__"):
    if "max_seq_length" in SFTConfig.__fields__:
        cfg_kwargs["max_seq_length"] = 540
    else:
        cfg_kwargs["max_length"] = 540
else:
    cfg_kwargs["max_length"] = 540

cfg = SFTConfig(**cfg_kwargs)
collator = DataCollatorAssistantOnlyLLamma(tokenizer)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[SpyderFriendlyCallback(),
               EpochCleanupCallback(),
               PerfAndMem(),
               transformers.EarlyStoppingCallback(early_stopping_patience=3)]
)

# ========================= Logs VRAM =========================
print(f"VRAM dispo : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
print(f"VRAM utilisée : {torch.cuda.memory_allocated()/1e9:.2f} Go")
print("pad_token:", tokenizer.pad_token, "id:", tokenizer.pad_token_id)

# ========================= Entraînement =========================
model.config.use_cache = False
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM libre juste avant train : {torch.cuda.mem_get_info()[0]/1e9:.2f} Go")
trainer.train(resume_from_checkpoint=True)

# ========================= Sauvegarde adapter LoRA + tokenizer =========================
adapter_out = "./llama31-8b-unikin-lora-final"
trainer.model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)

"""
# ========================= PHASE DE TEST =========================
base = model_name
adapter = adapter_out

tok = AutoTokenizer.from_pretrained(adapter_out, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

mdl = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, adapter)
mdl.eval()

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

def build_inputs(question: str):
    msgs = [
        {"role": "system",
         "content": "Tu es un conseiller d’orientation académique expérimenté. "
                    "Réponds clairement, contextualise pour l'UNIKIN, propose des démarches concrètes."},
        {"role": "user", "content": question}
    ]
    chat_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(chat_str, return_tensors="pt", padding=True).to(mdl.device)

enc = build_inputs("Je suis en L1, j’hésite entre Info et Géo. J’aime Python, niveau maths moyen.")
with torch.no_grad():
    output = mdl.generate(
        **enc,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        streamer=streamer
    )

generated_tokens = output[0][enc["input_ids"].shape[1]:]
response_text = tok.decode(generated_tokens, skip_special_tokens=True).strip()
print("\n=== Réponse (nettoyée) ===\n")
print(response_text)
"""
