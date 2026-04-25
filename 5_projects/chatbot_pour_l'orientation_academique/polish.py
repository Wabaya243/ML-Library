# ==========================================================
# POLISH — Fine-tuning LoRA (4-bit) continuation
# Modèle : DeepSeek-R1-Distill-Qwen-7B
# Objectif : raffiner le conseiller académique UNIKIN
# ==========================================================

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TextStreamer, TrainerCallback
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch, os, gc, psutil, time, sys, warnings, re
from collator import DataCollatorAssistantOnly

# ----------------------------------------------------------
# Config de base
# ----------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", message=".*h5py.*")

# ----------------------------------------------------------
# Callbacks légers
# ----------------------------------------------------------
class PerfAndMem(TrainerCallback):
    def __init__(self):
        self.t0 = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args, state, control, logs=None, **kwargs):
        steps = state.global_step
        dt = time.time() - self.t0 if steps > self.last_step else 0
        spd = (steps - self.last_step) / max(dt, 1e-6) if steps > self.last_step else 0
        self.t0 = time.time()
        self.last_step = steps
        if steps % 25 == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        vram = torch.cuda.memory_allocated() / 1e9
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        ram = psutil.virtual_memory().available / 1e9
        print(f"[step {steps}] {spd:.2f} steps/s | VRAM {vram:.2f}G (peak {vram_peak:.2f}G) | RAM free {ram:.1f}G")

class EpochCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n Fin epoch {state.epoch:.2f} — libération mémoire...")
        torch.cuda.empty_cache()
        gc.collect()

# ----------------------------------------------------------
# Modèle
# ----------------------------------------------------------
base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
adapter_path = "./deepseek-qwen7b-unikin-lora-final"  # ton LoRA précédent
output_dir = "./deepseek-qwen7b-unikin-polish-long"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant,
    device_map="auto",
    use_safetensors=True
)

# Important: préparer AVANT d’injecter LoRA
base = prepare_model_for_kbit_training(base)

# Optionnels mais utiles
base.config.use_cache = False
base.config.pretraining_tp = 1  # évite certains ralentissements SDPA

# Checkpointing & grads entrée pour VRAM et backward correct
base.enable_input_require_grads()

# Injecter LoRA APRES la préparation k-bit
model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
model.enable_adapter_layers()  # réactive les modules LoRA
model.train()

# Sanity check: afficher ce qui est vraiment entraînable
model.print_trainable_parameters()

model.config.use_cache = False

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------
def safe_str(x): return x if isinstance(x, str) and x.strip() != "" else ""

def truncate_text(txt, max_len=2000):
    txt = safe_str(txt).strip()
    return txt[:max_len].rsplit(" ", 1)[0] + "..." if len(txt) > max_len else txt

def strip_end_tag(x): return re.sub(r'\s*<END>\s*$', '', safe_str(x).strip())

def to_chat(example):
    user = truncate_text(example["instruction"], 1500)
    assistant = truncate_text(strip_end_tag(example["response"]), 2000)
    msgs = [
        {"role": "system", "content": "Tu es un conseiller d’orientation académique pour l’UNIKIN."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    enc = tokenizer(text, truncation=True, max_length=1024)
    return {"text": tokenizer.decode(enc["input_ids"], skip_special_tokens=False)}

dataset = load_dataset("csv", data_files="Data/unikin_orientation_tagged_FIXED.csv")["train"]
dataset = dataset.filter(lambda x: bool(x["instruction"]) and bool(x["response"]))
dataset = dataset.map(to_chat, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]

# ----------------------------------------------------------
# Training config (polish)
# ----------------------------------------------------------
cfg = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    warmup_ratio=0.02,
    weight_decay=0.02,
    lr_scheduler_type="cosine",
    logging_steps=100,
    num_train_epochs=3,
    eval_strategy="steps",
    save_steps=300,
    eval_steps=300,
    bf16=True,
    optim="paged_adamw_32bit",
    dataset_text_field="text",
    label_smoothing_factor=0.05,
    max_length=1024,
    group_by_length=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False
)

#collator = DataCollatorAssistantOnly(tokenizer)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[PerfAndMem(), EpochCleanupCallback()]
)

print(f"VRAM dispo : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
print(f"VRAM utilisée : {torch.cuda.memory_allocated()/1e9:.2f} Go")
print(f"pad_token : {tokenizer.pad_token}")

torch.cuda.empty_cache()
print(f"VRAM libre avant train : {torch.cuda.mem_get_info()[0]/1e9:.2f} Go")

# ----------------------------------------------------------
# Lancement du polissage
# ----------------------------------------------------------
trainer.train()

# ----------------------------------------------------------
# Sauvegarde
# ----------------------------------------------------------
adapter_out = "./deepseek-qwen7b-unikin-lora-final2"
trainer.model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)
print(f"\n Polissage terminé — modèle sauvegardé dans {adapter_out}")
