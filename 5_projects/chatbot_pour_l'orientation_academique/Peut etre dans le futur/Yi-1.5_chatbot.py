# ==========================================================
# Fine-tuning LoRA (4-bit) sur Yi-1.5-9B-Chat
# Environnement : RTX 4070 Laptop (8 Go VRAM)
# Objectif : conseiller académique (UNIKIN)
# ==========================================================

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorWithPadding, TextStreamer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
import torch, sys, re, transformers, trl
from collator import DataCollatorAssistantOnlyYI


# ========================= Optimisations GPU =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SpyderFriendlyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs: print(logs)
        sys.stdout.flush()

# ========================= Modèle de base =========================
model_name = "01-ai/Yi-1.5-9B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# Yi a souvent pad_token=None → on le cale sur eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ========================= Dataset (CSV: instruction,response) =========================
def strip_end_tag(x: str):
    if not isinstance(x, str): return ""
    # ton corpus termine par <END> → on le retire
    return re.sub(r'\s*<END>\s*$', '', x.strip())

def to_chat(example):
    msgs = [
        {"role": "system",
         "content": "Tu es un conseiller d’orientation académique expérimenté. "
                    "Réponds clairement, contextualise pour l’UNIKIN et propose des démarches concrètes."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": strip_end_tag(example["response"])}
    ]
    # IMPORTANT : Yi-1.5-Chat a un chat_template compatible apply_chat_template
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = load_dataset("csv", data_files="Data/unikin_orientation_tagged_FIXED.csv")
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

model.enable_xformers_memory_efficient_attention()
model.config.attn_implementation = "flash_attention_2"

print(model.config._attn_implementation)

model.config.use_cache = False

#model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = prepare_model_for_kbit_training(model)

# ========================= LoRA (attn + MLP) =========================
# Yi est archi proche LLaMA/Mistral côté noms de modules
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",      # Attention
        "gate_proj","up_proj","down_proj"         # MLP
    ]
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ========================= Trainer TRL (SFT) =========================
cfg_kwargs = dict(
    output_dir="./yi-1_5-9b-unikin",
    per_device_train_batch_size=4,      # 8 Go VRAM -> 1
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_ratio=0.05,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    num_train_epochs=3,                 # 3-4 si dataset pas énorme
    save_steps=250,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=50,
    bf16=True,
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    max_grad_norm=0.3,
    label_smoothing_factor=0.05,
    eval_accumulation_steps=16,
    packing=False,
    max_length=768,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)


cfg = SFTConfig(**cfg_kwargs)
collator = DataCollatorAssistantOnlyYI(tokenizer)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[SpyderFriendlyCallback(),
               transformers.EarlyStoppingCallback(early_stopping_patience=3)]
)

# ========================= Logs VRAM =========================
print(f"VRAM dispo : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
print(f"VRAM utilisée : {torch.cuda.memory_allocated()/1e9:.2f} Go")
print("TRL version:", trl.__version__)
print("pad_token:", tokenizer.pad_token, "id:", tokenizer.pad_token_id)

# ========================= Entraînement =========================
model.config.use_cache = False
torch.cuda.empty_cache()
trainer.train()

# ========================= Sauvegarde adapter LoRA + tokenizer =========================
adapter_out = "./yi-1_5-9b-unikin-lora-final"
trainer.model.save_pretrained(adapter_out)
tokenizer.save_pretrained(adapter_out)

# ========================= PHASE DE TEST =========================
base = model_name
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
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
        streamer=streamer
    )

generated_tokens = output[0][enc["input_ids"].shape[1]:]
response_text = tok.decode(generated_tokens, skip_special_tokens=True).strip()
print("\n=== Réponse (nettoyée) ===\n")
print(response_text)
