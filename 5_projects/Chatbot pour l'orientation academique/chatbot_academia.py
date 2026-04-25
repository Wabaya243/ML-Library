# Fine-tuning LoRA (4-bit) sur Mistral 7B Instruct pour que le model prends 4.3G en Vram que 12GB en 16bits
# Environnement : RTX 4070 Laptop (8 Go VRAM)
# Objectif : spécialiser le modèle pour le conseil académique

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch, sys, re
import transformers
import trl
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from transformers import TextStreamer
from peft import PeftModel
import pandas as pd


@dataclass
class DataCollatorAssistantOnly:
    tokenizer: object
    end_token: str = "<END>"
    inst_close: str = "[/INST]"

    def __call__(self, features):
        pad = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
        batch = pad(features)                       # input_ids, attention_mask
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, -100)   # tout masqué par défaut

        end_id = self.tokenizer.convert_tokens_to_ids(self.end_token)
        pat = self.tokenizer.encode(self.inst_close, add_special_tokens=False)

        for i, ids in enumerate(input_ids):
            ids_list = ids.tolist()

            # position du dernier [/INST]
            inst_pos = -1
            for j in range(len(ids_list) - len(pat) + 1):
                if ids_list[j:j+len(pat)] == pat:
                    inst_pos = j + len(pat)
            if inst_pos == -1:
                continue  # pas trouvé, on skippe prudemment

            # position du premier <END> APRÈS ce [/INST]
            try:
                end_pos = ids_list.index(end_id, inst_pos + 1)
            except ValueError:
                end_pos = len(ids_list)

            # démasque UNIQUEMENT la zone assistant
            if end_pos > inst_pos:
                labels[i, inst_pos:end_pos] = input_ids[i, inst_pos:end_pos]

        batch["labels"] = labels
        return batch


# Active les optimisations TF32 pour accélérer les calculs matriciels
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  # tu peux aussi activer celui-là

# Callback pour Spyder : affiche les logs d'entraînement
class SpyderFriendlyCallback(TrainerCallback): 
    def on_log(self, args, state, control, logs=None, **kwargs): 
        if logs: 
            print(logs) 
        sys.stdout.flush()

# Modèle de base : Mistral 7B Instruct v0.3
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Tokenizer et ajout du token de fin personnalisé
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = "right"

STOP_TOKEN = "<END>"
if STOP_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [STOP_TOKEN]})


# Préparation du dataset
# On applique le format de chat utilisé par Mistral
def strip_end(x): 
    return re.sub(r'\s*<END>\s*$', '', x or '').strip()

def to_chat(example):
    system = (
        "Tu es un conseiller d’orientation académique expérimenté. "
        "Réponds de façon claire, contextualisée et responsable."
    )
    user = example["instruction"]
    assistant = strip_end(example["response"]) + " <END>"
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# Chargement du fichier CSV
dataset = load_dataset("csv", data_files="Data/unikin_orientation_tagged_FIXED.csv")

# Transformation et split train/test
dataset = dataset["train"].map(to_chat)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]


# Configuration de la quantization (4-bit)
# Optimisée pour RTX 4070 Laptop (8 Go VRAM)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Chargement du modèle de base
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    use_safetensors=True
)


# Ajustement du vocabulaire pour inclure le token <END>
model.resize_token_embeddings(len(tokenizer))


# Préparation du modèle pour le fine-tuning LoRA
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = prepare_model_for_kbit_training(model)


# Configuration LoRA : attention sur les modules clés du Transformer
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)


# Masquage des labels sur la partie assistant uniquement


# Template minimal pour dire "labels = côté assistant uniquement"
response_template = tokenizer.apply_chat_template(
    [{"role":"user","content":""},{"role":"assistant","content":""}],
    tokenize=False, add_generation_prompt=True
)


collator = DataCollatorAssistantOnly(tokenizer)


model = get_peft_model(model, lora)
model.print_trainable_parameters()


# Configuration de l’entraîneur SFT (Supervised Fine-Tuning)

cfg_kwargs = dict(
    output_dir="./mistral-unikin_polis2",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    warmup_ratio=0.05,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    num_train_epochs=6,
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
    max_length=1024
)


cfg = SFTConfig(**cfg_kwargs)

model = PeftModel.from_pretrained(model, "./mistral-unikin-lora-final", is_trainable=True)  # Le LoRA existant pour le pollisage du modele
model.train() # on l'utilise que quand on a deja fait le premier entrainement et on est entrain de polir le modele pour qu'il soit moins rigide
model.print_trainable_parameters()

# Création du trainer TRL
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds, 
    eval_dataset=eval_ds,
    data_collator=collator,
    callbacks=[SpyderFriendlyCallback(),
               transformers.EarlyStoppingCallback(early_stopping_patience=3)
               ]
)

# Vérification mémoire GPU
print(f"VRAM dispo : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
print(f"VRAM utilisée : {torch.cuda.memory_allocated()/1e9:.2f} Go")

# Entraînement LoRA

model.config.use_cache = False
torch.cuda.empty_cache()

print("TRL version:", trl.__version__)
print("STOP_TOKEN id:", tokenizer.convert_tokens_to_ids(STOP_TOKEN))
print("pad_token:", tokenizer.pad_token, "id:", tokenizer.pad_token_id)
print("Model vocab size:", model.get_input_embeddings().weight.shape[0])

#lancé l'entrainement
trainer.train()

# Sauvegarde du modèle et du tokenizer
trainer.model.save_pretrained("./mistral-unikin-lora-final2")
tokenizer.save_pretrained("./mistral-unikin-lora-final2")




# PHASE DE TEST : génération d’une réponse de démonstration


base = "mistralai/Mistral-7B-Instruct-v0.3"
adapter = "./mistral-unikin-lora"

# Même config 4-bit pour la phase de test
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tok = AutoTokenizer.from_pretrained(adapter, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Chargement du modèle + adaptation LoRA
mdl = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
mdl = PeftModel.from_pretrained(mdl, adapter)
mdl.eval()

streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

# Fonction d’encodage pour les tests interactifs
def build_inputs(question: str):
    msgs = [
        {"role": "system", "content": "Tu es un conseiller d’orientation académique expérimenté. Réponds clairement, cite des pistes concrètes, reste factuel."},
        {"role": "user",   "content": question}
    ]
    chat_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(chat_str, return_tensors="pt", padding=True).to(mdl.device)
    return enc

# Exemple de génération
enc = build_inputs("Je suis en L1, j’hésite entre Info et Géo. J’aime Python, niveau maths moyen.")

with torch.no_grad():
    output = mdl.generate(
        **enc,
        max_new_tokens=220,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<END>")],
        streamer=streamer
    )

# Décodage propre du texte généré
generated_tokens = output[0][enc["input_ids"].shape[1]:]
response_text = tok.decode(generated_tokens, skip_special_tokens=True).strip()
response_text = response_text.split("<END>")[0].strip()

print("\n=== Réponse propre sans le <END>  ===\n")
print(response_text)
