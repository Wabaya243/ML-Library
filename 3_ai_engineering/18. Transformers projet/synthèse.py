from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import evaluate
import torch
import os
from transformers import DataCollatorForSeq2Seq

# =========================
# 1. CHARGEMENT DU DATASET
# =========================
try:
    # Charger le dataset WikiLingua en français (déjà format parquet, prêt à l'emploi)
    dataset = load_dataset("esdurmus/wiki_lingua", "french")
    print("Exemple du jeu d'entraînement :", dataset["train"][0])

except Exception as e:
    print("Erreur lors du chargement du dataset :", e)
    exit()


print(dataset["train"][0].keys())  # clés du dict 'article' ou autres colonnes
print(dataset["train"][0]["article"].keys())



# =========================
# 2. CHARGEMENT DU TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# =========================
# 3. FONCTION DE TOKENISATION
# =========================
# --- tokenisation corrigée ---
def tokenize_function(examples):
    inputs = []
    targets = []
    # examples["article"] est une liste de dicts (batch)
    for article in examples["article"]:
        doc = article["document"]
        if isinstance(doc, list):
            text = " ".join(doc)
        else:
            text = doc
        inputs.append("summarize: " + text)

        summ = article.get("summary", "")
        if isinstance(summ, list):
            summaries = " ".join(summ)
        else:
            summaries = summ or ""
        targets.append(summaries)

    # tokeniser les entrées (pas de padding forcé ici)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=False)

    # tokeniser les cibles / labels en forçant padding à max_length
    # (ici 150, comme tu veux)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    # Remplacer pad_token_id par -100 pour la loss
    label_ids = labels["input_ids"]
    label_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in label_ids
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


#=========================
# 4. CHARGEMENT DU MODÈLE
# =========================
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

# Tokenisation du dataset
# map (batched=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# --- data collator seq2seq (gère labels et padding dynamiquement) ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)


# =========================
# 5. PARAMÈTRES D'ENTRAÎNEMENT
# =========================

from transformers.integrations import TensorBoardCallback

# assure-toi que les dossiers existent (chemins absolus pour être sûr)
os.makedirs("./results", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
logging_dir = os.path.abspath("./logs")
output_dir = os.path.abspath("./results")

# Ajuste TrainingArguments — évite args non-supportés par ta version
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",   # si ta version supporte eval_strategy (tu as adapté avant)
    save_strategy="steps",
    save_steps=500,
    logging_dir=logging_dir,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    # Important : désactive toutes les intégrations (tensorboard, wandb, etc.)
    report_to="none",
    # no_cuda est déprécié ; tu peux garder use_cpu si besoin
    no_cuda=not torch.cuda.is_available(),)





# =========================
# 6. CRÉATION DU TRAINER
# =========================

# Exemple : créer un split validation manuellement si absent
if "validation" not in tokenized_datasets and "test" not in tokenized_datasets:
    # on split le train en train + validation (par ex 90%/10%)
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]  # ici test = validation car issue du split
else:
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"]





trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)



# Optionnel mais sûr : s'il y a encore un callback TensorBoard, on le retire
try:
    trainer.remove_callback(TensorBoardCallback)
except Exception:
    # méthode alternative : filtrer la liste des callbacks
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if cb.__class__.__name__ != "TensorBoardCallback"
    ]

# =========================
# 7. ENTRAÎNEMENT
# =========================
trainer.train()

# =========================
# 8. SAUVEGARDE DU MODÈLE
# =========================
save_dir = "./trained_model"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)       # sauvegarde des poids du modèle
tokenizer.save_pretrained(save_dir)   # sauvegarde du tokenizer
print(f" Modèle et tokenizer sauvegardés dans : {save_dir}")


# =========================
# 9. TEST MANUEL DU MODÈLE
# =========================
sample_text = """
Les modèles Transformers ont révolutionné le domaine du traitement automatique du langage naturel (NLP).
Ils permettent un traitement parallèle des séquences, améliorant ainsi les performances sur diverses tâches,
comme la traduction automatique, la classification de texte, et la synthèse. Leur architecture basée sur
l’attention permet de mieux capturer les relations contextuelles longues dans les textes.
"""

inputs = tokenizer("summarize: " + sample_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

print("Synthèse générée :", tokenizer.decode(outputs[0], skip_special_tokens=True))

# =========================
# 10. ÉVALUATION AUTOMATIQUE
# =========================
metric = evaluate.load('rouge')

predictions = []
references = []
eval_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"]

for example in eval_dataset:
    input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
    generated_ids = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    predictions.append(pred)

    # Nettoyage des labels (-100 = padding)
    label_ids = [id for id in example["labels"] if id != -100]
    ref = tokenizer.decode(label_ids, skip_special_tokens=True)
    references.append(ref)

results = metric.compute(predictions=predictions, references=references, use_stemmer=True)
print("Scores ROUGE :", results)


# =========================
# 11. EXPORT EN ONNX
# =========================
onnx_path = "./trained_model/model.onnx"

# Exemple d'entrée fictive à encoder
dummy_input = tokenizer("summarize: Exemple de texte", return_tensors="pt")

# Désactiver le cache pour éviter l'erreur liée au cache d'attention
model.config.use_cache = False

# Entrée fictive encodeur
dummy_encoder_input = dummy_input["input_ids"]

# Attention mask pour encodeur (1 partout pour les tokens non-paddés)
dummy_encoder_attention_mask = torch.ones_like(dummy_encoder_input)

# Entrée fictive décodeur (longueur arbitraire 10)
dummy_decoder_input = torch.tensor([[tokenizer.pad_token_id]*10])

# Attention mask décodeur (idem)
dummy_decoder_attention_mask = torch.ones_like(dummy_decoder_input)

# Export ONNX avec les 4 entrées nécessaires
torch.onnx.export(
    model,
    (dummy_encoder_input, dummy_encoder_attention_mask, dummy_decoder_input, dummy_decoder_attention_mask),
    onnx_path,
    input_names=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "decoder_input_ids": {0: "batch", 1: "sequence"},
        "decoder_attention_mask": {0: "batch", 1: "sequence"}
    },
    opset_version=14,
    do_constant_folding=True,
)

print(f" Modèle exporté en ONNX : {onnx_path}")





        ####################
######### pour charger les models ##################


# Chemin vers le dossier sauvegardé
save_dir = "./trained_model"

# Charger modèle et tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(save_dir)

# Test rapide
text = "summarize: Le deep learning révolutionne l'IA."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))





