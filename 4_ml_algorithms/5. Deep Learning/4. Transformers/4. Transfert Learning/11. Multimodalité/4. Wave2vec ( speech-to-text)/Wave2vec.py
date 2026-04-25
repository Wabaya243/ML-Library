import torch  # PyTorch pour le deep learning et GPU
from datasets import load_dataset, load_metric  # Hugging Face datasets et métriques
from transformers import (
    Wav2Vec2Processor,       # Prétraitement audio + tokenisation texte
    Wav2Vec2ForCTC,          # Modèle Wav2Vec2 pour transcription (CTC)
    TrainingArguments,       # Paramètres d'entraînement
    Trainer                  # Trainer Hugging Face pour fine-tuning
)
import torchaudio  # Pour charger des fichiers audio locaux


# ====================================================
# 1️⃣ Charger un dataset pour le fine-tuning
# Ici on prend Mozilla Common Voice en français ("fr")
# On prend 1% du train et du test juste pour la démo rapide
# ====================================================
dataset = load_dataset("common_voice", "fr", split={"train": "train[:1%]", "test": "test[:1%]"})

# Affiche un exemple pour voir la structure : {"audio": ..., "sentence": ...}
print(dataset["train"][0])


# ====================================================
# 2️⃣ Charger le processor
# Wav2Vec2Processor va :
# - normaliser l'audio
# - convertir le texte en tokens pour le modèle
# - gérer les labels pour CTC
# ====================================================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")


# ====================================================
# 3️⃣ Prétraitement des données
# Chaque audio doit être converti en features, chaque transcription en labels
# ====================================================
def prepare_batch(batch):
    audio = batch["audio"]  # récupère la forme d'onde et le sampling_rate

    # Transforme l'audio en "input_values" que le modèle peut traiter
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    # Convertit la transcription texte en tokens CTC
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids

    return batch


# Applique la fonction à tout le dataset et supprime les colonnes inutiles
dataset = dataset.map(prepare_batch, remove_columns=dataset["train"].column_names)


# ====================================================
# 4️⃣ Charger le modèle Wav2Vec2 pour CTC
# - Connectionist Temporal Classification (CTC) permet d'aligner l'audio et le texte
# - pad_token_id = nécessaire pour gérer les séquences de différentes longueurs
# ====================================================
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    ctc_loss_reduction="mean",  # moyenne de la perte sur le batch
    pad_token_id=processor.tokenizer.pad_token_id,
)


# ====================================================
# 5️⃣ Définir la métrique d'évaluation
# WER = Word Error Rate (taux d'erreur des mots)
# ====================================================
wer_metric = load_metric("wer")

def compute_metrics(pred):
    # Prédiction : récupère l'id du token avec la probabilité max
    pred_ids = torch.argmax(pred.predictions, dim=-1)

    # Convertit les ids en texte
    pred_str = processor.batch_decode(pred_ids)

    # Labels : convertit les ids en texte pour comparaison
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Calcule le WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ====================================================
# 6️⃣ Paramètres d'entraînement
# - group_by_length=True regroupe les audios similaires pour efficacité
# - fp16 pour accélérer sur GPU
# ====================================================
training_args = TrainingArguments(
    output_dir="./wav2vec2-ft-fr",
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)


# ====================================================
# 7️⃣ Création du Trainer Hugging Face
# data_collator = crée un batch PyTorch à partir des input_values et labels
# ====================================================
trainer = Trainer(
    model=model,
    data_collator=lambda data: {
        "input_values": torch.tensor([f["input_values"] for f in data], dtype=torch.float32),
        "labels": torch.tensor([f["labels"] for f in data], dtype=torch.long),
    },
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)


# ====================================================
# 8️⃣ Fine-tuning du modèle
# ====================================================
trainer.train()


# ====================================================
# 9️⃣ Sauvegarde du modèle fine-tuné
# ====================================================
model.save_pretrained("./wav2vec2-ft-fr")
processor.save_pretrained("./wav2vec2-ft-fr")


# ====================================================
# 10️⃣ Tester le modèle fine-tuné sur un audio local
# ====================================================
# Charger le modèle fine-tuné
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-ft-fr").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-ft-fr")

# Charger un fichier audio local
speech_array, sampling_rate = torchaudio.load("audio_test.wav")

# Prétraitement
inputs = processor(speech_array[0], sampling_rate=sampling_rate, return_tensors="pt", padding=True).to("cuda")

# Inférence
with torch.no_grad():
    logits = model(inputs.input_values).logits

# Prédiction
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]

print("Texte prédit :", transcription)
