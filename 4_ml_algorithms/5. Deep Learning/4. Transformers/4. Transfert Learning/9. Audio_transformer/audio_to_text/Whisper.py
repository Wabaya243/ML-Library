# -------------------------
# Whisper pour transcription audio avec HuggingFace Trainer
# -------------------------

# Import des librairies
import torch
from datasets import load_dataset, load_metric, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

# -------------------------
# 0. Device & seed
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)  # pour reproductibilité

# -------------------------
# 1. Charger un dataset audio
# -------------------------
# Ici on utilise un subset de "common_voice" pour exemple
# 'fr' = français, 'train[:5]' = petit subset pour test rapide
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="train[:5]")

# Split train / val / test
train_ds = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="train[:1000]")
val_ds   = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="validation[:200]")
test_ds  = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="test[:200]")

# Vérification
print(len(train_ds), len(val_ds), len(test_ds))

#column
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
val_ds   = val_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ds  = test_ds.cast_column("audio", Audio(sampling_rate=16000))



# -------------------------
# 2. Préparer le processor Whisper
# -------------------------
# Processor gère la tokenisation du texte et la préparation des audios
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# On définit la langue cible (Whisper supporte la transcription multilingue)
language = "fr"
task = "transcribe"

# -------------------------
# 3. Preprocessing
# -------------------------
# Transformer audio en input_values et texte en labels
def preprocess(batch):
    audio = batch["audio"]["array"]
    batch["input_features"] = processor(audio, sampling_rate=16000).input_features
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

# Format torch pour Trainer
for ds in [train_ds, val_ds, test_ds]:
    ds.set_format(type="torch", columns=["input_features", "labels"])

# -------------------------
# 4. Charger le modèle Whisper
# -------------------------
# Ici Whisper-small (pré-entraîné) pour fine-tuning
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
).to(device)

# -------------------------
# 5. Définir métrique (WER)
# -------------------------
metric = load_metric("wer")

def compute_metrics(pred):
    # Transformer logits en texte
    pred_ids = pred.predictions
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # Transformer labels en texte
    label_ids = pred.label_ids
    label_text = processor.batch_decode(label_ids, skip_special_tokens=True)
    # Calcul WER
    wer = metric.compute(predictions=pred_text, references=label_text)
    return {"wer": wer}

# -------------------------
# 6. TrainingArguments pour fine-tuning
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_finetune",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,  # pour générer texte pendant l'évaluation
    fp16=True,  # accélère le training sur GPU
    logging_dir="./logs"
)

# -------------------------
# 7. Définir le Trainer
# -------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,  # petit dataset test ici, idéalement séparer train/val
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------
# 8. Lancer le fine-tuning
# -------------------------
trainer.train()

# -------------------------
# 9. Évaluation finale
# -------------------------
results = trainer.evaluate()
print("WER:", results["eval_wer"])

# -------------------------
# 10. Tester sur nouvel audio
# -------------------------
import soundfile as sf

def predict_whisper(audio_path):
    """
    Prend un fichier .wav et renvoie la transcription
    """
    audio_input, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError("Whisper attend 16kHz")
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    model.eval()
    with torch.no_grad():
        predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Exemple
# print("Transcription:", predict_whisper("./audio/test_fr.wav"))
