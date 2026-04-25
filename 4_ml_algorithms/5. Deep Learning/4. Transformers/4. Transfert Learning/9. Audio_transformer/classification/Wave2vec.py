# -------------------------
# Wav2Vec2 pour classification audio avec HuggingFace Trainer
# -------------------------

# Import des librairies
import torch
import numpy as np
from datasets import Dataset as HFDataset, load_metric, Audio, load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
import soundfile as sf

# -------------------------
# 0. Device & seed
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)  # reproductibilité

# -------------------------
# 1. Dataset exemple (custom)
# -------------------------
# Ici on crée un petit dataset toy pour illustrer
# dataset = [{"audio_path": "path/to/file.wav", "label": 0}, ...]
# labels = ["class_0", "class_1", "class_2", ...]
dataset_list = [
    {"audio_path": "./audio/class0_1.wav", "label": 0},
    {"audio_path": "./audio/class0_2.wav", "label": 0},
    {"audio_path": "./audio/class1_1.wav", "label": 1},
    {"audio_path": "./audio/class1_2.wav", "label": 1},
]

labels = ["class_0", "class_1"]


# -------------------------
# 1. Charger un dataset existant (ex: Speech Commands)
# -------------------------
# Ce dataset contient des fichiers audio .wav avec des labels simples ("yes", "no", ...)
dataset = load_dataset("speech_commands", split="test[:5]")  # prendre un petit subset pour test

# Liste des labels
labels = dataset.features["label"].names  # ex: ["backward","bed","bird",...]


# -------------------------
# 2. Convertir en HF Dataset et définir audio feature
# -------------------------
hf_dataset = HFDataset.from_list(dataset)
#Sur speech_commands, la colonne s’appelle "audio" et non "audio_path".
hf_dataset = hf_dataset.cast_column("audio_path", Audio(sampling_rate=16000))  # Wav2Vec2 attend 16kHz


# -------------------------
# 3. Charger feature extractor Wav2Vec2
# -------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# normalise l'audio et transforme en input_values pour le modèle

# -------------------------
# 4. Préparer le preprocessing
# -------------------------
def preprocess(batch):
    # charge l'audio, le normalise et transforme en input_values
    #Sur speech_commands, la colonne s’appelle "audio" et non "audio_path".
    audio = batch["audio_path"]["array"]
    batch["input_values"] = feature_extractor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = batch["label"]
    return batch

hf_dataset = hf_dataset.map(preprocess)

# Définir le format des colonnes pour Trainer
hf_dataset.set_format(type="torch", columns=["input_values", "labels"])

# -------------------------
# 5. Charger le modèle Wav2Vec2 pour classification
# -------------------------
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(labels),
    problem_type="single_label_classification"
).to(device)

# -------------------------
# 6. Metrics (accuracy)
# -------------------------
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels_tensor = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels_tensor)

# -------------------------
# 7. TrainingArguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./wav2vec2_audio_trainer",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# -------------------------
# 8. Définir Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    eval_dataset=hf_dataset,  # ici dataset toy, pour vrai dataset séparer train/val
    compute_metrics=compute_metrics
)

# -------------------------
# 9. Lancer le fine-tuning (optionnel)
# -------------------------

trainer.train()

# -------------------------
# 10. Évaluation finale
# -------------------------
# results = trainer.evaluate()
# print("Accuracy:", results["eval_accuracy"])

# -------------------------
# 11. Tester sur nouvel audio direct
# -------------------------
def predict_audio(audio_path):
    """
    Prend un fichier .wav, applique le feature extractor et prédit sa classe.
    """
    # Charger audio
    audio_input, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError("Wav2Vec2 attend 16kHz audio. Re-échantillonner avant.")
    # Convertir en input_values
    inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    # Prédiction
    model.eval()
    with torch.no_grad():
        logits = model(inputs).logits
        pred_idx = torch.argmax(logits, dim=-1).item()
    pred_class = labels[pred_idx]
    return pred_class

# Exemple d'utilisation
# print("Classe prédite:", predict_audio("./audio/class0_1.wav"))
