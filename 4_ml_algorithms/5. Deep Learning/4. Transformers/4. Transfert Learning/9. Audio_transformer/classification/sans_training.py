# -------------------------
# Wav2Vec2 pour prédiction audio directe (sans fine-tuning)
# -------------------------

import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import soundfile as sf

# -------------------------
# 0. Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. Charger un dataset existant (ex: Speech Commands)
# -------------------------
# Ce dataset contient des fichiers audio .wav avec des labels simples ("yes", "no", ...)
dataset = load_dataset("speech_commands", split="test[:5]")  # prendre un petit subset pour test

# Liste des labels
labels = dataset.features["label"].names  # ex: ["backward","bed","bird",...]

# -------------------------
# 2. Charger Wav2Vec2 pré-entraîné
# -------------------------
# Ici on prend un modèle général pré-entraîné pour la reconnaissance de caractéristiques audio
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=len(labels),
    problem_type="single_label_classification"
).to(device)

# -------------------------
# 3. Fonction de prédiction directe
# -------------------------
def predict_audio(audio_array, sampling_rate=16000):
    """
    Prend un audio numpy array et prédit sa classe avec Wav2Vec2 sans fine-tuning.
    """
    # Si audio non à 16kHz, il faut ré-échantillonner
    if sampling_rate != 16000:
        raise ValueError("Wav2Vec2 attend 16kHz audio.")
    
    # Préparer input_values
    inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # Prédiction
    model.eval()
    with torch.no_grad():
        logits = model(inputs).logits
        pred_idx = torch.argmax(logits, dim=-1).item()
    return labels[pred_idx]

# -------------------------
# 4. Tester sur dataset existant
# -------------------------
for sample in dataset:
    audio_array, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
    pred = predict_audio(audio_array, sr)
    print(f"Gold label: {labels[sample['label']]} | Predicted: {pred}")


#### teste sur audio exerieur

import soundfile as sf
audio_array, sr = sf.read("mon_audio.wav")
print("Classe prédite:", predict_audio(audio_array, sr))
