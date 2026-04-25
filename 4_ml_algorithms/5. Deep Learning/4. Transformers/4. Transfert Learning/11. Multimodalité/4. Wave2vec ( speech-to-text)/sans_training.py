from datasets import load_dataset

# On charge un petit subset du dataset Librispeech (audio + transcription)
dataset = load_dataset("librispeech_asr", "clean", split="validation[:2]")

# Affiche le premier exemple du dataset (contient l'audio + le texte réel)
print(dataset[0])


#### ou en local

import torchaudio

# Charge un fichier audio local (wav) -> retourne un tenseur et le taux d'échantillonnage
speech_array, sampling_rate = torchaudio.load("audio_test.wav")

# Affiche la forme du tenseur audio et le sampling rate (ex: (1, 16000) et 16000 Hz)
print(speech_array.shape, sampling_rate)


import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Utilise le GPU si dispo, sinon CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Le processor fait la normalisation + tokenization (convertit audio brut → input modèle)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Le modèle Wav2Vec2 pré-entraîné (fine-tuné sur 960h d'anglais parlé)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)


# On prend le premier exemple du dataset (audio + texte)
sample = dataset[0]

# Récupère l'audio sous forme de tableau numpy (signal brut)
speech = sample["audio"]["array"]


# Prétraitement : normalisation et padding, retourne un tenseur PyTorch prêt pour le modèle
inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(device)


# Inférence : pas de calcul de gradient car on fait juste de la prédiction
with torch.no_grad():
    # Le modèle transforme l'input audio en logits (scores par token du vocabulaire)
    logits = model(inputs.input_values).logits



# Prend l'index du token le plus probable à chaque étape (décodage greedy)
pred_ids = torch.argmax(logits, dim=-1)

# Convertit les indices prédits en texte lisible
transcription = processor.batch_decode(pred_ids)[0]

# Affiche le texte prédit par le modèle
print("🎧 Audio -> Texte :", transcription)

# Affiche la transcription réelle fournie par le dataset
print("📜 Transcription réelle :", sample["text"])
# Prend l'index du token le plus probable à chaque étape (décodage greedy)
pred_ids = torch.argmax(logits, dim=-1)

# Convertit les indices prédits en texte lisible
transcription = processor.batch_decode(pred_ids)[0]

# Affiche le texte prédit par le modèle
print("🎧 Audio -> Texte :", transcription)

# Affiche la transcription réelle fournie par le dataset
print("📜 Transcription réelle :", sample["text"])



from jiwer import wer, cer

# Convertit les textes en minuscule pour comparaison plus juste
true_text = sample["text"].lower()
pred_text = transcription.lower()

# Calcule le Word Error Rate (erreur par mot)
print("WER:", wer(true_text, pred_text))

# Calcule le Character Error Rate (erreur par caractère)
print("CER:", cer(true_text, pred_text))

