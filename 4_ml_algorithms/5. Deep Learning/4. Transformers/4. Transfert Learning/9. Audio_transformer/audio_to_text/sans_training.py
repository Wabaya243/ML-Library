# -------------------------
# Whisper pour transcription audio sans fine-tuning
# -------------------------

# Import des librairies
import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

# -------------------------
# 0. Device & seed
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# -------------------------
# 1. Charger le dataset Speech Commands (subset pour test)
# -------------------------
# Dataset anglais par défaut, pour test simple
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split="validation[:5]")

# On cast la colonne audio pour que Whisper puisse lire
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# -------------------------
# 2. Charger le processor et le modèle Whisper
# -------------------------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

# On définit la langue et la tâche (Whisper supporte multilingue)
language = "fr"  
task = "transcribe"

# -------------------------
# 3. Fonction de prédiction directe
# -------------------------
def predict_whisper(audio_array, sr=16000):
    """
    Prend un array audio numpy (ou chemin .wav) et renvoie la transcription avec Whisper.
    """
    # Si input est un chemin, on lit le fichier
    if isinstance(audio_array, str):
        audio_array, sr = sf.read(audio_array)

    # Vérifier la fréquence d'échantillonnage
    if sr != 16000:
        raise ValueError("Whisper attend 16kHz. Ré-échantillonner l'audio.")

    # Préparer input
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Prédiction
    model.eval()
    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    # Décoder en texte
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# -------------------------
# 4. Tester sur le dataset de validation
# -------------------------
print("=== Transcriptions sur Speech Commands validation ===")
for example in dataset:
    audio_array = example["audio"]["array"]
    expected_label = example["sentence"]  # ex: "yes", "no", ...
    transcription = predict_whisper(audio_array)
    print(f"Attendu: {expected_label} | Whisper: {transcription}")

# -------------------------
# 5. Tester sur un audio custom
# -------------------------
# Exemple fichier wav
# chemin_audio = "./audio/test_yes.wav"
# print("Transcription custom:", predict_whisper(chemin_audio))
