#### Partie 1️ : Import des librairies et configuration
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
torchaudio.set_audio_backend("soundfile")

########## Partie 2️ : Charger le dataset
# Charger le dataset SpeechCommands (mini)
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)
        def load_list(filename):
            with open(filename) as f:
                return [os.path.join(self._path, line.strip()) for line in f]
        if subset == "validation":
            self._walker = load_list(os.path.join(self._path, "validation_list.txt"))
        elif subset == "testing":
            self._walker = load_list(os.path.join(self._path, "testing_list.txt"))
        elif subset == "training":
            excludes = load_list(os.path.join(self._path, "validation_list.txt")) + \
                       load_list(os.path.join(self._path, "testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]

train_dataset = SubsetSC("training")
test_dataset = SubsetSC("testing")

# Construire la liste unique des labels dans le dataset
labels_set = sorted(
    entry.name for entry in os.scandir(train_dataset._path) if entry.is_dir() and entry.name != "_background_noise_"
)
label_to_idx = {label: idx for idx, label in enumerate(labels_set)}

print("Labels trouvés:", labels_set[:10], "...")

'''
On crée des sous-ensembles training, validation, testing.
SPEECHCOMMANDS contient 35 mots → simplification pour tests rapides.
'''

############ Partie 3️ : Prétraitement audio (Spectrogrammes)

transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_mels=64, hop_length=512
)

def collate_fn(batch):
    tensors, targets = [], []
    max_len = 0

    # Calculer la longueur max des spectrogrammes du batch
    for waveform, sample_rate, label, *_ in batch:
        # Normalisation par échantillon (corrigée)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
        spec = transform(waveform)
        tensors.append(spec)
        targets.append(label)
        if spec.size(-1) > max_len:
            max_len = spec.size(-1)

    # Pad tous les spectrogrammes à la même longueur
    padded_tensors = []
    for spec in tensors:
        pad_size = max_len - spec.size(-1)
        if pad_size > 0:
            spec = F.pad(spec, (0, pad_size))  # pad sur la dimension temps
        padded_tensors.append(spec)

    padded_tensors = torch.stack(padded_tensors)
    return padded_tensors, targets

'''
Transformer ne travaille pas sur le signal brut → on utilise spectrogrammes (séquences temporelles).
MelSpectrogram → transformée en fréquences perceptuellement pertinentes.
'''

#######Partie 4️ : Tokenisation temporelle

##Chaque colonne du spectrogramme peut être vue comme un token temporel.
def prepare_spectrograms(tensors):
    # shape: (batch_size, 1, n_mels, time) → transformer en (batch_size, time, n_mels)
    return tensors.squeeze(1).permute(0, 2, 1)

'''
On obtient une séquence de tokens temporels, chaque token représentant une portion de l'audio.
'''

#### Partie 5️ : Définir un bloc Transformer Audio
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (time, batch, embed_dim) pour nn.MultiheadAttention
        x2 = self.att(x, x, x)[0]
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm2(x + self.dropout(x2))
        return x

########## Partie 6️ : Définir l’Audio Transformer
# --- Nouveau bloc sinusoidal pour remplacer la pos_encoding aléatoire ---
def sinusoidal_position_encoding(max_len, embed_dim):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
    pe = torch.zeros(max_len, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class AudioTransformer(nn.Module):
    def __init__(self, n_mels=64, embed_dim=128, num_heads=4, ff_dim=256, num_classes=35, depth=4):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, embed_dim)  # projeter les features en embed_dim
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_encoding = sinusoidal_position_encoding(1000+1, embed_dim)  # stable et non-paramétrique
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(depth)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        # x shape: (batch, time, n_mels)
        x = self.input_proj(x)  # (batch, time, embed_dim)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # ajouter token CLS
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = x.permute(1,0,2)  # (time, batch, embed_dim) pour MultiheadAttention
        for blk in self.transformer_blocks:
            x = blk(x)
        cls_out = x[0]  # CLS token
        out = self.mlp_head(cls_out)
        return out

'''
input_proj → projette chaque frame (64 mél coefficients) en vecteur embed_dim.
cls_token → vecteur qui résume toute la séquence.
pos_encoding → information temporelle sur chaque frame.
transformer_blocks → empile plusieurs blocs attention.
x.permute(1,0,2) → MultiheadAttention attend (seq_len, batch, embed_dim).
cls_out → token CLS → classification finale.
'''

############### Partie 7️ : Préparer DataLoader
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size = 64

# Split train/val pour éviter overfitting
total_train = len(train_dataset)
train_len = int(0.9 * total_train)
val_len = total_train - train_len
train_subset, val_subset = random_split(train_dataset, [train_len, val_len])

# DataLoader pour training, validation et test
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

## collate_fn → transforme chaque audio en spectrogramme et batch correctement.

######### Partie 8️ : Instancier le modèle, loss et optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 35  # nombre de mots dans SpeechCommands
model = AudioTransformer(n_mels=64, embed_dim=128, num_heads=4, ff_dim=256, num_classes=num_classes, depth=4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

############ Partie 9️ : Boucle d’entraînement
num_epochs = 10  # un peu plus long pour stabilité

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for spectrograms, labels in train_loader:
        spectrograms = prepare_spectrograms(spectrograms).to(device)
        # Convertir labels en indices
        label_indices = torch.tensor([label_to_idx[l] for l in labels if l in label_to_idx]).to(device)

        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, label_indices)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == label_indices).sum().item()
        total += label_indices.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {acc:.2f}%")

    # Validation rapide
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = prepare_spectrograms(spectrograms).to(device)
            label_indices = torch.tensor([label_to_idx[l] for l in labels if l in label_to_idx]).to(device)
            outputs = model(spectrograms)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == label_indices).sum().item()
            val_total += label_indices.size(0)
    val_acc = 100 * val_correct / val_total
    print(f" → Validation Accuracy: {val_acc:.2f}%")

### prepare_spectrograms → convertit (batch, 1, n_mels, time) → (batch, time, n_mels).

########### Partie 10 : Évaluation sur le test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms = prepare_spectrograms(spectrograms).to(device)
        label_indices = torch.tensor([label_to_idx[l] for l in labels if l in label_to_idx]).to(device)
        outputs = model(spectrograms)
        preds = outputs.argmax(dim=1)
        correct += (preds == label_indices).sum().item()
        total += label_indices.size(0)

print(f"Test Accuracy: {100*correct/total:.2f}%")

# Sauvegarde du modèle
torch.save(model.state_dict(), "audio_transformer.pth")

######## Partie 11 : Tester avec tes propres fichiers audio
from pathlib import Path

def predict_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
    spec = transform(waveform).unsqueeze(0)  # batch = 1
    spec = prepare_spectrograms(spec).to(device)

    model.eval()
    with torch.no_grad():
        output = model(spec)
        pred_index = output.argmax(dim=1).item()
        predicted_label = labels_set[pred_index]

    print(f"Audio: {file_path} -> Predicted label: {predicted_label}")

# Exemple
predict_audio("mon_audio.wav")
