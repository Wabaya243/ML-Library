# ============================
# 1. Import des librairies
# ============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import re
from collections import Counter
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# ============================
# 2. Charger le dataset IMDb
# ============================
num_words = 10000   # On garde seulement les 10 000 mots les plus fréquents
max_len = 200       # Chaque critique sera tronquée/paddée à 200 mots

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

'''
IMDb = dataset avec 25 000 critiques d’entraînement et 25 000 de test.
num_words=10000 → on limite le vocabulaire aux mots les plus fréquents.
Chaque critique est représentée par une liste d’indices entiers.
'''

# ============================
# 3. Préparer les données
# ============================
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)


'''
Les critiques ont des longueurs différentes (50 mots, 300 mots…).
pad_sequences → toutes les séquences font 200 tokens (padding = ajout de zéros à gauche si trop court, troncature si trop long).
'''

# Conversion en tenseurs PyTorch
x_train = torch.tensor(x_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

# ============================
# 4. Positional Encoding (PyTorch)
# ============================
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================
# 5. Bloc Transformer personnalisé
# ============================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # Multi-Head Attention (Self-Attention)
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate, batch_first=True)

        # Réseau Feed-Forward (MLP interne du Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Normalisation de couche (stabilise l'entraînement)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Dropout pour régularisation
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        # Self-Attention : chaque mot "regarde" les autres mots de la séquence
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        # Passage par le MLP interne (feed-forward)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)

        # Résidual connection + normalisation finale
        return self.layernorm2(out1 + ffn_output)

'''
Ceci est un bloc Transformer classique :
Self-Attention (chaque mot s’aligne sur les autres).
Skip connections + normalisation pour stabiliser.
Feed-forward pour traiter l’information localement.
'''

# ============================
# 6. Construire le modèle
# ============================
embed_dim = 64   # Dimension des embeddings
num_heads = 4     # Nombre de têtes d'attention
ff_dim = 128      # Taille du réseau feed-forward
dropout = 0.2

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, dropout, max_len):
        super().__init__()
        # Couche Embedding : transforme chaque indice de mot en vecteur dense
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(max_len, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.transformer2 = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 20)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.transformer2(x)
        x = x.mean(dim=1)               # Pooling global (moyenne sur la séquence)
        x = F.relu(self.fc1(self.dropout1(x)))
        x = self.dropout2(x)
        return self.sigmoid(self.fc2(x)) # Sortie binaire (probabilité)

model = TransformerClassifier(num_words, embed_dim, num_heads, ff_dim, dropout, max_len)

# ============================
# 7. Compiler le modèle
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()  # car c’est une classification binaire
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================
# 8. Entraîner le modèle
# ============================
for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Époque {epoch+1}/10"):
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Perte moyenne (train): {total_loss/len(train_loader):.4f}")

# ============================
# 9. Évaluer sur le test
# ============================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        predicted = (preds.squeeze() > 0.5).long()
        correct += (predicted == yb).sum().item()
        total += yb.size(0)
print(f"Exactitude sur le test: {correct/total:.4f}")

# ============================
# 10. Prédiction sur un nouveau texte
# ============================
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

def encode_review(text):
    words = text.lower().split()
    encoded = [1]  # <START>
    for w in words:
        if w in word_index and word_index[w] < num_words:
            encoded.append(word_index[w])
        else:
            encoded.append(2)  # <UNK>
    encoded = encoded[:max_len]
    if len(encoded) < max_len:
        encoded = [0] * (max_len - len(encoded)) + encoded
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

# Exemple
new_review = "This movie was absolutely fantastic, I loved it!"
seq = encode_review(new_review).to(device)
prediction = model(seq).item()
print("Score:", prediction)
if prediction > 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
