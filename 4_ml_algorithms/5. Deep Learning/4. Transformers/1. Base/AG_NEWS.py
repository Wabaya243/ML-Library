import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import numpy as np
import re

# ==========================================================
# 1. Chargement du jeu de données AG News
# ==========================================================
# Ce dataset contient 4 classes : World, Sports, Business, Sci/Tech
dataset = load_dataset("ag_news")

train_texts = dataset["train"]["text"]
train_labels = torch.tensor(dataset["train"]["label"])
test_texts = dataset["test"]["text"]
test_labels = torch.tensor(dataset["test"]["label"])

print("Exemple texte:", train_texts[0])
print("Label:", train_labels[0].item())

# ==========================================================
# 2. Prétraitement : Tokenisation et vocabulaire maison
# ==========================================================
# On crée un tokenizer simple basé sur des regex.
def basic_tokenizer(text):
    # Sépare les mots alphabétiques et numériques, supprime la casse
    #découper le texte en “tokens” (mots, nombres, etc.).
    return re.findall(r"\b\w+\b", text.lower())

# Vocabulaire minimal sans torchtext
class SimpleVocab:
    def __init__(self, counter, max_size=20000, specials=["<pad>", "<unk>"]):
        # specials = tokens réservés pour le padding et les mots inconnus
        self.itos = specials + [w for w, _ in counter.most_common(max_size - len(specials))]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

# Construction du vocabulaire à partir du corpus

counter = Counter() #contient la fréquence de chaque mot
for text in tqdm(train_texts, desc="Construction du vocabulaire"):
    counter.update(basic_tokenizer(text))

num_words = 20000
vocab = SimpleVocab(counter, max_size=num_words)
pad_idx = vocab["<pad>"]
unk_idx = vocab["<unk>"]

# Fonction d’encodage (tokenisation + indexation + padding)
def encode(text, max_len=200):
    tokens = basic_tokenizer(text) #Découpe le texte en une liste de mots normalisés.
    ids = [vocab[token] for token in tokens[:max_len]]  #Chaque mot est remplacé par son identifiant numérique dans ton vocabulaire.
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids)) # Si le texte est trop court, on le remplit avec le token <pad> (souvent = 0).
    return torch.tensor(ids)

# Encodage du jeu d’entraînement et de test
x_train = torch.stack([encode(t) for t in tqdm(train_texts, desc="Encodage train")])
x_test = torch.stack([encode(t) for t in tqdm(test_texts, desc="Encodage test")])

y_train = train_labels
y_test = test_labels

print("Dimensions train:", x_train.shape, y_train.shape)

# Création des DataLoaders pour l’entraînement et l’évaluation
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ==========================================================
# 3. Positional Encoding (sinusoïdal)
# ==========================================================
# Le modèle Transformer est invariant à l’ordre des mots.
# On ajoute donc un encodage sinusoïdal pour injecter la notion de position.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # dimension batch
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Ajoute la position aux embeddings
        return x + self.pe[:, :x.size(1)]

# ==========================================================
# 4. Modèle Transformer simplifié pour la classification
# ==========================================================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=256,
                 num_classes=4, max_len=200, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)           # [batch, seq_len, embed_dim]
        x = self.pos_enc(x)             # ajout des positions
        x = self.transformer(x)         # bloc Transformer
        x = x.mean(dim=1)               # pooling global (moyenne sur la séquence)
        x = F.relu(self.fc1(self.dropout1(x)))
        x = self.dropout2(x)
        return self.fc2(x)

# ==========================================================
# 5. Entraînement
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(len(vocab))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Perte moyenne (train): {total_loss/len(train_loader):.4f}")

# ==========================================================
# 6. Évaluation
# ==========================================================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        correct += (preds.argmax(1) == yb).sum().item()
        total += yb.size(0)
print(f"Exactitude sur le test: {correct/total:.4f}")

# ==========================================================
# 7. Prédiction sur un texte libre
# ==========================================================
def predict_text(text):
    model.eval()
    seq = encode(text).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(seq)
        label = pred.argmax(1).item()
    categories = ["World", "Sports", "Business", "Sci/Tech"]
    return categories[label]

print(predict_text("Stock markets are down after the new economic policy."))
print(predict_text("The football team won the championship yesterday."))
