# ===============================================================
# 1. Imports et chargement du dataset
# ===============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import pickle
from datasets import load_dataset

# La bonne config pour français ↔ anglais
data = load_dataset("opus_books", "en-fr")

print(data)

size = 60000  
english_sentences = [ex["en"] for ex in data["train"]["translation"]][:size]
french_sentences  = [ex["fr"] for ex in data["train"]["translation"]][:size]

# Ajouter les tokens spéciaux
french_sentences  = ["<SOS> " + s + " <EOS>" for s in french_sentences]
english_sentences = ["<SOS> " + s + " <EOS>" for s in english_sentences]


# Charger les données
#en_df = pd.read_csv("Data/small_vocab_en.csv", header=None, usecols=[0])
#fr_df = pd.read_csv("Data/small_vocab_fr.csv", header=None, usecols=[0])

#french_sentences = fr_df[0].astype(str).tolist()
#english_sentences = en_df[0].astype(str).tolist()




# ===============================================================
# 2) Tokenisation + vocabs
# ===============================================================
def tokenize(text):
    return text.lower().replace("’", "'").split()

def build_vocab(texts, min_freq=1, specials=["<PAD>", "<UNK>", "<SOS>", "<EOS>"]):
    counter = Counter()
    for sent in texts:
        counter.update(tokenize(sent))
    vocab = {tok: i for i, tok in enumerate(specials)}
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    rev = {i: w for w, i in vocab.items()}
    return vocab, rev

vocab_fr, rev_fr = build_vocab(french_sentences)   # source
vocab_en, rev_en = build_vocab(english_sentences)  # cible

def encode(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

max_len = 50
X = [encode(s, vocab_fr, max_len) for s in french_sentences]
Y = [encode(s, vocab_en, max_len) for s in english_sentences]

# Dataset PyTorch
class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src, self.tgt = src, tgt
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        src = torch.tensor(self.src[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt[idx], dtype=torch.long)
        return src, tgt

dataset = TranslationDataset(X, Y)

# ===============================================================
# 3. Définition du modèle Transformer seq2seq
# ===============================================================
pad_idx = vocab_fr["<PAD>"]
sos_idx = vocab_fr["<SOS>"]
eos_idx = vocab_fr["<EOS>"]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_in, vocab_out, emb_size=256, nhead=8, num_layers=6, ff_dim=1024, dropout=0.1):
        super().__init__()
        
        # Embedding pour la langue source (entrée)
        self.embedding_src = nn.Embedding(vocab_in, emb_size)
        
        # Embedding pour la langue cible (sortie)
        self.embedding_tgt = nn.Embedding(vocab_out, emb_size)
        
        # Encodage de position pour la séquence source (permet de savoir la position des mots)
        self.pos_src = nn.Embedding(100, emb_size)
        
        # Encodage de position pour la séquence cible
        self.pos_tgt = nn.Embedding(100, emb_size)
        
        # Le cœur du modèle : le Transformer complet (encodeur + décodeur)
        self.transformer = nn.Transformer(
            d_model=emb_size,               # taille de l’embedding
            nhead=nhead,                    # nombre de têtes d’attention
            num_encoder_layers=num_layers,  # couches d’encodeur
            num_decoder_layers=num_layers,  # couches de décodeur
            dim_feedforward=ff_dim,         # taille du réseau interne
            dropout=dropout,                # taux de dropout
            batch_first=True                # le batch est la première dimension
        )
        
        # Couche finale : transforme les vecteurs du décodeur en scores sur le vocabulaire cible
        self.fc_out = nn.Linear(emb_size, vocab_out)
        
        # Dropout global pour régularisation
        self.dropout = nn.Dropout(dropout)

    # ===============================
    # PHASE AVANT (entraînement)
    # ===============================
    def forward(self, src, tgt):
        B, S = src.shape   # B = taille du batch, S = longueur de la séquence source
        _, T = tgt.shape   # T = longueur de la séquence cible
        device = src.device

        # Calcul des positions de chaque mot dans la séquence
        pos_s = torch.arange(0, S, device=device).unsqueeze(0).repeat(B, 1)
        pos_t = torch.arange(0, T, device=device).unsqueeze(0).repeat(B, 1)

        # Ajout des embeddings de mots + embeddings de position
        src_emb = self.embedding_src(src) + self.pos_src(pos_s)
        tgt_emb = self.embedding_tgt(tgt) + self.pos_tgt(pos_t)

        # Masque triangulaire : empêche le décodeur de voir les mots futurs
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        # Masques de padding (True = position à ignorer)
        src_key_mask = (src == pad_idx)
        tgt_key_mask = (tgt == pad_idx)

        # Passage dans le modèle Transformer
        out = self.transformer(
            src_emb,                    # entrée de l’encodeur
            tgt_emb,                    # entrée du décodeur
            tgt_mask=tgt_mask,          # masque d’auto-attention du décodeur
            src_key_padding_mask=src_key_mask,   # ignore les <PAD> dans la source
            tgt_key_padding_mask=tgt_key_mask,   # ignore les <PAD> dans la cible
            memory_key_padding_mask=src_key_mask # ignore les <PAD> dans la mémoire du décodeur
        )

        # Projection finale en scores de vocabulaire
        return self.fc_out(out)

    # ===============================
    # PHASE DE GÉNÉRATION (inférence)
    # ===============================
    def greedy_decode(self, src, max_len=30):
        self.eval()  # mode évaluation : pas de dropout, pas de gradient
        device = src.device
        B = src.size(0)

        # Initialise la séquence de sortie avec uniquement le token <SOS>
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        # Boucle de génération mot par mot
        with torch.no_grad():
            for _ in range(max_len - 1):
                # Fait une passe complète du modèle avec la sortie actuelle
                logits = self.forward(src, generated)

                # Prend le mot le plus probable à la dernière position
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)

                # Ajoute le mot prédit à la séquence générée
                generated = torch.cat([generated, next_token], dim=1)

                # Si tous les tokens prédits sont <EOS>, on arrête la génération
                if (next_token == eos_idx).all():
                    break
        
        # Retourne la séquence complète générée
        return generated


# ===============================================================
# 4. Entraînement
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqTransformer(len(vocab_fr), len(vocab_en)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# Split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

print("Max src index:", max(max(x) for x in X), "/", len(vocab_fr))
print("Max tgt index:", max(max(y) for y in Y), "/", len(vocab_en))
torch.cuda.empty_cache()

nb_epoch = 50
for epoch in range(nb_epoch):
    # -------- Train --------
    model.train()
    total_loss, total_correct, total_token = 0.0, 0, 0
    for src, tgt in tqdm(train_loader, desc=f"Train {epoch+1}/{nb_epoch}"):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = (tgt_out != pad_idx)
        total_correct += ((preds == tgt_out) & mask).sum().item()
        total_token   += mask.sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc  = total_correct / total_token if total_token else 0.0

    # -------- Val --------
    model.eval()
    val_loss, val_correct, val_token = 0.0, 0, 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            val_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = (tgt_out != pad_idx)
            val_correct += ((preds == tgt_out) & mask).sum().item()
            val_token   += mask.sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc  = val_correct / val_token if val_token else 0.0

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")



# Sauvegarde poids (pas l’objet modèle)

torch.save(model.state_dict(), "Save/transformer_fr_en.pth") 
model = Seq2SeqTransformer(len(vocab_fr), len(vocab_en)).to(device) 
model.load_state_dict(torch.load("Save/transformer_fr_en.pth", map_location=device)) 
model.eval() 

#sauvegarde du model 
torch.save(model, "Save/transformer_fr_en_full.pth") 
model = torch.load("Save/transformer_fr_en_full.pth", map_location=device) 
model.eval() 

#Sauvegarde du vocab  
with open("Save/vocab_fr.pkl", "wb") as f: 
    pickle.dump(vocab_fr, f) 
with open("Save/vocab_en.pkl", "wb") as f: 
    pickle.dump(vocab_en, f) 

#chargement du vocab
with open("Save/vocab_fr.pkl", "rb") as f: 
    vocab_fr = pickle.load(f)
with open("Save/vocab_en.pkl", "rb") as f: 
    vocab_en = pickle.load(f)





# ===============================================================
# 5) Beam search + traduction
# ===============================================================
def beam_search_decode(model, src, max_len=30, beam_width=5):
    model.eval()
    device = src.device
    B = src.size(0)
    sequences = [(torch.full((B, 1), sos_idx, dtype=torch.long, device=device), 0.0)]

    with torch.no_grad():
        for _ in range(max_len - 1):
            all_candidates = []
            for seq, score in sequences:
                logits = model(src, seq)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                topk_vals, topk_idx = torch.topk(log_probs, beam_width, dim=-1)  # [B, k]
                # B vaut 1 ici (on décode phrase par phrase)
                for i in range(beam_width):
                    token = topk_idx[0, i].view(1, 1)
                    candidate_seq = torch.cat([seq, token.to(device)], dim=1)
                    candidate_score = score + topk_vals[0, i].item()
                    all_candidates.append((candidate_seq, candidate_score))
            sequences = sorted(all_candidates, key=lambda t: t[1], reverse=True)[:beam_width]
            if all(seq[0][0, -1].item() == eos_idx for seq in sequences):
                break

    return sequences[0][0]  # meilleure séquence (tensor [1, T])

def decode_sentence(ids, rev_vocab):
    words = []
    for i in ids:
        if i in [pad_idx, sos_idx]:
            continue
        if i == eos_idx:
            break
        words.append(rev_vocab.get(i, "<UNK>"))
    return " ".join(words)

def translate_sentence(model, sentence, vocab_src, vocab_tgt, rev_tgt, max_len=30, beam_width=5):
    model.eval()
    ids = encode("<SOS> " + sentence + " <EOS>", vocab_src, max_len)
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    gen = beam_search_decode(model, src, max_len=max_len, beam_width=beam_width)
    return decode_sentence(gen[0].detach().cpu().tolist(), rev_tgt)

# ===============================================================
# 6) Test
# ===============================================================
text = "Bonjour je m'ennuie un peu"
print("FR:", text)
print("EN:", translate_sentence(model, text, vocab_fr, vocab_en, rev_en, max_len=30, beam_width=5))

