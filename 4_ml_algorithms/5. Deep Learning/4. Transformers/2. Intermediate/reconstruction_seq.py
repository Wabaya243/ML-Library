# Partie 1 — Charger et préparer le dataset
from datasets import load_dataset

# charger dataset
dataset = load_dataset("imdb")

# pour l'entrainement rapide, juste 20000 exemples
train_subset = dataset["train"].select(range(20000))
val_subset = dataset["test"].select(range(2000))

train_texts = [ex["text"] for ex in train_subset]
val_texts   = [ex["text"] for ex in val_subset]

print("Exemple texte:", train_texts[0])


#Partie 2 — Construire le vocabulaire

from collections import Counter

def tokenizer(text):
    return text.lower().split()

def build_vocab_from_texts(texts, max_size=20000, min_freq=2, specials=("<PAD>","<UNK>","<SOS>","<EOS>")):
    counter = Counter()
    for t in texts:
        counter.update(tokenizer(t))
    vocab = {s:i for i,s in enumerate(specials)}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        if word in vocab:
            continue
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab_from_texts(train_texts)
rev_vocab = {i:w for w,i in vocab.items()}

print("Taille vocab:", len(vocab))
print("Exemples tokens:", list(vocab.items())[:10])


############ Partie 3 — Encodage des séquences

max_len = 100

def encode(text, vocab, max_len, add_sos=False, add_eos=False):
    tokens = tokenizer(text)
    ids = []
    if add_sos:
        ids.append(vocab["<SOS>"])
    ids += [vocab.get(t, vocab["<UNK>"]) for t in tokens]   # sécurité
    if add_eos:
        ids.append(vocab["<EOS>"])
    ids = ids[:max_len]  # tronquer si trop long
    ids += [vocab["<PAD>"]] * (max_len - len(ids))  # compléter si trop court
    return ids


#  Correction : src = texte brut, tgt = avec <SOS> et <EOS>
train_encoded_in  = [encode(t, vocab, max_len) for t in train_texts]
train_encoded_out = [encode(t, vocab, max_len, add_sos=True, add_eos=True) for t in train_texts]

val_encoded_in  = [encode(t, vocab, max_len) for t in val_texts]
val_encoded_out = [encode(t, vocab, max_len, add_sos=True, add_eos=True) for t in val_texts]


'''
train_encoded_in → séquence avec <SOS> pour l’entrée du décodeur
train_encoded_out → séquence avec <EOS> comme target à prédire
Le modèle apprend à prédire chaque mot suivant, jusqu’au <EOS>
'''

############### Partie 4 — Dataset et DataLoader PyTorch

import torch
from torch.utils.data import DataLoader, Dataset

class SeqReconstructionDataset(Dataset):
    def __init__(self, enc_in, enc_out):
        self.enc_in = enc_in
        self.enc_out = enc_out
    def __len__(self):
        return len(self.enc_in)
    def __getitem__(self, idx):
        return torch.tensor(self.enc_in[idx], dtype=torch.long), torch.tensor(self.enc_out[idx], dtype=torch.long)

batch_size = 64
train_dataset = SeqReconstructionDataset(train_encoded_in, train_encoded_out)
val_dataset   = SeqReconstructionDataset(val_encoded_in, val_encoded_out)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)


############ Partie 5 — Mini-Transformer Auto-encodeur

import torch.nn as nn

# --- Bloc Transformer simple ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Multi-Head Attention : le coeur du Transformer, permet au modèle de "regarder" chaque mot en fonction des autres
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Feed Forward : un petit réseau dense pour traiter chaque vecteur après l'attention
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        # Normalisation de couche pour stabiliser l'entraînement
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention multi-head : besoin de permuter les dimensions pour MultiheadAttention
        # shape attendue : (seq_len, batch_size, embed_dim)
        x_attn, _ = self.attention(x, x, x)  # self-attention
        x = self.norm1(x + self.dropout(x_attn))  # résidu + norm
        x_ff = self.ff(x)  # passage dans le feed-forward
        x = self.norm2(x + self.dropout(x_ff))  # résidu + norm
        return x

### Les Transformers ne savent pas l’ordre des mots, donc il faut ajouter des vecteurs positionnels.

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        # matrice de position
        pos = torch.arange(0, max_len).unsqueeze(1)  # shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)  # sine sur les dimensions paires
        pe[:, 1::2] = torch.cos(pos * div_term)  # cosine sur les dimensions impaires
        self.pe = pe.unsqueeze(1)  # shape (max_len, 1, embed_dim)
    
    def forward(self, x):
        # x shape : (seq_len, batch_size, embed_dim)
        return x + self.pe[:x.size(0), :]

'''
Chaque position a un vecteur unique, injecté dans l’embedding.
Sine et Cosine permettent au modèle de généraliser à des séquences plus longues que celles vues en entraînement.
'''

# ------------------------------
# Seq2Seq Transformer (Encoder-Decoder)
# ------------------------------

import torch.nn as nn

pad_idx = vocab["<PAD>"]
sos_idx = vocab["<SOS>"]
eos_idx = vocab["<EOS>"]

# Définition d’un modèle Seq2Seq basé sur un Transformer complet (encodeur + décodeur)
class Seq2SeqTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, ff_dim=512,
                 num_enc_layers=6, num_dec_layers=6, max_len=100, dropout=0.1):
        super().__init__()

        # Couche d’embedding pour convertir les indices de mots en vecteurs denses
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Embedding positionnel pour indiquer la position de chaque mot dans la phrase
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # Transformer complet (PyTorch) : encodeur + décodeur
        self.transformer = nn.Transformer(
            d_model=embed_dim,           # dimension des vecteurs d’embedding
            nhead=num_heads,             # nombre de têtes d’attention multi-head
            num_encoder_layers=num_enc_layers,  # couches d’encodeur
            num_decoder_layers=num_dec_layers,  # couches de décodeur
            dim_feedforward=ff_dim,      # taille du réseau feed-forward
            dropout=dropout,             # taux de dropout pour la régularisation
            batch_first=True             # les batchs sont au format (batch, seq_len, embed_dim)
        )

        # Couche linéaire finale : transforme les vecteurs du décodeur en scores pour chaque mot du vocabulaire
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        # Initialisation Xavier pour stabiliser l’apprentissage
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, src, tgt_input):
        # Récupération du device (CPU ou GPU)
        device = src.device

        # Dimensions du batch : B = taille du batch, S = longueur source, T = longueur cible
        B, S = src.shape
        _, T = tgt_input.shape

        # Positions de chaque token (0 à S-1 pour la source, 0 à T-1 pour la cible)
        pos_src = torch.arange(0, S, device=device).unsqueeze(0).repeat(B, 1)
        pos_tgt = torch.arange(0, T, device=device).unsqueeze(0).repeat(B, 1)

        # Embedding + position : chaque mot = vecteur + position correspondante
        src_emb = self.embed(src) + self.pos_emb(pos_src)
        tgt_emb = self.embed(tgt_input) + self.pos_emb(pos_tgt)

        # --- Création des masques ---
        # Masque triangulaire : empêche le décodeur de "voir" les tokens futurs
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        # Masques de padding : True là où il faut ignorer (tokens <PAD>)
        src_key_padding_mask = (src == pad_idx)         # pour la source
        tgt_key_padding_mask = (tgt_input == pad_idx)   # pour la cible

        # Passage dans le Transformer (encodeur + décodeur + attention croisée)
        out = self.transformer(
            src_emb,                      # entrée encodeur
            tgt_emb,                      # entrée décodeur
            tgt_mask=tgt_mask,            # masque futur
            src_key_padding_mask=src_key_padding_mask,   # ignore <PAD> côté source
            tgt_key_padding_mask=tgt_key_padding_mask,   # ignore <PAD> côté cible
            memory_key_padding_mask=src_key_padding_mask # ignore <PAD> dans la mémoire de l’encodeur
        )

        # Projection finale : transformer chaque vecteur en probabilité sur le vocabulaire
        return self.fc_out(out)

    def generate_greedy(self, src, max_len=50, sos_idx=None, eos_idx=None):
        # Mode évaluation : pas de dropout, pas de gradients
        self.eval()

        # Récupération automatique du device du modèle
        device = next(self.parameters()).device

        # B = taille du batch (souvent 1)
        B = src.size(0)

        # Si aucun token spécial fourni, utiliser ceux du vocabulaire
        if sos_idx is None:
            sos_idx = vocab["<SOS>"]
        if eos_idx is None:
            eos_idx = vocab["<EOS>"]

        # Initialisation de la génération avec uniquement le token <SOS>
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
            
        # Boucle de génération mot par mot
        with torch.no_grad():
            for step in range(max_len - 1):
                # Passage dans le modèle avec la séquence générée actuelle
                logits = self.forward(src, generated)

                # Prend le mot le plus probable (greedy decoding)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                # Ajoute ce mot à la séquence générée
                generated = torch.cat([generated, next_token], dim=1)

                # Si tous les tokens sont <EOS>, on arrête la génération
                if (next_token == eos_idx).all():
                    break

        # Retourne la séquence générée complète
        return generated


    
    
'''
Avec un seul bloc, ton modèle n’apprend qu’un niveau de dépendance entre tokens.
Mais dans le langage (ou les séquences en général) tu as des relations hiérarchiques :

Bloc 1 : apprend des relations locales (ex : “the” → “cat”).
Bloc 2 : combine pour capter des relations plus larges (ex : “the cat” → “is sleeping”).
Bloc 3, 4, … : va chercher des dépendances encore plus longues et abstraites.
'''

##### Partie 6 — Préparer la loss, l’optimiseur et le device

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print("Device Utilisé : ", device)

## iniatiliser le modele et envoyer sur device
vocab_size = len(vocab)
model = Seq2SeqTransformerModel(vocab_size=len(vocab), embed_dim=128, num_heads=8, ff_dim=512, num_enc_layers=6, num_dec_layers=6, max_len=max_len, dropout=0.1).to(device)

# Loss : CrossEntropyLoss pour la prédiction token par token
# ignore_index = PAD pour ne pas compter les tokens de padding

pad_idx = vocab["<PAD>"]
sos_idx = vocab["<SOS>"]
eos_idx = vocab["<EOS>"]

# ---------------- Correction ici -----------------
# Perte : ne pas lisser trop tôt (supprime label_smoothing)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Optimiseur : LR plus haut pour débloquer l'apprentissage
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

# (Optionnel) scheduler désactivé pour ne pas étouffer le LR au début
scheduler = None


############# Partie 7 — Boucle d’entraînement (une époque) AVEC ACCURACY

from tqdm import tqdm

def train_epoch_seq2seq(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for src, tgt in tqdm(loader, desc=f"Epoch {epoch+1}/{nb_epoch}"):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        mask = tgt_out != pad_idx
        correct = (preds == tgt_out) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()
    return total_loss / len(loader), total_correct / total_tokens

def eval_epoch_seq2seq(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.no_grad():
        for src, tgt in tqdm(loader, desc="validation"):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask = tgt_out != pad_idx
            correct = (preds == tgt_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    return total_loss / len(loader), total_correct / total_tokens



########### Partie 9 — Boucle complète d’entraînement

nb_epoch = 20

for epoch in range(nb_epoch):
    train_loss, train_acc = train_epoch_seq2seq(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch_seq2seq(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{nb_epoch} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
          f"- Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# --- TEST RECONSTRUCTION ---


def decode_ids(ids, rev_vocab):
    words = []
    for i in ids:
        if i in (vocab["<PAD>"], vocab["<SOS>"]):
            continue
        if i == vocab["<EOS>"]:
            break
        words.append(rev_vocab.get(i, "<UNK>"))
    return " ".join(words)

######## Partie 10 — Tester la reconstruction
model.eval()
with torch.no_grad():
    sample_src, sample_tgt = next(iter(val_loader))
    sample_src, sample_tgt = sample_src.to(device), sample_tgt.to(device)
    tgt_input = sample_tgt[:, :-1]
    logits = model(sample_src, tgt_input)
    preds = logits.argmax(dim=-1).cpu().numpy()

input_tokens = [rev_vocab[i] for i in sample_src[0].cpu().numpy() if i != vocab["<PAD>"]]


pred_tokens  = [rev_vocab[i] for i in preds[0] if i != vocab["<PAD>"]]

print("Input      :", " ".join(input_tokens))
print("Reconstruction :", " ".join(pred_tokens))


# generate for one custom input (raw text)

def generate_from_text(model, raw_text, vocab, rev_vocab, max_len=60):
    model.to(device)
    model.eval()
    ids = encode(raw_text, vocab, max_len)  # input pour l’encodeur = sans <SOS>/<EOS>
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    gen = model.generate_greedy(src, max_len=max_len, sos_idx=vocab["<SOS>"], eos_idx=vocab["<EOS>"])
    return decode_ids(gen[0].cpu().tolist(), rev_vocab)


example = "this movie had a very slow start but the finale was great"
print("Input :", example)
print("Output:", generate_from_text(model, example, vocab, rev_vocab, max_len=80))


