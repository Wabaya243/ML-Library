###### Partie 1 — Construire les vocabulaires (source + cible)

from datasets import load_dataset

from collections import Counter
import json

dataset = load_dataset("GEM/OrangeSum")

def tokenizer(text):
    return text.lower().split()

def build_vocab_from_texts(texts, max_size=20000, min_freq=2, specials=("<PAD>","<UNK>","<SOS>","<EOS>")):
    counter = Counter()   # ✅ il faut les parenthèses !
    for t in texts:
        counter.update(tokenizer(t))
    vocab = {s:i for i,s in enumerate(specials)}   # indices spéciaux: PAD=0, UNK=1, SOS=2, EOS=3
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        if word in vocab:
            continue
        vocab[word] = len(vocab)
    return vocab


# Extraire textes d'entraînement (existence des clés dépend du dataset)
train_docs = [ex["document"] for ex in dataset["train"]]
train_sums = [ex["summary"] for ex in dataset["train"]]

# Construire vocabulaires séparés (source = documents, target = résumés)
src_vocab = build_vocab_from_texts(train_docs, max_size=20000, min_freq=2)
tgt_vocab = build_vocab_from_texts(train_sums, max_size=10000, min_freq=1)

# inverse mappings (utile pour décoder)
src_rev = {i:w for w,i in src_vocab.items()}
tgt_rev = {i:w for w,i in tgt_vocab.items()}

# sauvegarder pour réutiliser plus tard
with open("src_vocab.json","w", encoding="utf-8") as f:
    json.dump(src_vocab, f, ensure_ascii=False)
with open("tgt_vocab.json","w", encoding="utf-8") as f:
    json.dump(tgt_vocab, f, ensure_ascii=False)

print("Taille vocab source:", len(src_vocab))
print("Taille vocab target:", len(tgt_vocab))
print("Exemples tokens source:", list(src_vocab.items())[:8])

'''
On construit deux vocabulaires séparés : un pour les documents (source) et un pour les résumés (cible).
On garde des tokens spéciaux fixes : <PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3. Important : PAD doit être 0 si tu veux l’ignorer dans la loss (ignore_index=0).
max_size limite la taille du vocabulaire (gain mémoire). min_freq évite d’ajouter des mots très rares.
Tokenisation ici = séparation par espace (simple, pédagogique). Pour du réel/produc­tion, on remplacerait par BPE/WordPiece (meilleur pour OOV).
'''

############## Partie 2 — Encodage des textes en indices ###############
import torch

def encode(text, vocab, max_len, add_sos=False, add_eos=False):
    tokens = text.lower().split()
    ids = []
    if add_sos:
        ids.append(vocab.get("<SOS>", 1))
    ids += [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if add_eos:
        ids.append(vocab.get("<EOS>", 1))
    # padding
    if len(ids) < max_len:
        ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    else :
        ids = ids[:max_len]
    return ids

# On prend un exemple réel du dataset
doc = train_docs[0]    # document du dataset
summ = train_sums[0]   # résumé correspondant

src_ids = encode(doc, src_vocab, max_len=12)
tgt_in_ids = encode(summ, tgt_vocab, max_len=8, add_sos=True)   # pour le décodeur
tgt_out_ids = encode(summ, tgt_vocab, max_len=8, add_eos=True)  # cible de l’apprentissage

print("Document encodé:", src_ids)
print("Résumé (entrée décodeur):", tgt_in_ids)
print("Résumé (cible sortie):", tgt_out_ids)

################# Partie 3 : transformer tout le dataset en tenseurs PyTorch et créer des DataLoaders #########

###Étape 1 : Encodage complet du dataset

#longeur max (a adapter selon dataset)
max_src_len = 200
max_tgt_len = 50

# Encodage complet
encoded_src = [encode(doc, src_vocab, max_len=max_src_len) for doc in train_docs]
encoded_tgt_in = [encode(summ, tgt_vocab, max_len=max_tgt_len, add_sos=True) for summ in train_sums]
encoded_tgt_out = [encode(summ, tgt_vocab, max_len=max_tgt_len, add_eos=True) for summ in train_sums]

'''
encoded_src[i] → indices du document i.
encoded_tgt_in[i] → indices du résumé i pour l’entrée du décodeur.
encoded_tgt_out[i] → indices du résumé i pour la cible du modèle
'''
###### Étape 2 : Transformer en tenseurs PyTorch #####

import torch

src_tensor = torch.tensor(encoded_src, dtype=torch.long)
tgt_in_tensor = torch.tensor(encoded_tgt_in, dtype=torch.long)
tgt_out_tensor = torch.tensor(encoded_tgt_out, dtype=torch.long)

print("Tenseur documents :", src_tensor.shape)
print("Tenseur résumés (entrée) :", tgt_in_tensor.shape)
print("Tenseur résumés (sortie) :", tgt_out_tensor.shape)

############## Étape 3 : Créer un DataLoader PyTorch ######

'''
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32 

train_dataset = TensorDataset(src_tensor, tgt_in_tensor, tgt_out_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Vérification d'un batch
for batch_src, batch_tgt_in, batch_tgt_out in train_loader:
    print(batch_src.shape, batch_tgt_in.shape, batch_tgt_out.shape)
    break

#################### Partie 4 — Construire un mini-Transformer pour le résumé #################
'''
import torch
import torch.nn as nn
import math

### Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        # x.shape = (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

### transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

#### Mini-Transformer seq2seq pour résumé
class MiniTransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, ff_dim, max_src_len, max_tgt_len, dropout=0.1):
        super().__init__()
        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.src_pos = PositionalEncoding(embed_dim, max_src_len)
        self.tgt_pos = PositionalEncoding(embed_dim, max_tgt_len)
        
        # Encoder
        self.encoder = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        # Decoder
        self.decoder_self = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.decoder_cross = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Final Linear to predict next token
        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        
        # Embedding + positional
        src_emb = self.src_pos(self.src_embedding(src))
        tgt_emb = self.tgt_pos(self.tgt_embedding(tgt))
        
        # Encoder
        enc_out = self.encoder(src_emb)
        
        # Decoder masked self-attention
        dec_self_out = self.decoder_self(tgt_emb)
        
        # Cross-attention: query=decoder, key=value=encoder
        dec_out, _ = self.decoder_cross(dec_self_out, enc_out, enc_out)
        
        # Final token prediction
        logits = self.fc_out(dec_out)  # (batch, tgt_len, tgt_vocab_size)
        return logits

'''

Self-attention : chaque token du résumé regarde ses prédécesseurs pour apprendre à générer la séquence (on peut ajouter un mask pour empêcher de regarder les tokens futurs).
Cross-attention : le decoder regarde l’output de l’encoder pour savoir quelle partie du document est pertinente pour générer le token suivant.
'''

######## Partie 5 — DataLoader seq2seq et préparation des batches ######

from torch.utils.data import Dataset, DataLoader
import torch

class Seq2SeqDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, 
                 max_src_len=200, max_tgt_len=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_seq = encode(self.src_texts[idx], self.src_vocab, self.max_src_len)
        tgt_in_seq = encode(self.tgt_texts[idx], self.tgt_vocab, 
                            self.max_tgt_len, add_sos=True)
        tgt_out_seq = encode(self.tgt_texts[idx], self.tgt_vocab, 
                             self.max_tgt_len, add_eos=True)

        return (torch.tensor(src_seq, dtype=torch.long),
                torch.tensor(tgt_in_seq, dtype=torch.long),
                torch.tensor(tgt_out_seq, dtype=torch.long))


from torch.utils.data import random_split

# 90% train, 10% validation
val_ratio = 0.1
val_size = int(len(train_docs) * val_ratio)
train_size = len(train_docs) - val_size

train_docs_split, val_docs_split = random_split(train_docs, [train_size, val_size])
train_sums_split, val_sums_split = random_split(train_sums, [train_size, val_size])

# Dataset
train_dataset = Seq2SeqDataset(train_docs_split, train_sums_split, src_vocab, tgt_vocab)
val_dataset = Seq2SeqDataset(val_docs_split, val_sums_split, src_vocab, tgt_vocab)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Vérification d'un batch validation
for x_src, x_tgt_in, y_tgt_out in val_loader:
    print("Validation batch shapes:", x_src.shape, x_tgt_in.shape, y_tgt_out.shape)
    break

# Exemple batch
for x_src, x_tgt_in, y_tgt_out in train_loader:
    print("x_src.shape:", x_src.shape)         # (batch_size, max_src_len)
    print("x_tgt_in.shape:", x_tgt_in.shape)   # (batch_size, max_tgt_len)
    print("y_tgt_out.shape:", y_tgt_out.shape) # (batch_size, max_tgt_len)
    break

## Loss avec Ignore_index 
criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab["<PAD>"])

'''
On ignore <PAD> pour ne pas pénaliser le modèle pour le padding.
Input logits du modèle : (batch_size, tgt_len, tgt_vocab_size)
Target : (batch_size, tgt_len)
💡 Important : il faudra aplatir la dimension tgt_len pour la lo
'''

################ Partie 6 — Entraînement du modèle ###############


import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparamètres
embed_dim = 128
num_heads = 4
ff_dim = 512
max_src_len = 200
max_tgt_len = 50
dropout = 0.1
batch_size = 32
epochs = 5
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciation du modèle
model = MiniTransformerSeq2Seq(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    max_src_len=max_src_len,
    max_tgt_len=max_tgt_len,
    dropout=dropout
).to(device)

# Loss (ignore <PAD>)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Fonction pour entraîner sur un batch
def train_one_batch(model, src, tgt_in, tgt_out):
    model.train()
    optimizer.zero_grad()
    
    src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
    
    # Forward
    logits = model(src, tgt_in)  # (batch, tgt_len, tgt_vocab_size)
    
    # Loss : CrossEntropy attend (N, C) et target (N), donc on aplatit
    logits = logits.view(-1, logits.size(-1))   # (batch*tgt_len, vocab_size)
    tgt_out = tgt_out.view(-1)                  # (batch*tgt_len)
    
    loss = criterion(logits, tgt_out)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Fonction d'évaluation
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for src, tgt_in, tgt_out in val_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            logits = logits.view(-1, logits.size(-1))   # (batch*tgt_len, vocab_size)
            tgt_out_flat = tgt_out.view(-1)             # (batch*tgt_len)

            loss = criterion(logits, tgt_out_flat)
            total_loss += loss.item()

            # Token accuracy
            preds = logits.argmax(dim=-1)
            mask = tgt_out_flat != tgt_vocab["<PAD>"]   # ignore PAD
            correct_tokens += (preds[mask] == tgt_out_flat[mask]).sum().item()
            total_tokens += mask.sum().item()

    val_loss = total_loss / len(val_loader)
    val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return val_loss, val_acc

# Boucle d'entraînement complète
for epoch in range(1, epochs + 1):
    train_loss = 0
    for src, tgt_in, tgt_out in train_loader:
        batch_loss = train_one_batch(model, src, tgt_in, tgt_out)
        train_loss += batch_loss
    train_loss /= len(train_loader)
    
    val_loss, val_acc = evaluate(model, val_loader)
    
    print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")



############# Sauvegarde et chargement

## le poids plus conseillé

# Sauvegarder
torch.save(model.state_dict(), "mini_transformer_weights.pth")

# Charger
model_loaded = MiniTransformerSeq2Seq(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    max_src_len=max_src_len,
    max_tgt_len=max_tgt_len,
    dropout=dropout
).to(device)

model_loaded.load_state_dict(torch.load("mini_transformer_weights.pth", map_location=device))
model_loaded.eval()  # mettre en mode évaluation


#### modele complet 

# Sauvegarder
torch.save(model, "mini_transformer_full.pth")

# Charger
model_loaded = torch.load("mini_transformer_full.pth", map_location=device)
model_loaded.eval()

#⚠️ Inconvénient : dépend strictement du code exact de la classe, moins flexible si tu modifies quelque chose

'''
state_dict = ce que tu veux presque toujours.
Toujours mettre .eval() avant de faire des prédictions pour désactiver dropout, etc.
'''

################## Partie 7 — Test et évaluation finale sur le jeu de validation.

import torch

# Fonction pour générer un résumé à partir d'un document
def generate_summary(model, src_seq, max_tgt_len, tgt_vocab, device):
    model.eval()
    src_seq = torch.tensor(src_seq, dtype=torch.long).unsqueeze(0).to(device)  # batch=1
    generated = [tgt_vocab["<SOS>"]]
    
    for _ in range(max_tgt_len):
        tgt_input = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src_seq, tgt_input)  # (1, seq_len, vocab_size)
            next_token = logits[0, -1].argmax().item()
        if next_token == tgt_vocab["<EOS>"]:
            break
        generated.append(next_token)
    
    return generated[1:]  # on retire le <SOS>

# Évaluation complète sur le val_dataset
def test_model(model, val_loader, tgt_vocab, device, print_samples=3):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    all_src = []
    all_tgt = []
    all_pred = []
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab["<PAD>"])
    
    with torch.no_grad():
        for i, (src, tgt_in, tgt_out) in enumerate(val_loader):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            
            logits = model(src, tgt_in)
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_out_flat = tgt_out.view(-1)
            
            loss = criterion(logits_flat, tgt_out_flat)
            total_loss += loss.item()
            
            # token accuracy
            preds = logits_flat.argmax(dim=-1)
            mask = tgt_out_flat != tgt_vocab["<PAD>"]
            correct_tokens += (preds[mask] == tgt_out_flat[mask]).sum().item()
            total_tokens += mask.sum().item()
            
            # Stocker quelques exemples pour affichage
            if i < print_samples:
                all_src.extend(src.cpu().tolist())
                all_tgt.extend(tgt_out.cpu().tolist())
                all_pred.extend([generate_summary(model, s, tgt_in.size(1), tgt_vocab, device) for s in src.cpu().tolist()])
    
    val_loss = total_loss / len(val_loader)
    val_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    print(f"Test Loss: {val_loss:.4f} | Token Accuracy: {val_acc:.4f}")
    
    # Afficher quelques résumés générés
    for i in range(print_samples):
        src_text = " ".join([src_rev[idx] for idx in all_src[i] if idx != src_vocab["<PAD>"]])
        tgt_text = " ".join([tgt_rev[idx] for idx in all_tgt[i] if idx not in (tgt_vocab["<PAD>"], tgt_vocab["<SOS>"], tgt_vocab["<EOS>"])])
        pred_text = " ".join([tgt_rev[idx] for idx in all_pred[i] if idx not in (tgt_vocab["<PAD>"], tgt_vocab["<SOS>"], tgt_vocab["<EOS>"])])
        print("\nDocument :", src_text[:200], "...")
        print("Réel    :", tgt_text)
        print("Généré  :", pred_text)

# 1️⃣ Tester le modèle sur le val_loader et afficher quelques exemples
test_model(model, val_loader, tgt_vocab, device, print_samples=3)

##################### Partie 8 correspond à tester le modèle sur de nouvelles données

#On ne connaît pas encore le résumé complet, donc on va générer pas à pas (auto-régressif) :
def generate_summary(model, src_text, src_vocab, tgt_vocab, tgt_rev, 
                     max_src_len=200, max_tgt_len=50, device=device):
    model.eval()
    with torch.no_grad():
        # Encoder le document
        src_seq = torch.tensor([encode(src_text, src_vocab, max_len=max_src_len)], dtype=torch.long).to(device)
        
        # Démarrer la séquence de sortie avec <SOS>
        tgt_seq = torch.tensor([[tgt_vocab["<SOS>"]]], dtype=torch.long).to(device)
        
        for _ in range(max_tgt_len):
            logits = model(src_seq, tgt_seq)  # (1, seq_len, tgt_vocab_size)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)  # dernier token
            tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
            if next_token.item() == tgt_vocab["<EOS>"]:
                break
        
        # Décoder en mots
        tokens = [tgt_rev[idx.item()] for idx in tgt_seq[0][1:]]  # ignorer <SOS>
        # Stop à <EOS>
        if "<EOS>" in tokens:
            tokens = tokens[:tokens.index("<EOS>")]
        
        return " ".join(tokens)


example_doc = "Orange has just released its new quarterly report showing a significant increase in revenue."
summary = generate_summary(model, example_doc, src_vocab, tgt_vocab, tgt_rev)
print("Résumé généré :", summary)


'''
for doc in dataset["test"][:5]:  # prendre 5 documents du test
    print("Document :", doc["document"])
    print("Résumé généré :", generate_summary(model, doc["document"], src_vocab, tgt_vocab, tgt_rev))
    print("Résumé réel :", doc["summary"])
    print("---")

'''





