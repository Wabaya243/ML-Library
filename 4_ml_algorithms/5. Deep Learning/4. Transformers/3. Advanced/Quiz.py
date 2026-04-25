# QA_minimal.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd

# -------------------------
# Partie 0 — Réglages globaux
# -------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Partie 1 — Petit dataset toy (SQuAD-style)
# -------------------------
# Chaque exemple : context (contenant la réponse), question, answer (exact span)

toy_data = [
    # --- RÉPUBLIQUE DÉMOCRATIQUE DU CONGO (RDC) ---
    {"context": "Kinshasa est la capitale de la République démocratique du Congo et l’une des plus grandes villes d’Afrique.",
     "question": "Quelle est la capitale de la République démocratique du Congo ?",
     "answer": "Kinshasa"},

    {"context": "Le fleuve Congo traverse la République démocratique du Congo et est l’un des plus puissants du monde.",
     "question": "Quel grand fleuve traverse la République démocratique du Congo ?",
     "answer": "le fleuve Congo"},

    {"context": "Le volcan Nyiragongo, situé près de Goma, est l’un des volcans les plus actifs du monde.",
     "question": "Quel volcan actif se trouve près de Goma ?",
     "answer": "le Nyiragongo"},

    {"context": "La République démocratique du Congo possède d’immenses forêts tropicales qui font partie du bassin du Congo.",
     "question": "Quel grand bassin forestier couvre une grande partie de la RDC ?",
     "answer": "le bassin du Congo"},

    {"context": "Patrice Lumumba a été le premier Premier ministre de la République démocratique du Congo après l’indépendance en 1960.",
     "question": "Qui a été le premier Premier ministre de la République démocratique du Congo ?",
     "answer": "Patrice Lumumba"},

    {"context": "La République démocratique du Congo a obtenu son indépendance de la Belgique le 30 juin 1960.",
     "question": "En quelle année la RDC a-t-elle obtenu son indépendance ?",
     "answer": "1960"},

    {"context": "La RDC est l’un des pays les plus riches du monde en ressources naturelles, notamment en cobalt et en cuivre.",
     "question": "Quels minéraux sont particulièrement abondants en RDC ?",
     "answer": "le cobalt et le cuivre"},

    {"context": "Le lingala est l’une des principales langues parlées en République démocratique du Congo, surtout à Kinshasa.",
     "question": "Quelle langue est largement parlée à Kinshasa ?",
     "answer": "le lingala"},

    {"context": "Le parc national des Virunga, situé à l’est du pays, est célèbre pour ses gorilles de montagne.",
     "question": "Quel parc congolais abrite les gorilles de montagne ?",
     "answer": "le parc national des Virunga"},

    {"context": "Joseph Kabila a été président de la République démocratique du Congo de 2001 à 2019.",
     "question": "Qui a dirigé la RDC entre 2001 et 2019 ?",
     "answer": "Joseph Kabila"},

    {"context": "Félix Tshisekedi est le président de la République démocratique du Congo depuis 2019.",
     "question": "Qui est le président de la République démocratique du Congo depuis 2019 ?",
     "answer": "Félix Tshisekedi"},

    {"context": "Le soukous est un style musical populaire en République démocratique du Congo, représenté par des artistes comme Papa Wemba et Koffi Olomidé.",
     "question": "Quel genre musical congolais est représenté par Papa Wemba ?",
     "answer": "le soukous"},

    {"context": "La Gombe est un quartier central et administratif de la ville de Kinshasa.",
     "question": "Quel quartier est le centre administratif de Kinshasa ?",
     "answer": "la Gombe"},

    {"context": "Le stade des Martyrs est le plus grand stade de la République démocratique du Congo, situé à Kinshasa.",
     "question": "Quel est le plus grand stade de la RDC ?",
     "answer": "le stade des Martyrs"},

    {"context": "Le cuivre extrait du Katanga représente une part importante des exportations congolaises.",
     "question": "Quelle ressource naturelle est principalement exploitée dans la province du Katanga ?",
     "answer": "le cuivre"},

    {"context": "La ville de Lubumbashi est la deuxième plus grande ville de la République démocratique du Congo.",
     "question": "Quelle est la deuxième plus grande ville de la RDC ?",
     "answer": "Lubumbashi"},

    {"context": "Le parc national de la Salonga est la plus grande réserve forestière d’Afrique et se trouve en RDC.",
     "question": "Quelle est la plus grande réserve forestière d’Afrique ?",
     "answer": "le parc national de la Salonga"},

    {"context": "La musique congolaise moderne a influencé toute l’Afrique centrale avec ses rythmes et ses danses.",
     "question": "Quelle région du continent a été influencée par la musique congolaise ?",
     "answer": "l’Afrique centrale"},

    {"context": "Mobutu Sese Seko a dirigé le Zaïre, ancien nom de la RDC, pendant plus de 30 ans.",
     "question": "Quel dirigeant a gouverné le Zaïre pendant plus de 30 ans ?",
     "answer": "Mobutu Sese Seko"},

    {"context": "La RDC partage des frontières avec neuf pays, dont l’Angola, l’Ouganda et le Rwanda.",
     "question": "Combien de pays partagent une frontière avec la RDC ?",
     "answer": "neuf"},

]


df = pd.read_csv("Data/toy_dataset.csv")

# Nettoyer un peu
df = df.dropna(subset=["question", "answer"])

# Convertir en liste de dictionnaires (format attendu)
data = df[["context", "question", "answer"]].to_dict(orient="records")

print("Exemple :", data[0])
print("Nombre total d'exemples :", len(data))

# Explication :
# On utilise un petit jeu d'exemples. Les réponses sont des spans (sous-chaînes) du contexte,
# ce qui permet de déterminer un start/end token index clair.

SPECIALS = ("<PAD>", "<UNK>", "<SOS>", "<SEP>", "<EOS>")

def tokenizer(text):
    # simple whitespace tokenizer, lowercase
    return text.lower().split()

def build_vocab(examples, max_size=10000, min_freq=1, specials=SPECIALS):
    counter = Counter()
    for ex in examples:
        counter.update(tokenizer(ex["context"]))
        counter.update(tokenizer(ex["question"]))
        # also include answer tokens (redundant because answer is in context)
    vocab = {s:i for i,s in enumerate(specials)}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(data, max_size=1000, min_freq=1)
rev_vocab = {i:w for w,i in vocab.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)
print("Some vocab entries:", list(vocab.items())[:10])

# Encodage d'une paire question+context en une séquence :
# [<SOS>] question_tokens [<SEP>] context_tokens [<EOS>] + padding

# -------------------------
# Partie 2 — Encodage & calcul span start/end
# -------------------------
def find_sublist(haystack, needle):
    """Retourne l'index de la première occurrence de needle dans haystack, ou -1 si absent."""
    if len(needle) == 0:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1

def encode_qc_and_span(question, context, answer, vocab, max_len=50):
    """
    Retourne (input_ids_padded, start_idx, end_idx)
    Sequence: [<SOS>] q_tokens [<SEP>] c_tokens [<EOS>] + padding
    start/end sont indices dans la séquence (0-based).
    """
    q_tokens = tokenizer(question)
    c_tokens = tokenizer(context)
    a_tokens = tokenizer(answer)

    # ids
    q_ids = [vocab.get(t, vocab["<UNK>"]) for t in q_tokens]
    c_ids = [vocab.get(t, vocab["<UNK>"]) for t in c_tokens]
    a_ids = [vocab.get(t, vocab["<UNK>"]) for t in a_tokens]

    # construire la séquence
    ids = [vocab["<SOS>"]] + q_ids + [vocab["<SEP>"]] + c_ids + [vocab["<EOS>"]]

    # trouver span answer dans c_tokens
    match_idx_in_c = find_sublist(c_ids, a_ids)
    # position du premier token context dans la séquence:
    context_start_idx = 1 + len(q_ids) + 1  # <SOS> + q_ids + <SEP> => next index
    if match_idx_in_c >= 0:
        start = context_start_idx + match_idx_in_c
        end = start + len(a_ids) - 1
    else:
        # si on ne trouve pas, on met 0 (ou autre stratégie)
        start, end = 0, 0

    # padding / troncature
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        # si on tronque, il est possible que le span soit coupé => for safety on clamp
        ids = ids[:max_len]
        if start >= max_len:
            start = 0
            end = 0
        if end >= max_len:
            end = start  # clamp

    return ids, start, end

# construire arrays encodés
MAX_LEN = 50
examples_encoded = []
for ex in data:
    ids, s, e = encode_qc_and_span(ex["question"], ex["context"], ex["answer"], vocab, max_len=MAX_LEN)
    examples_encoded.append({"input_ids": ids, "start": s, "end": e, "question": ex["question"], "context": ex["context"], "answer": ex["answer"]})

# quick check
print("Exemple encodé (seq, start, end) :")
print(examples_encoded[0]["input_ids"][:20], examples_encoded[0]["start"], examples_encoded[0]["end"])
print("Decoded seq tokens:", [rev_vocab[i] for i in examples_encoded[0]["input_ids"][:12]])


# -------------------------
# Partie 4 — Dataset & DataLoader
# -------------------------
class QADataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "start": torch.tensor(ex["start"], dtype=torch.long),
            "end": torch.tensor(ex["end"], dtype=torch.long),
            "question": ex["question"],
            "context": ex["context"],
            "answer": ex["answer"]
        }

batch_size = 34
dataset = QADataset(examples_encoded)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


####  Partie 4 — Modèle Transformer QA

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


class TransformerQA(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=6, ff_dim=512, num_layers=4, max_len=50, dropout=0.1):
        super().__init__()

        #1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        #2. positional enconding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)

        #3. block de transformer
        self.transformer_block = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        )

        # 4. Têtes de sortie (une pour start, une pour end)
        self.start_head = nn.Linear(embed_dim, 1)
        self.end_head = nn.Linear(embed_dim, 1)

    def forward(self, input_ids):
        """
        input_ids : (batch_size, seq_len)
        return : start_logits, end_logits (batch_size, seq_len)
        """
        #embedding 
        x = self.embedding(input_ids) # (batch, seq_len, embed_dim)

        #ajouter position
        x = self.pos_encoding(x)
        

        # Passer dans les blocs Transformer
        for blk in self.transformer_block:
            x = blk(x)

        # Appliquer les têtes linéaires
        start_logits = self.start_head(x).squeeze(-1) # (batch, seq_len)
        end_logits = self.end_head(x).squeeze(-1)

        return start_logits, end_logits


# Instantiate model
model = TransformerQA(vocab_size=vocab_size, embed_dim=128, num_heads=8, ff_dim=512, num_layers=6, max_len=MAX_LEN).to(device)
print("Model created. Params:", sum(p.numel() for p in model.parameters()))

# -------------------------
# Partie 5 — Entraînement
# -------------------------
from tqdm import tqdm

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20  # toy dataset -> on peut beaucoup itérer
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    total_exact = 0
    n = 0
    for batch in tqdm(loader, desc=f"{epoch}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(device)    # (B, seq_len)
        start_targets = batch["start"].to(device)    # (B,)
        end_targets = batch["end"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids)  # (B, seq_len), (B, seq_len)

        # CrossEntropyLoss: input (B, C) and target (B) -> we transpose/select
        loss_start = criterion(start_logits, start_targets)
        loss_end = criterion(end_logits, end_targets)
        loss = (loss_start + loss_end) / 2.0
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # accuracy exact span
        start_pred = start_logits.argmax(dim=-1)
        end_pred = end_logits.argmax(dim=-1)
        total_exact += ((start_pred == start_targets) & (end_pred == end_targets)).sum().item()
        n += input_ids.size(0)

    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        avg_loss = total_loss / len(loader)
        exact_acc = total_exact / n
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f} — Exact match: {exact_acc*100:.2f}%")



# -------------------------
# Partie 6 — Évaluation
# -------------------------
import torch.nn.functional as F

def predict_answer(model, question, context, vocab, rev_vocab, max_len=MAX_LEN):
    model.eval()
    # construire séquence question+context
    q_tokens = tokenizer(question)
    c_tokens = tokenizer(context)
    q_ids = [vocab.get(t, vocab["<UNK>"]) for t in q_tokens]
    c_ids = [vocab.get(t, vocab["<UNK>"]) for t in c_tokens]
    ids = [vocab["<SOS>"]] + q_ids + [vocab["<SEP>"]] + c_ids + [vocab["<EOS>"]]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
    with torch.no_grad():
        start_logits, end_logits = model(input_ids)
        s = start_logits.argmax(dim=-1).item()
        e = end_logits.argmax(dim=-1).item()
        if e < s:
            e = s
    # reconstruire tokens pour affichage
    seq_tokens = [rev_vocab.get(i, "<UNK>") for i in ids]
    answer_tokens = seq_tokens[s:e+1]
    answer_text = " ".join(answer_tokens)
    return answer_text, (s, e), seq_tokens

# tester sur toy dataset
for ex in data[:-10]:
    ans, (s,e), seq_toks = predict_answer(model, ex["question"], ex["context"], vocab, rev_vocab)
    print("Q:", ex["question"])
    print("C:", ex["context"])
    print("Gold answer:", ex["answer"])
    print("Pred answer:", ans, "| span:", (s,e))
    print("-"*50)

# tester sur phrase custom
q = "comment s'appele les meilleur amis de l'homme ?"
c = "meilleur amis de l'homme c'est chien"
ans, span, seq_toks = predict_answer(model, q, c, vocab, rev_vocab)
print("Test custom ->", ans, span)
