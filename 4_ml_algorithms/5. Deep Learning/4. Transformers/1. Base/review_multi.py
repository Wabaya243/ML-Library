from seaborn import relational
from tensorflow.keras.datasets import reuters

# Charger Reuters (intégrée Keras)
num_words = 20000  # vocabulaire limité
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)

from tensorflow.keras.datasets import reuters

# Charger le mapping mot -> indice
word_index = reuters.get_word_index(path="reuters_word_index.json")

# Important : Keras réserve les indices 0,1,2,3
# On décale donc tous les indices du vocabulaire de +3
word_index = {k: (v + 3) for k, v in word_index.items()}

# Ajouter les tokens spéciaux
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2   # unknown token
word_index["<UNUSED>"] = 3


print("Nombre d'articles train:", len(x_train))
print("Nombre d'articles test:", len(x_test))
print("Exemple d'article (liste d'indices):", x_train[0][:20])
print("Label:", y_train[0])

####### Partie 2 : Prétraitement (padding & conversion en PyTorch)

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

max_len = 200  # longueur max de l'article

# Padding manuellement
def pad_sequence_torch(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            seq = seq + [0]* (max_len - len(seq))
        else :
            seq = seq[:max_len]
        padded.append(seq)
    return torch.tensor(padded, dtype=torch.long)

x_train = pad_sequence_torch(x_train, max_len)
x_test = pad_sequence_torch(x_test, max_len)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Création DataLoader PyTorch
batch_size = 64
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

'''
Pourquoi ?
PyTorch travaille avec tensors → on convertit les séquences en torch.tensor.
Padding = toutes les séquences ont max_len tokens pour batch.
DataLoader → simplifie l’itération sur mini-batch.
'''

################## Partie 3 : Embedding + Positional Encoding ############
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Création d'une matrice (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # indices pairs
        pe[:, 1::2] = torch.cos(position * div_term)  # indices impairs
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)  # stocke en tant que buffer non entraînable
    
    def forward(self, x):
         # x.shape = (batch_size, seq_len, d_model)
         x = x + self.pe[:, :x.size(1), :]
         return x

# Paramètres
vocab_size = 20000
embed_dim = 128

'''
✨ Pourquoi ?

Embedding : transforme chaque token (entier) en vecteur dense de dimension embed_dim.
Permet au modèle de travailler sur des vecteurs continus plutôt que des IDs.

Positional Encoding : ajoute l’information de position des mots.
Self-attention ignore l’ordre des mots si on ne le précise pas.
Sinus/cosinus = fonction déterministe qui encode chaque position différemment.

register_buffer : le positional encoding n’est pas entraînable.
On l’ajoute simplement aux embeddings.
'''

########### : Transformer Block (Self-Attention + Feed-Forward + Residual)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Feed-Forward Network (MLP)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # --- Self-Attention ---
        attn_output, _ = self.attention(x, x, x)  # Q=K=V=x
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)           # Residual + Norm
        
        # --- Feed-Forward ---
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output)
        x = self.norm2(x + ffn_output)           # Residual + Norm
        return x

'''
Multi-Head Self-Attention
Chaque mot regarde tous les autres mots pour calculer sa nouvelle représentation.
Ici Q=K=V=x → self-attention.

Residual + LayerNorm
Résidual (x + attn_output) = permet de préserver l’information originale et stabilise le gradient.
LayerNorm = normalisation pour éviter que les valeurs explosent.

Feed-Forward Network (MLP)
Transforme chaque position indépendamment pour enrichir la représentation.
Linear -> ReLU -> Linear = classique dans le Transformer.

Dropout
Régularisation pour éviter l’overfitting sur les données.
'''

class MiniTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_classes, max_len, dropout=0.1):
        super(MiniTransformerClassifier, self).__init__()
        # Embedding + Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(d_model=embed_dim, max_len=max_len)
        
        # Transformer Block
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.transformer2 = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.transformer3 = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        
        # Pooling global
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)            # (batch, seq_len, embed_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)          # TransformerBlock
        x = self.transformer2(x)
        x = self.transformer3(x)
        x = x.permute(0, 2, 1)           # (batch, embed_dim, seq_len) pour avg pooling
        x = self.pool(x).squeeze(-1)     # (batch, embed_dim)
        x = self.classifier(x)           # logits (batch, num_classes)
        return x

'''
Embedding + Positional Encoding
Chaque token devient un vecteur dense + information de position pour que l’ordre des mots soit pris en compte.

TransformerBlock
Self-attention enrichit chaque mot en regardant tous les autres.
FFN et residuals pour stabiliser et enrichir les représentations.

AdaptiveAvgPool1d
Fait un pooling global sur la séquence → transforme (seq_len, embed_dim) en (embed_dim) pour chaque article.
Utile pour passer à une couche fully-connected finale.

Classifier
2 couches fully-connected + dropout pour la classification.
La dernière couche = num_classes=46 → chaque neurone correspond à une catégorie.
'''

vocab_size = 20000
embed_dim = 252
num_heads = 12
ff_dim = 1024  
num_classes = 46
max_len = 200
dropout = 0.2

model = MiniTransformerClassifier(vocab_size, embed_dim, num_heads, ff_dim, num_classes, max_len, dropout)
print(model)

########## Partie 6 : Entraînement ##############

import torch.optim as optim

#hyperparametres
epochs = 15
batch_size = 128
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#deplacer model sur GPU si Disponible
model = model.to(device)

#definir loss function et optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

#Data Loader deja creer : train loader et test_loader

## Boucle d'entrainement
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from tqdm import tqdm


for epoch in range(epochs):
    # --- Entraînement ---
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_label_train = []
    all_probs_train = []

    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()  # ici tu avais une faute: setp -> step

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)

        probs = F.softmax(output, dim=1).detach().cpu()
        all_probs_train.append(probs)
        all_label_train.append(batch_y.cpu())

    # Concaténer tous les batches
    all_probs_train = torch.cat(all_probs_train, dim=0).numpy()
    all_label_train = torch.cat(all_label_train, dim=0).numpy()

    acc_train = correct / total
    auc_train = roc_auc_score(all_label_train, all_probs_train, multi_class='ovr')

    # --- Validation ---
    model.eval()
    val_correct = 0
    val_total = 0
    all_label_val = []
    all_probs_val = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Validation", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)

            pred = output.argmax(dim=1)
            val_correct += (pred==batch_y).sum().item()
            val_total += batch_y.size(0)

            probs = F.softmax(output, dim=1).cpu()
            all_probs_val.append(probs)
            all_label_val.append(batch_y.cpu())

    all_probs_val = torch.cat(all_probs_val, dim=0).numpy()
    all_label_val = torch.cat(all_label_val, dim=0).numpy()
    acc_val = val_correct / val_total
    auc_val = roc_auc_score(all_label_val, all_probs_val, multi_class='ovr')

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {acc_train:.4f}, Train AUC: {auc_train:.4f} | "
          f"Val Acc: {acc_val:.4f}, Val AUC: {auc_val:.4f}")

'''
rossEntropyLoss
Multi-class : combine LogSoftmax + NLLLoss.
Chaque sortie de model(x) est un logit, pas encore softmax.

optimizer.zero_grad()
On reset les gradients à chaque batch pour éviter accumulation.
loss.backward() + optimizer.step()
Backpropagation : calcule les gradients puis met à jour les poids.

Calcul de l’accuracy
argmax sur les logits pour prédire la classe.
Compare avec batch_y pour mesurer la performance.
'''


######################## Partie 7 : Prédiction sur de nouveaux textes ######################

model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)

        probs = F.softmax(output, dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(batch_y.cpu())

all_probs_test = torch.cat(all_probs, dim=0).numpy()
all_labels_test = torch.cat(all_labels, dim=0).numpy()

## Accuracy et Auc Final
test_acc = (all_probs_test.argmax(axis=1) == all_labels_test).mean()
test_auc = roc_auc_score(all_labels_test, all_probs_test, multi_class='ovr')

print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test AUC: {test_auc:.4f}")

#### Fonction de prédiction ###

def preprocess_text(text, word_index, max_len=200):
    tokens = text.lower().split()
    seq = [word_index.get(w, 2) for w in tokens]  #2 = OOV (hors vocabulaire)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else :
        seq = seq[:max_len]
    return torch.tensor([seq]) 

def predict(text, model, word_index, class_names, device='cpu'):
    model.eval()
    x = preprocess_text(text, word_index).to(device)
    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
    top5 = probs.argsort()[-5:][::-1]
    print("\nTexte:", text)
    print("Classe prédite:", class_names[top5[0]])
    print("Top-5 probabilités:")
    for i in top5:
        print(f"{class_names[i]}: {probs[i]:.4f}")


# Exemple de mapping Reuters (à remplacer par la vraie liste)
class_names = [f"Classe {i}" for i in range(46)]

# Supposons que word_index vienne du preprocessing Reuters
sample_text = "The company reported a strong increase in quarterly profits due to higher oil prices."
predict(sample_text, model, word_index, class_names, device)

'''
Quand tu changes de dataset :

Recréer le word_index (chaque dataset a le sien).
Changer num_classes (dépend des labels).
Adapter max_len (taille max des séquences).
Adapter num_words / vocab_size (taille du vocabulaire).
Le reste (Positional Encoding, TransformerBlock, structure globale) reste réutilisable sans changement
'''