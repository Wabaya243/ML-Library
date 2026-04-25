# multimodal_lab.py
# Multimodal Lab (Image + Text) in PyTorch
# CIFAR-10 + synthetic captions

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# 1) Config
# -------------------------

IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 50
EMBED_DIM = 128
TEXT_SEQ_LEN = 15
TRANSFORMER_UNITS = 512
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 6
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

label_names = ['Avion','automobile','oiseaux','chatte','cerf','chienne','grenouille','cheval','bateau','truck']

# -------------------------
# 2) Load CIFAR-10 + captions
# -------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
test_data  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)


# Fonction pour générer une légende (caption) à partir d’un label CIFAR-10
def make_caption(lbl):
    """
    Génère une légende contextuelle cohérente avec la classe CIFAR-10
    sans jamais révéler le nom du label.
    Objectif : encourager la fusion image + texte réelle.
    """

    # Contexte probable selon la classe
    context_map = {
    0: [  # avion
        "dans le ciel", "au-dessus des nuages", "en vol", "sur une piste de décollage",
        "au-dessus de la mer", "dans un aéroport"
    ],
    1: [  # automobile
        "sur la route", "en ville", "dans un parking", "en mouvement sur l’asphalte",
        "près d’un feu de circulation", "dans un embouteillage"
    ],
    2: [  # oiseau
        "dans les arbres", "en vol", "dans un jardin", "au milieu du feuillage",
        "sur une branche", "près d’un nid"
    ],
    3: [  # chatte
        "dans une maison", "sur un canapé", "près d’une fenêtre", "dans un panier",
        "sur une couverture", "dans une pièce lumineuse"
    ],
    4: [  # cerf
        "dans la forêt", "au milieu des arbres", "dans un champ", "en train de courir",
        "près d’un ruisseau", "dans une clairière"
    ],
    5: [  # chienne
        "dans un jardin", "sur un trottoir", "dans la nature", "près d’un humain",
        "dans un parc", "au bord d’un chemin"
    ],
    6: [  # grenouille
        "près d’un étang", "dans l’eau", "sur une feuille", "au bord d’un marais",
        "dans un bassin", "près d’un rocher humide"
    ],
    7: [  # cheval
        "dans un pré", "au galop", "près d’une clôture", "dans une étable",
        "sur une prairie", "près d’un champ ouvert"
    ],
    8: [  # bateau
        "sur l’eau", "près d’un port", "au large", "dans un lac",
        "près du rivage", "dans une marina"
    ],
    9: [  # truck
        "sur une route", "dans un chantier", "transportant une cargaison", "près d’un entrepôt",
        "sur une autoroute", "stationné sur le bas-côté"
    ]
}

    adjectifs = [
        "petit", "grand", "magnifique", "coloré", "adorable", "rapide",
        "vif", "paisible", "joli", "impressionnant", "dynamique"
    ]
    actions = [
        "en mouvement", "en train de se reposer", "capturé de loin",
        "vu de près", "pris sur le vif", "photographié en action"
    ]

    # Récupère un contexte probable selon le label
    context = random.choice(context_map.get(int(lbl), ["dans une scène inconnue"]))
    template = random.choice([
        f"une image {random.choice(adjectifs)} {random.choice(actions)} {context}",
        f"photo {random.choice(adjectifs)} prise {context}",
        f"un sujet {random.choice(actions)} {context}",
        f"une scène {random.choice(adjectifs)} {context}",
        f"une capture montrant quelque chose {random.choice(adjectifs)} {context}"
    ])

    # 10 % de légendes volontairement neutres/bruitées
    if random.random() < 0.1:
        bruit = [
            "image difficile à interpréter",
            "on distingue vaguement une forme",
            "photo floue ou abstraite",
            "on devine un mouvement dans l’image",
            "rien de très clair à première vue"
        ]
        return random.choice(bruit)

    return template

def make_caption_val(lbl):
    """
    Génère des légendes de validation/test avec un style grammatical différent
    pour tester la vraie généralisation du modèle.
    Moins d’adjectifs, phrases plus descriptives, plus naturelles.
    """

    context_map = {
        0: ["visible dans le ciel", "volant haut au-dessus du sol", "se déplaçant entre les nuages"],
        1: ["roulant sur la route", "garé dans la rue", "détecté près d’autres véhicules"],
        2: ["posé sur une branche", "en vol au-dessus d’un jardin", "visible dans la végétation"],
        3: ["installé sur un meuble", "près d’un coussin ou d’un tissu", "posé dans un intérieur calme"],
        4: ["au milieu de la forêt", "dans un espace boisé", "visible sur un terrain naturel"],
        5: ["marchant près d’un humain", "dans un espace ouvert", "associé à une activité extérieure"],
        6: ["reposant sur une surface humide", "près de l’eau", "visible sur une feuille ou une pierre"],
        7: ["en mouvement dans un champ", "près d’une clôture ou d’un pré", "debout sur un terrain dégagé"],
        8: ["flottant sur l’eau", "amarré près du rivage", "dans une zone portuaire"],
        9: ["sur une voie de circulation", "dans un environnement industriel", "chargé de marchandises"],
    }

    intro = [
        "L’image semble montrer",
        "On observe",
        "La scène présente",
        "Cette image illustre",
        "Il est possible de voir",
        "La photo capture",
        "On distingue"
    ]

    fin = [
        "dans un environnement naturel.",
        "dans un cadre urbain.",
        "dans une ambiance calme.",
        "dans une situation dynamique.",
        "dans des conditions lumineuses normales."
    ]

    context = random.choice(context_map.get(int(lbl), ["dans un lieu indéfini"]))
    sentence = f"{random.choice(intro)} quelque chose {context} {random.choice(fin)}"

    # 10 % de phrases neutres / bruitées
    if random.random() < 0.1:
        bruit = [
            "photo partiellement floue, sujet peu identifiable.",
            "on perçoit une forme sans certitude.",
            "l’image ne révèle pas clairement le sujet.",
            "contenu difficile à interpréter.",
            "détails visuels limités."
        ]
        return random.choice(bruit)

    return sentence



# Génère les légendes pour chaque image du jeu d’entraînement et de test
captions_train = [make_caption(y) for _, y in train_data]
captions_test  = [make_caption_val(y) for _, y in test_data]


# Initialisation du tokenizer Keras (convertit le texte en séquences numériques)
tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")  # Limite le vocabulaire à 2000 mots, gère les mots inconnus
tokenizer.fit_on_texts(captions_train)  # Apprend le vocabulaire à partir des légendes d’entraînement


# Fonction utilitaire pour convertir les phrases en séquences numériques et les normaliser à une longueur fixe
def texts_to_padded_sequences(texts, maxlen=TEXT_SEQ_LEN):
    seq = tokenizer.texts_to_sequences(texts)  # Convertit les mots en indices
    seq = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')  # Ajoute des zéros à la fin si trop court
    return np.array(seq)  # Retourne un tableau NumPy prêt pour PyTorch


# Application de la conversion aux légendes d’entraînement et de test
seq_train = texts_to_padded_sequences(captions_train)
seq_test  = texts_to_padded_sequences(captions_test)


# Normalisation des images (0–255 → 0–1) et conversion en float32
x_train = train_data.data.astype("float32") / 255.0
y_train = np.array(train_data.targets)  # Labels d’entraînement
x_test  = test_data.data.astype("float32") / 255.0
y_test  = np.array(test_data.targets)   # Labels de test


# Division du jeu d’entraînement en sous-ensembles : entraînement (85%) et validation (15%)
x_train, x_val, seq_train, seq_val, y_train, y_val = train_test_split(
    x_train, seq_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# --- Recréer les captions de validation avec un autre style ---
captions_val = [make_caption_val(y) for y in y_val]
seq_val = texts_to_padded_sequences(captions_val)


# - test_size=0.15 : 15 % des données pour la validation
# - random_state=42 : pour des résultats reproductibles
# - stratify=y_train : conserve les mêmes proportions de classes dans les deux ensembles

# -------------------------
# 3) PyTorch Dataset
# -------------------------

class MultimodalDataset(Dataset):
    def __init__(self, images, seqs, labels, transform=None):
        self.images = images
        self.seqs = seqs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.tensor(img).permute(2,0,1)  # HWC->CHW
        if self.transform:
            img = self.transform(img)
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (img, seq), label

train_ds = MultimodalDataset(x_train, seq_train, y_train)
val_ds   = MultimodalDataset(x_val, seq_val, y_val)
test_ds  = MultimodalDataset(x_test, seq_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

vocab_size = min(2000, len(tokenizer.word_index) + 1)

# -------------------------
# 4) Model components
# -------------------------

# ENCODEUR D'IMAGE (ImageEncoder)
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        # Petit CNN pour extraire des caractéristiques visuelles des images CIFAR-10
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # Conv → ReLU → Réduction de moitié (16x16)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # Conv → ReLU → Réduction (8x8)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()                   # Conv → ReLU (sortie : 128 canaux)
        )
        # Pooling adaptatif pour obtenir une seule valeur par canal (résumé global de l’image)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # Couche fully-connected pour projeter les features en vecteur d’embedding
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.conv(x)                 # Passe dans les convolutions
        x = self.pool(x).view(x.size(0), -1)  # Pooling global + mise à plat → (batch, 128)
        return self.fc(x)                # Projection finale → (batch, embed_dim)
    
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len):
        super().__init__()
        # Embedding pour les tokens (mots)
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Embedding pour les positions (ordre des mots)
        self.pos_emb   = nn.Embedding(seq_len, embed_dim)

        # Bloc Transformer : encode la séquence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_UNITS,
            batch_first=True   # évite de permuter les dimensions
        )
        # Empile plusieurs couches Transformer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)

    def forward(self, seq):
        # Crée un vecteur d'indices de position
        positions = torch.arange(0, seq.size(1), device=seq.device)
        # Passe dans l'embedding de position
        positions = self.pos_emb(positions).unsqueeze(0)   # (1, seq, embed)

        # Somme des embeddings (token + position)
        x = self.token_emb(seq) + positions                # (B, seq, embed)
        # Passage dans le Transformer
        x = self.transformer(x)                            # (B, seq, embed)
        # Moyenne sur la séquence → vecteur global
        x = x.mean(dim=1)                                  # (B, embed)
        return x


class MultimodalModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_classes=10):
        super().__init__()
        # Encodeur image
        self.img_enc = ImageEncoder(embed_dim)
        # Encodeur texte
        self.txt_enc = TextEncoder(vocab_size, embed_dim, TEXT_SEQ_LEN)

        # Fusion texte + image avec un petit Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classifieur final
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img, seq):
        # Encode séparément image et texte
        img_emb = self.img_enc(img)  # (B, embed)
        txt_emb = self.txt_enc(seq)  # (B, embed)

        # Empile les deux modalités pour la fusion
        fused = torch.stack([img_emb, txt_emb], dim=0)  # (2, B, embed)
        # Passage dans le Transformer de fusion
        x = self.fusion(fused)                          # (2, B, embed)
        # Moyenne des deux représentations
        x = x.mean(dim=0)                               # (B, embed)
        # Classification finale
        return self.fc(x)

# -------------------------
# 5) Training Loop
# -------------------------

from tqdm import tqdm

model = MultimodalModel(vocab_size).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

strong_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
])

def train_model(mask_prob=0.3, misalign_prob=0.1):
    """
    Entraîne le modèle multimodal en réduisant la dépendance excessive au texte.
    - mask_prob : proportion d’exemples où le texte est complètement masqué.
    - misalign_prob : proportion d’exemples où le texte est remplacé par une légende d’une autre image.
    """
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct = 0, 0

        for (imgs, seqs), labels in tqdm(train_loader, desc=f"{epoch+1}/{EPOCHS}"):
            imgs, seqs, labels = imgs.to(DEVICE), seqs.to(DEVICE), labels.to(DEVICE)

            # --- Masquage du texte (30 % du temps) ---
            mask = torch.rand(seqs.size(0)) < mask_prob
            seqs[mask] = 0  # texte masqué → séquence de zéros

            # --- Faux appariements texte-image (10 % du temps) ---
            misalign = torch.rand(seqs.size(0)) < misalign_prob
            if misalign.any():
                idx_perm = torch.randperm(seqs.size(0))
                seqs[misalign] = seqs[idx_perm[misalign]]  # mélange des textes entre exemples

            # --- Augmentations visuelles fortes ---
            imgs_aug = torch.stack([strong_transform(img.cpu()).to(DEVICE) for img in imgs])

            optimizer.zero_grad()
            outputs = model(imgs_aug, seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = total_correct / len(train_ds)
        val_acc, val_loss = evaluate(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_ds):.4f} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
def evaluate(loader):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for (imgs, seqs), labels in loader:
            imgs, seqs, labels = imgs.to(DEVICE), seqs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs, seqs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()*labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

train_model()

model.eval()
test_acc, test_loss = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")


torch.save(model.state_dict(), "Save/multimodal_model.pth")
tokenizer_json = tokenizer.to_json()  # <-- convertit le tokenizer en JSON
with open("Save/tokenizer.json","w") as f: f.write(tokenizer_json)

from tensorflow.keras.preprocessing.text import tokenizer_from_json
with open("Save/tokenizer.json") as f:
    data = f.read()
tokenizer = tokenizer_from_json(data)


# -------------------------
# 6) Predict on example
# -------------------------
def predict_image_text(img_array, text):
    seq = texts_to_padded_sequences([text], maxlen=TEXT_SEQ_LEN)
    seq = torch.tensor(seq, dtype=torch.long).to(DEVICE)
    img = torch.tensor(img_array).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(img, seq), dim=1)
    pred_idx = probs.argmax(1).item()
    return pred_idx, probs.cpu().numpy()[0]

idx = 10
img_ex = x_test[idx]
text_ex = captions_test[idx]
pred_idx, probs = predict_image_text(img_ex, text_ex)
print("Ground truth:", label_names[y_test[idx]])
print("Caption:", text_ex)
print("Prediction:", label_names[pred_idx])


##### tester avec une image perso

from PIL import Image
import numpy as np

# Charger ton image perso
img_path = "Data/chat.jpg"
img = np.array(Image.open(img_path).convert('RGB').resize((32,32))).astype('float32') / 255.0
# Tester le poids du texte vs image
captions = [
    "la photo d'un animal sur un canapé",  # cohérente
    "un camion sur une route",             # complètement fausse
    "photo floue d’un truc bizarre",       # neutre/bruitée
]
for cap in captions:
    pred_idx, probs = predict_image_text(img, cap)
    print(f"Caption: {cap} -> Prediction: {label_names[pred_idx]}")