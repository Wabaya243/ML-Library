# ======================
# 📦 Imports
# ======================
import numpy as np
import pandas as pd
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Tokenizer et padding (on garde Keras pour ça)
from tensorflow.keras.preprocessing import text, sequence

# Stopwords (scikit-learn pour éviter NLTK)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = set(ENGLISH_STOP_WORDS)


# ======================
# 📂 Chargement des données
# ======================
train = pd.read_csv('Data/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('Data/validation.csv')

# Supprimer colonnes inutiles
train.drop(["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], 
           axis=1, inplace=True)


# ======================
# 🧹 Prétraitement du texte
# ======================
def preprocess_text(text):
    if type(text) is not str:
        text = str(text)
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
    text = re.sub('\s+',' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Appliquer au dataset
train['comment_text'] = train['comment_text'].apply(lambda x: preprocess_text(x))


# ======================
# ✂️ Split train / test
# ======================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train.comment_text.values,
    train.toxic.values,
    stratify=train.toxic.values,
    random_state=42,
    test_size=0.2,
    shuffle=True
)


# ======================
# 🔡 Tokenisation et padding
# ======================
token = text.Tokenizer(num_words=None)
token.fit_on_texts(list(X_train) + list(X_test))

X_train_seq = token.texts_to_sequences(X_train)
X_test_seq = token.texts_to_sequences(X_test)

max_len = 100
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_len)

word_index = token.word_index


# ======================
# 📦 Dataset PyTorch
# ======================
class ToxicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ToxicDataset(X_train_pad, y_train)
test_dataset = ToxicDataset(X_test_pad, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# ======================
# 🧠 Modèle PyTorch (Embedding + RNN bidirectionnel + Dense)
# ======================
class ToxicModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(ToxicModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 1)  # *2 car bidirectionnel
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]   # Dernier état caché
        out = self.fc(out)
        return self.sigmoid(out)

# Instancier le modèle
vocab_size = len(word_index) + 1
embed_dim = 256
hidden_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicModel(vocab_size, embed_dim, hidden_dim).to(device)


# ======================
# ⚙️ Entraînement
# ======================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# ======================
# 📊 Évaluation
# ======================
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch).squeeze()
        preds = (outputs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"✅ Test Accuracy: {correct/total:.4f}")


# ======================
# 🔮 Test sur un nouveau texte
# ======================
def predict_text(text_input):
    text_input = preprocess_text(text_input)
    seq = token.texts_to_sequences([text_input])
    pad = sequence.pad_sequences(seq, maxlen=max_len)
    tensor = torch.tensor(pad, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor).item()
    return pred

# Exemple
new_text = "I hate you, you are so dumb!"
pred = predict_text(new_text)
if pred > 0.5:
    print("🚨 Commentaire toxique détecté")
else:
    print("✅ Commentaire non toxique")


# ======================
# 📂 Prédiction finale sur dataset test
# ======================
test = pd.read_csv('Data/test.csv')
test['comment_text'] = test['comment_text'].apply(lambda x: preprocess_text(x))
test_seq = token.texts_to_sequences(test['comment_text'].values)
test_pad = sequence.pad_sequences(test_seq, maxlen=max_len)

test_tensor = torch.tensor(test_pad, dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    preds = model(test_tensor).cpu().numpy()

test['toxic'] = preds
test[['id', 'toxic']].to_csv('Data/submission.csv', index=False)
