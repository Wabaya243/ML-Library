# ============================
# IMPORTS
# ============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================
# LECTURE ET NETTOYAGE DU TEXTE
# ============================
with open("Data/alice_in_wonderland.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    line = line.strip().lower()
    line = line.encode("ascii", "ignore").decode()
    if len(line) == 0:
        continue
    cleaned_lines.append(line)

text = " ".join(cleaned_lines)

print("Extrait du texte :")
print(text[:500])


# ============================
# ENCODAGE DES CARACTÈRES
# ============================
chars = sorted(list(set(text)))  # trié pour garder un ordre stable
nb_chars = len(chars)
print("Nombre de caractères uniques :", nb_chars)

char2index = {c: i for i, c in enumerate(chars)}
index2char = {i: c for i, c in enumerate(chars)}


# ============================
# CRÉATION DES SÉQUENCES
# ============================
SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i:i+SEQLEN])
    label_chars.append(text[i+SEQLEN])

# One-hot encoding
x = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.float32)
y = np.zeros((len(input_chars), nb_chars), dtype=np.float32)

for i, seq in enumerate(input_chars):
    for j, ch in enumerate(seq):
        x[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1

# Conversion en tenseurs PyTorch
X = torch.tensor(x)  # shape : [nb_seq, SEQLEN, nb_chars]
Y = torch.tensor(y)  # shape : [nb_seq, nb_chars]


# ============================
# DÉFINITION DU MODÈLE PyTorch
# ============================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x : [batch, seq_len, input_size]
        out, hidden = self.rnn(x)  # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]        # garder seulement le dernier état de la séquence
        out = self.fc(out)         # [batch, output_size]
        return out

HIDDEN_SIZE = 128
model = CharRNN(nb_chars, HIDDEN_SIZE, nb_chars)


# ============================
# ENTRAÎNEMENT
# ============================
criterion = nn.CrossEntropyLoss()    # adapté pour classification multi-classes
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

BATCH_SIZE = 128
NUM_ITERATIONS = 10
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

# Fonction utilitaire : générer du texte
def generate_text(model, seed, length=100):
    model.eval()
    generated = seed
    test_chars = seed
    for _ in range(length):
        # Préparer l’entrée
        Xtest = np.zeros((1, SEQLEN, nb_chars), dtype=np.float32)
        for j, ch in enumerate(test_chars):
            Xtest[0, j, char2index[ch]] = 1
        Xtest = torch.tensor(Xtest)

        # Prédire
        with torch.no_grad():
            pred = model(Xtest)
            pred_idx = torch.argmax(pred, dim=1).item()
            ypred = index2char[pred_idx]

        generated += ypred
        test_chars = test_chars[1:] + ypred
    return generated

# Boucle d’entraînement
for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration:", iteration)

    model.train()
    for epoch in range(NUM_EPOCHS_PER_ITERATION):
        permutation = torch.randperm(X.size(0))
        for i in range(0, X.size(0), BATCH_SIZE):
            idx = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X[idx], Y[idx]

            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, torch.argmax(batch_y, dim=1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Génération après entraînement
    test_idx = np.random.randint(len(input_chars))
    seed = input_chars[test_idx]
    print("Seed:", seed)
    print(generate_text(model, seed, NUM_PREDS_PER_EPOCH))

print("\n=== Fin de l’entraînement et génération ===")
