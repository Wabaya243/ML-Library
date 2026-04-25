import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM


vocab_size = 15000 # on veut juste 15000 mots les plus courant
max_len = 300 

#On charge les données
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len, padding ='post') # post pour faire les remplissage  a la fin
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

#Imprimer les jeux des donnes
print(f"Donnes d'entrainement : {x_train.shape}")
print(f"Donnes de test : {x_test.shape}")

#On crée le modèle
model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        SimpleRNN(unites=128, activation='tanh', return_sequences=False),
        Dense(1, activation='sigmoid')

    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss , accuracy = model.evaluate(x_test, y_test)
print(f'Loss : {loss}')
print(f'accuracy : {accuracy}')


##### Version Pytorch
import torch
import torch.nn as nn
import torch.nn.functionnel as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

vocab_size_pytorch = 15000
max_len_py = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#On charge les données
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len, padding ='post') # post pour faire les remplissage  a la fin
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')

train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class RNNmodel(nn.Module):
    def __init__(self, vocab_size_pytorch, embedding_dim, hidden_dim, output_dim):
        super(RNNmodel, self).__init__()
        self.embedding = nn.Embedding(vocab_size_pytorch, embedding_dim)
        self.rnn = nn.SimpleRNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output , hidden = self.rnn(embedded)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))

model = RNNmodel(vocab_size_pytorch, 128, 128, 1).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()   
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x_batch, y_batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        train_accuracy = evaluate_model(model, train_loader)
        print(f"Train Accuracy: {train_accuracy}")
        test_accuracy = evaluate_model(model, test_loader)
        print("------------------")
        print(f"Test Accuracy: {test_accuracy}")


train_model(model, train_loader, criterion, optimizer, epochs=10)


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            outputs = model(x_batch)
            predicted = (outputs.squeeze() > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch.float()).sum().item()
    return correct / total

test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy}")










