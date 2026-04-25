import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Chargement des données
# ----------------------------
feature = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
           "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
           "root_shell","su_attempted","num_root","num_file_creations","num_shells",
           "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
           "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
           "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
           "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
           "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
           "label","difficulty"]

train='Data/KDDTrain+.txt'
test='Data/KDDTest+.txt'
test21='Data/KDDTest-21.txt'

train_data=pd.read_csv(train,names=feature)
test_data=pd.read_csv(test,names=feature)
test_data21=pd.read_csv(test21,names=feature)

data = pd.concat([train_data, test_data], ignore_index=True)
data.drop(['difficulty'], axis=1, inplace=True)

# ----------------------------
# 2️⃣ Regroupement des attaques
# ----------------------------
def change_label(df):
    df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable',
                      'smurf','teardrop','udpstorm','worm'],'Dos', inplace=True)
    df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named',
                      'phf','sendmail','snmpgetattack','snmpguess','spy','warezclient',
                      'warezmaster','xlock','xsnoop'],'R2L', inplace=True)
    df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe', inplace=True)
    df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R', inplace=True)

change_label(data)

# ----------------------------
# 3️⃣ Prétraitement
# ----------------------------
numeric_col = data.select_dtypes(include='number').columns
scaler = StandardScaler()
data[numeric_col] = scaler.fit_transform(data[numeric_col])

# Label encoding
le = LabelEncoder()
data['intrusion'] = le.fit_transform(data['label'])
data.drop(['label'], axis=1, inplace=True)

# One-hot pour catégoriel
data = pd.get_dummies(data, columns=['protocol_type','service','flag'])
data = data.astype(np.float32)

# Séparation X / y
y_data = LabelBinarizer().fit_transform(data['intrusion'])
X_data = data.drop(['intrusion'], axis=1).values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Reshape pour RNN [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Conversion en tenseurs
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5000)

# ----------------------------
# 4️⃣ Définition du modèle
# ----------------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=5, rnn_type='RNN', dropout=0.2):
        super(RNNClassifier, self).__init__()
        if rnn_type=='RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        elif rnn_type=='LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        elif rnn_type=='GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # prendre la dernière timestep
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

input_size = X_train.shape[2]
model_rnn = RNNClassifier(input_size=input_size, rnn_type='RNN')
model_lstm = RNNClassifier(input_size=input_size, rnn_type='LSTM')
model_gru = RNNClassifier(input_size=input_size, rnn_type='GRU')

# ----------------------------
# 5️⃣ Fonction d'entraînement
# ----------------------------
def train_model(model, train_loader, epochs=100, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'loss':[], 'accuracy':[]}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            predicted = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y_batch, dim=1)
            correct += (predicted==labels).sum().item()
            total += X_batch.size(0)
        history['loss'].append(epoch_loss/total)
        history['accuracy'].append(correct/total)
        if (epoch+1)%10==0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {history['loss'][-1]:.4f}, Accuracy: {history['accuracy'][-1]:.4f}")
    return history

# ----------------------------
# 6️⃣ Entraînement
# ----------------------------
history_rnn = train_model(model_rnn, train_loader)
history_lstm = train_model(model_lstm, train_loader)
history_gru = train_model(model_gru, train_loader)

# ----------------------------
# 7️⃣ Évaluation
# ----------------------------
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y_batch, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

print("🔹 RNN")
evaluate_model(model_rnn, test_loader)
print("🔹 LSTM")
evaluate_model(model_lstm, test_loader)
print("🔹 GRU")
evaluate_model(model_gru, test_loader)

# ----------------------------
# 8️⃣ Prédiction utilisateur
# ----------------------------
user_input = {
    'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
    'src_bytes': 181, 'dst_bytes': 5450, 'land': 0,
    'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0,
    'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'count': 2,
    'srv_count': 2, 'serror_rate': 0, 'srv_serror_rate': 0, 'rerror_rate': 0,
    'srv_rerror_rate': 0, 'same_srv_rate': 1, 'diff_srv_rate': 0, 'srv_diff_host_rate': 0,
    'dst_host_count': 5, 'dst_host_srv_count': 5, 'dst_host_same_srv_rate': 1,
    'dst_host_diff_srv_rate': 0, 'dst_host_same_src_port_rate': 0, 'dst_host_srv_diff_host_rate': 0,
    'dst_host_serror_rate': 0, 'dst_host_srv_serror_rate': 0, 'dst_host_rerror_rate': 0,
    'dst_host_srv_rerror_rate': 0
}
user_df = pd.DataFrame([user_input])
user_df[numeric_col.drop('intrusion', errors='ignore')] = scaler.transform(user_df[numeric_col.drop('intrusion', errors='ignore')])
user_df = pd.get_dummies(user_df, columns=['protocol_type','service','flag'])
# Ajouter colonnes manquantes
missing_cols = set(data.drop(['intrusion'], axis=1).columns) - set(user_df.columns)
for c in missing_cols: user_df[c] = 0
user_df = user_df[data.drop(['intrusion'], axis=1).columns]
X_user = torch.tensor(user_df.values.reshape(1,1,-1), dtype=torch.float32)

model_lstm.eval()
with torch.no_grad():
    pred = model_lstm(X_user)
    predicted_class = torch.argmax(pred, dim=1).item()
    predicted_label = le.inverse_transform([predicted_class])
    print("Classe prédite utilisateur:", predicted_label[0])
