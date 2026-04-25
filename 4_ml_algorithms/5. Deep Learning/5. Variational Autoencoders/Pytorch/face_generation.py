# ===============================
# Importation des bibliothèques nécessaires
# ===============================
import matplotlib.pyplot as plt  # pour l'affichage de graphiques et images
import os                        # pour la manipulation des chemins et fichiers
import glob                      # pour parcourir les fichiers avec motifs
import pandas as pd              # pour manipuler les DataFrames
import random                    # pour générer des nombres aléatoires
import numpy as np               # pour les calculs numériques
import cv2                       # pour traitement d'image (OpenCV)
import base64                    # pour encodage/décodage en base64
import imageio                   # pour lire/écrire des images
import torch                     # PyTorch framework
import torch.nn as nn             # pour construire des réseaux de neurones
import torch.nn.functional as F   # pour les fonctions utiles (activations, loss)
import torch.optim as optim       # pour les optimizers
import torch.utils.data as data_utils  # pour créer des DataLoader
from copy import deepcopy         # pour copier des objets sans référence
from torch.autograd import Variable  # pour autograd (variables pour calcul des gradients)
from tqdm import tqdm             # pour afficher des barres de progression
from PIL import Image             # pour manipuler des images
from sklearn.model_selection import train_test_split  # pour split train/validation

# ===============================
# Détection du device (GPU ou CPU)
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # utilisation GPU si dispo
print('Training on', DEVICE)  # affichage du device utilisé

# ===============================
# Définition des chemins du dataset et des attributs
# ===============================
DATASET_PATH = "Data/Face/lfw-deepfunneled"  # dossier des images
ATTRIBUTES_PATH = "Data/Face/lfw_attributes.txt"            # fichier des attributs

# ===============================
# Exploration du dataset : lister toutes les images
# ===============================
dataset = []
for path in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg"), recursive=True):
    person = os.path.basename(os.path.dirname(path))  # récupère le dossier parent
    dataset.append({"person": person, "path": path})  # ajouter un dictionnaire pour chaque image
    
dataset = pd.DataFrame(dataset)  # convertir en DataFrame pour manipulation facile
print(dataset.head())

# Filtrer les personnes ayant moins de 25 images
dataset = dataset.groupby("person").filter(lambda x: len(x) < 25)
dataset.head(10)  # afficher les 10 premières lignes

# Visualiser le nombre d'images par personne (top 200)
dataset.groupby("person").count()[:200].plot(kind='bar', figsize=(20,5))

# Afficher 20 images aléatoires avec leurs noms
plt.figure(figsize=(20,10))
for i in range(20):
    idx = random.randint(0, len(dataset))  # choisir un index aléatoire
    img = plt.imread(dataset.path.iloc[idx])  # lire l'image
    plt.subplot(4, 5, i+1)  # créer une sous-figure
    plt.imshow(img)  # afficher l'image
    plt.title(dataset.person.iloc[idx])  # mettre le nom en titre
    plt.xticks([])  # retirer ticks x
    plt.yticks([])  # retirer ticks y
plt.tight_layout()  # ajuster l'affichage
plt.show()  # afficher toutes les images

# ===============================
# Préparer le dataset (crop, resize et attributs)
# ===============================
def fetch_dataset(dx=80, dy=80, dimx=45, dimy=45):
    # Charger les attributs depuis le fichier texte
    df_attrs = pd.read_csv(ATTRIBUTES_PATH, sep='\t', skiprows=1) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    
    # Lister toutes les images et leurs informations
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath, fname)
                photo_id = fname[:-4].replace('_',' ').split()  # séparer nom et numéro
                person_id = ' '.join(photo_id[:-1])  # nom de la personne
                photo_number = int(photo_id[-1])     # numéro de la photo
                photo_ids.append({'person': person_id, 'imagenum': photo_number, 'photo_path': fpath})

    photo_ids = pd.DataFrame(photo_ids)  # convertir en DataFrame
    df = pd.merge(df_attrs, photo_ids, on=('person','imagenum'))  # fusionner avec les attributs
    assert len(df) == len(df_attrs), "lost some data when merging dataframes"
    
    # Lire les images, crop et resize
    all_photos = df['photo_path'].apply(imageio.imread)\
                                .apply(lambda img: img[dy:-dy, dx:-dx])\
                                .apply(lambda img: np.array(Image.fromarray(img).resize([dimx, dimy])) )
    
    all_photos = np.stack(all_photos.values).astype('uint8')  # convertir en array numpy
    all_attrs = df.drop(["photo_path","person","imagenum"], axis=1)  # garder seulement les attributs
    
    return all_photos, all_attrs

# Charger les images et attributs
data, attrs = fetch_dataset()

# Informations sur la taille des images
IMAGE_H = data.shape[1]
IMAGE_W = data.shape[2]
N_CHANNELS = 3

# Normaliser les pixels en [0,1]
data = np.array(data / 255, dtype='float32')

# Split train/validation
X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)

# Convertir en tensors PyTorch
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)

# ===============================
# Construire un simple Autoencoder fully connected
# ===============================
dim_z = 100  # dimension latente

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(45*45*3, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, dim_z),
            nn.BatchNorm1d(dim_z),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim_z, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Linear(1500, 45*45*3)
        )
      
    def encode(self, x):
        return self.encoder(x)  # encode l'image en vecteur latent
    
    def decode(self, z):
        return self.decoder(z)  # decode le vecteur latent en image
        
    def forward(self, x):
        encoded = self.encode(x)  # encoder
        decoded = self.decode(encoded)  # decoder
        return encoded, decoded

# ===============================
# Autoencoder CNN
# ===============================
class Autoencoder_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Decoder CNN
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5, stride=2),
        )
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        x = x.permute(0,3,1,2)  # permuter les dimensions pour CNN (N,C,H,W)
        encoded = self.encoder(x)
        decoded = self.decode(encoded)
        return encoded, decoded

# ===============================
# Instanciation du modèle
# ===============================
model_auto = Autoencoder().to(DEVICE)  # envoyer le modèle sur le GPU/CPU

# ===============================
# Fonctions pour le training et visualisation
# ===============================
def get_batch(data, batch_size=64):
    total_len = data.shape[0]
    for i in range(0, total_len, batch_size):
        yield data[i:min(i+batch_size, total_len)] # La fonction utilise un yield pour renvoyer un batch à la fois, sans charger tout le dataset en mémoire

def plot_gallery(images, h, w, n_row=3, n_col=6, with_title=False, titles=[]):
    plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        try:
            plt.imshow(images[i].reshape((h, w, 3)), cmap=plt.cm.gray, vmin=-1, vmax=1, interpolation='nearest')
            if with_title:
                plt.title(titles[i])
            plt.xticks(())
            plt.yticks(())
        except:
            pass

# ===============================
# Fonctions d'entrainement d'une époque et évaluation
# ===============================


def fit_epoch(model, train_x, criterion, optimizer, batch_size, is_cnn=False):
    """
    Effectue une époque d'entraînement sur le dataset train_x.
    Arguments :
        model : le modèle (autoencoder MLP ou CNN)
        train_x : données d'entraînement
        criterion : fonction de perte (MSE ici)
        optimizer : optimizer PyTorch (Adam ici)
        batch_size : taille des mini-batchs
        is_cnn : booléen pour indiquer si le modèle est CNN
    Retour :
        train_loss : perte moyenne sur cette époque
    """
    running_loss = 0.0      # cumul de la perte
    processed_data = 0      # compteur du nombre d'exemples traités
    
    # Boucle sur chaque batch
    for inputs in tqdm(get_batch(train_x, batch_size), total=int(np.ceil(len(train_x)/batch_size)), 
                       desc="Batches", ncols=100):
        if not is_cnn:
            # Pour MLP : aplatir l'image (45*45*3 = 6075)
            inputs = inputs.view(-1, 45*45*3)
        inputs = inputs.to(DEVICE)  # envoyer les données sur GPU ou CPU
        
        optimizer.zero_grad()       # réinitialiser les gradients
        
        encoder, decoder = model(inputs)  # forward pass : encoder puis decoder
        
        if not is_cnn:
            outputs = decoder.view(-1, 45*45*3)  # pour MLP, reshape sortie
        else:
            outputs = decoder.permute(0,2,3,1)   # pour CNN, remettre en (N,H,W,C)
        
        loss = criterion(outputs, inputs)       # calcul de la perte (MSE)
        loss.backward()                         # backpropagation
        optimizer.step()                        # mise à jour des poids
        
        running_loss += loss.item() * inputs.shape[0]  # accumuler perte pondérée par batch
        processed_data += inputs.shape[0]              # compter les exemples traités
    
    train_loss = running_loss / processed_data  # perte moyenne sur tous les exemples
    return train_loss

def eval_epoch(model, x_val, criterion, is_cnn=False):
    """
    Évalue le modèle sur le dataset de validation.
    Arguments :
        model : le modèle
        x_val : données de validation
        criterion : fonction de perte
        is_cnn : booléen si le modèle est CNN
    Retour :
        val_loss : perte moyenne sur le dataset de validation
    """
    running_loss = 0.0
    processed_data = 0
    model.eval()  # mettre le modèle en mode évaluation (désactive dropout, batchnorm...)
    
    # Boucle sur chaque batch
    for inputs in get_batch(x_val):
        if not is_cnn:
            inputs = inputs.view(-1, 45*45*3)  # aplatir pour MLP
        inputs = inputs.to(DEVICE)
        
        # Ne pas calculer les gradients pour l'évaluation
        with torch.set_grad_enabled(False):
            encoder, decoder = model(inputs)  
            
            if not is_cnn:
                outputs = decoder.view(-1, 45*45*3)
            else:
                outputs = decoder.permute(0,2,3,1)
            
            loss = criterion(outputs, inputs)  # calculer perte
            running_loss += loss.item() * inputs.shape[0]  # cumuler perte
            processed_data += inputs.shape[0]              # cumuler nombre d'exemples
    
    val_loss = running_loss / processed_data  # perte moyenne
    return val_loss


def train(train_x, val_x, model, epochs=15, batch_size=64, is_cnn=False):
    """
    Fonction pour entraîner le modèle sur plusieurs epochs.
    Arguments :
        train_x, val_x : datasets train et validation
        model : modèle PyTorch
        epochs : nombre d'époques
        batch_size : taille des mini-batchs
        is_cnn : booléen si le modèle est CNN
    Retour :
        history : liste des tuples (train_loss, val_loss) pour chaque époque
    """
    criterion = nn.MSELoss()                   # fonction de perte MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer Adam
    history = []                                # stocker les pertes
    
    # Boucle sur chaque époque
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=100):
        # entraînement sur le dataset train
        train_loss = fit_epoch(model, train_x, criterion, optimizer, batch_size, is_cnn)
        # évaluation sur le dataset validation
        val_loss = eval_epoch(model, val_x, criterion, is_cnn)
        # sauvegarder les pertes
        history.append((train_loss, val_loss))
    
    return history


# ===============================
# Entrainement du modèle Autoencoder
# ===============================
history = train(X_train, X_val, model_auto, epochs=50, batch_size=64)

# ===============================
# Sampling : génération aléatoire d'images depuis l'espace latent de l'autoencoder
# ===============================
z = np.random.randn(25, dim_z)  # générer 25 vecteurs latents aléatoires (dimension dim_z)
print(z.shape)  # afficher la forme (25, dim_z)

with torch.no_grad():  # désactiver le calcul des gradients (inference)
    inputs = torch.FloatTensor(z)  # convertir en tensor PyTorch
    inputs = inputs.to(DEVICE)     # envoyer sur GPU/CPU
    model_auto.eval()              # mettre le modèle en mode evaluation
    output = model_auto.decode(inputs)  # décoder les vecteurs latents en images
    plot_gallery(output.data.cpu().numpy(), IMAGE_H, IMAGE_W, n_row=5, n_col=5)  # afficher les images générées

# ===============================
# Ajouter des attributs (smile, lunettes)
# ===============================
# Sélectionner les indices des images souriantes
smile_ids = attrs['Smiling'].sort_values(ascending=False).iloc[100:125].index.values
smile_data = data[smile_ids]  # images correspondantes

# Sélectionner les indices des images sans sourire
no_smile_ids = attrs['Smiling'].sort_values(ascending=True).head(25).index.values
no_smile_data = data[no_smile_ids]

# Sélectionner les images avec lunettes normales
eyeglasses_ids = attrs['Eyeglasses'].sort_values(ascending=False).head(25).index.values
eyeglasses_data = data[eyeglasses_ids]

# Sélectionner les images avec lunettes de soleil
sunglasses_ids = attrs['Sunglasses'].sort_values(ascending=False).head(25).index.values
sunglasses_data = data[sunglasses_ids]

# Afficher ces différentes catégories
plot_gallery(smile_data, IMAGE_H, IMAGE_W, n_row=5, n_col=5, with_title=True, titles=smile_ids)
plot_gallery(no_smile_data, IMAGE_H, IMAGE_W, n_row=5, n_col=5, with_title=True, titles=no_smile_ids)
plot_gallery(eyeglasses_data, IMAGE_H, IMAGE_W, n_row=5, n_col=5, with_title=True, titles=eyeglasses_ids)
plot_gallery(sunglasses_data, IMAGE_H, IMAGE_W, n_row=5, n_col=5, with_title=True, titles=sunglasses_ids)

# ===============================
# Fonctions pour encoder/décoder depuis l'espace latent
# ===============================
def to_latent(pic):
    with torch.no_grad():
        inputs = torch.FloatTensor(pic.reshape(-1, 45*45*3))  # aplatir l'image
        inputs = inputs.to(DEVICE)
        model_auto.eval()
        output = model_auto.encode(inputs)  # encoder vers latent
        return output

def from_latent(vec):
    with torch.no_grad():
        inputs = vec.to(DEVICE)
        model_auto.eval()
        output = model_auto.decode(inputs)  # décoder latent en image
        return output

# ===============================
# Calcul des vecteurs latents moyens pour certains attributs
# ===============================
smile_latent = to_latent(smile_data).mean(axis=0)      # latent moyen des sourires
no_smile_latent = to_latent(no_smile_data).mean(axis=0)  # latent moyen sans sourire
sunglasses_latent = to_latent(sunglasses_data).mean(axis=0)  # latent moyen lunettes de soleil

# Différences de vecteurs pour appliquer l'attribut
smile_vec = smile_latent - no_smile_latent
sunglasses_vec = sunglasses_latent - smile_latent

# ===============================
# Fonctions pour ajouter attributs aux images
# ===============================
def make_me_smile(ids):
    for id in ids:
        pic = data[id:id+1]
        latent_vec = to_latent(pic)
        latent_vec[0] += smile_vec  # ajouter le vecteur sourire
        pic_output = from_latent(latent_vec)
        pic_output = pic_output.view(-1,45,45,3).cpu()  # remettre en image
        plot_gallery([pic, pic_output], IMAGE_H, IMAGE_W, n_row=1, n_col=2)  # afficher avant/après
        
def give_me_sunglasses(ids):
    for id in ids:
        pic = data[id:id+1]
        latent_vec = to_latent(pic)
        latent_vec[0] += sunglasses_vec  # ajouter le vecteur lunettes
        pic_output = from_latent(latent_vec)
        pic_output = pic_output.view(-1,45,45,3).cpu()
        plot_gallery([pic, pic_output], IMAGE_H, IMAGE_W, n_row=1, n_col=2)

# Appliquer sourire aux images sans sourire
make_me_smile(no_smile_ids)

# ===============================
# Construction d'une VAE (Variational Autoencoder)
# ===============================
dim_z = 256  # dimension latente

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(45*45*3, 1500)  # input -> couche cachée
        self.fc21 = nn.Linear(1500, dim_z)   # moyenne mu
        self.fc22 = nn.Linear(1500, dim_z)   # log variance
        self.fc3 = nn.Linear(dim_z, 1500)    # latent -> couche cachée
        self.fc4 = nn.Linear(1500, 45*45*3)  # couche sortie
        self.relu = nn.LeakyReLU()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)  # mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # écart type
        eps = torch.randn_like(std)    # bruit normal
        return eps.mul(std).add_(mu)   # échantillon latent

    def decode(self, z):
        z = self.relu(self.fc3(z))       # couche cachée
        return torch.sigmoid(self.fc4(z))  # sortie [0,1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)  # échantillonnage latent
        z = self.decode(z)                    # décoder
        return z, mu, logvar

# ===============================
# Fonction de loss pour la VAE
# ===============================
def loss_vae_fn(x, recon_x, mu, logvar):    
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # reconstruction loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return BCE + KLD

# Instanciation du modèle VAE
model_vae = VAE().to(DEVICE)

# ===============================
# Fonctions pour entrainer la VAE
# ===============================
def fit_epoch_vae(model, train_x, optimizer, batch_size, is_cnn=False):
    running_loss = 0.0
    processed_data = 0
    
    for inputs in tqdm(get_batch(train_x, batch_size), total=int(np.ceil(len(train_x)/batch_size)), 
                       desc="Batches", ncols=100):
        inputs = inputs.view(-1, 45*45*3)
        inputs = inputs.to(DEVICE)        
        optimizer.zero_grad()
        
        decoded, mu, logvar = model(inputs)  # forward pass
        outputs = decoded.view(-1, 45*45*3)
        
        loss = loss_vae_fn(inputs, outputs, mu, logvar)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.shape[0]
        processed_data += inputs.shape[0]
    
    train_loss = running_loss / processed_data    
    return train_loss

def eval_epoch_vae(model, x_val, batch_size):
    running_loss = 0.0
    processed_data = 0
    model.eval()
    
    for inputs in get_batch(x_val, batch_size=batch_size):
        inputs = inputs.view(-1, 45*45*3)
        inputs = inputs.to(DEVICE)
        
        with torch.set_grad_enabled(False):
            decoded, mu, logvar = model(inputs)
            outputs = decoded.view(-1, 45*45*3)        
            loss = loss_vae_fn(inputs, outputs, mu, logvar)
            running_loss += loss.item() * inputs.shape[0]
            processed_data += inputs.shape[0]
    
    val_loss = running_loss / processed_data
    
    # visualisation d'une image de validation
    with torch.set_grad_enabled(False):
        pic = x_val[3]         
        pic_input = pic.view(-1, 45*45*3).to(DEVICE)        
        decoded, mu, logvar = model(pic_input)        
        pic_output = decoded[0].view(-1, 45*45*3).cpu()
        pic_input = pic_input.cpu()
        plot_gallery([pic_input, pic_output], 45, 45, 1, 2)
    
    return val_loss

# ===============================
# Fonction de training complet pour la VAE
# ===============================
def train_vae(train_x, val_x, model, epochs=10, batch_size=32, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)        
    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss: {val_loss:0.4f}"
    
    
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=100):            
        train_loss = fit_epoch_vae(model, train_x, optimizer, batch_size)
        val_loss = eval_epoch_vae(model, val_x, batch_size)
        history.append((train_loss, val_loss))
        tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss, val_loss=val_loss))            
    
    return history

# ===============================
# Entrainement de la VAE
# ===============================
history_vae = train_vae(X_train, X_val, model_vae, epochs=150, batch_size=128, lr=0.001)

# Visualisation de la loss au cours des epochs
train_loss, val_loss = zip(*history_vae)
plt.figure(figsize=(15,10))
plt.plot(train_loss, label='Train loss')
plt.plot(val_loss, label='Val loss')
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# ===============================
# Fonctions pour encoder/décoder depuis l'espace latent
# ===============================
def to_latentVAE(pic):
    with torch.no_grad():
        inputs = torch.FloatTensor(pic.reshape(-1, 45*45*3))  # aplatir l'image
        inputs = inputs.to(DEVICE)
        model_vae.eval()
        output = model_vae.encode(inputs)  # encoder vers latent
        return output

def from_latentVAE(vec):
    with torch.no_grad():
        inputs = vec.to(DEVICE)
        model_vae.eval()
        output = model_vae.decode(inputs)  # décoder latent en image
        return output

# ===============================
# Calcul des vecteurs latents moyens pour certains attributs
# ===============================
mu, logvar = to_latentVAE(smile_data)   # récupérer mu et logvar séparément
smile_latentVAE = mu.mean(axis=0)

mu, logvar = to_latentVAE(no_smile_data)
no_smile_latentVAE = mu.mean(axis=0)  # latent moyen sans sourire

mu, logvar = to_latentVAE(sunglasses_data)
sunglasses_latentVAE = mu.mean(axis=0)  # latent moyen lunettes de soleil

# Différences de vecteurs pour appliquer l'attribut
smile_vecVAE = smile_latentVAE - no_smile_latentVAE
sunglasses_vecVAE = sunglasses_latentVAE - smile_latentVAE

# ===============================
# Fonctions pour ajouter attributs aux images
# ===============================
def make_me_smileVAE(ids):
    for id in ids:
        pic = data[id:id+1]
        mu, logvar = to_latentVAE(pic)  # récupérer mu uniquement
        latent_vec = mu.clone()         # copier pour pouvoir modifier
        latent_vec[0] += smile_vecVAE  # ajouter le vecteur sourire
        pic_output = from_latentVAE(latent_vec)
        pic_output = pic_output.view(-1,45,45,3).cpu()  # remettre en image
        plot_gallery([pic, pic_output], IMAGE_H, IMAGE_W, n_row=1, n_col=2)  # afficher avant/après
        
def give_me_sunglassesVAE(ids):
    for id in ids:
        pic = data[id:id+1]
        mu, logvar = to_latentVAE(pic)
        latent_vec = mu.clone()
        latent_vec[0] += sunglasses_vecVAE  # ajouter le vecteur lunettes
        pic_output = from_latentVAE(latent_vec)
        pic_output = pic_output.view(-1,45,45,3).cpu()
        plot_gallery([pic, pic_output], IMAGE_H, IMAGE_W, n_row=1, n_col=2)

# Appliquer sourire aux images sans sourire
make_me_smileVAE(no_smile_ids)
give_me_sunglassesVAE(no_smile_ids)
