# ==============================================
# Importation des librairies
# ==============================================
from __future__ import print_function
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets
from torchvision.utils import save_image

from tqdm import tqdm_notebook as tqdm
from PIL import Image
import cv2


# ==============================================
# Chargement et visualisation des données
# ==============================================

# Dossier contenant les images saines
PATH1 = '../input/diabetic-retinopathy-dataset/Healthy/'
images = os.listdir(PATH1)
print(f"Il y a {len(images)} images en bonne santé")

# Affichage d'un échantillon d'images saines
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
for indx, axis in enumerate(axes.flatten()):
    rnd_indx = np.random.randint(0, len(images))
    img = plt.imread(PATH1 + images[rnd_indx])
    axis.imshow(img)
    axis.set_title(images[rnd_indx])
    axis.set_axis_off()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Dossier contenant les images sévères
PATH2 = '../input/diabetic-retinopathy-dataset/Severe DR/'
images2 = os.listdir(PATH2)
print(f"Il y a {len(images2)} images de sévérité grave")

# Affichage d'un échantillon d'images sévères
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
for indx, axis in enumerate(axes.flatten()):
    rnd_indx = np.random.randint(0, len(images2))
    img = plt.imread(PATH2 + images2[rnd_indx])
    axis.imshow(img)
    axis.set_title(images2[rnd_indx])
    axis.set_axis_off()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ==============================================
# Prétraitement des images
# ==============================================

batch_size = 32
image_size = 64

# Transformation de base : redimensionnement, normalisation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Chargement des données
train_data = datasets.ImageFolder(root='../input/diabetic-retinopathy-dataset/', transform=transform)
dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Exemple d'un batch
imgs, label = next(iter(dataloader))
imgs = imgs.numpy().transpose(0, 2, 3, 1)

# Transformation avec augmentation de données (rotation, jitter, flip, etc.)
random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
transform_aug = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(random_transforms, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = datasets.ImageFolder('../input/diabetic-retinopathy-dataset/', transform=transform_aug)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)


# ==============================================
# Fonctions utilitaires
# ==============================================

def weights_init(m):
    """Initialisation des poids pour les couches Conv et BatchNorm"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ==============================================
# Définition du générateur (DCGAN classique)
# ==============================================

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)


# ==============================================
# Définition du discriminateur (DCGAN classique)
# ==============================================

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input).view(-1)


# ==============================================
# Entraînement simple (1 epoch de test)
# ==============================================

EPOCH = 1
LR = 0.0001

netG = G()
netD = D()
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

# Boucle d'entraînement
for epoch in range(EPOCH):
    for i, data in enumerate(dataloader, 0):
        
        # ---------------------
        # 1. Mise à jour du discriminateur
        # ---------------------
        netD.zero_grad()
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)

        # Bruit pour générateur
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise)
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        # Perte totale du D
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # ---------------------
        # 2. Mise à jour du générateur
        # ---------------------
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))  # Le générateur veut tromper D
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()

        # Affichage
        print('[%d/%d][%d/%d] Loss_D: %.4f; Loss_G: %.4f' %
              (epoch, EPOCH, i, len(dataloader), errD.item(), errG.item()))

        if i % 100 == 0:
            vutils.save_image(real, './results/real_samples.png', normalize=True)
            fake = netG(noise)
            vutils.save_image(fake.data, './results/fake_samples_epoch_%03d.png' % epoch, normalize=True)

