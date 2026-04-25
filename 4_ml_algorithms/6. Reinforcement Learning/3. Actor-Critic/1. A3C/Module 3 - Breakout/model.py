# IA pour Breakout

# Import des librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialisation et réglage de la variance d’un tenseur de poids
def normalized_columns_initializer(weights, std=1.0):
    '''
    Initialise les poids d’un layer (souvent la sortie d’un réseau) de manière à ce que chaque colonne ait une variance égale à std^2.
    Très utilisé pour les couches finales de l’Actor et du Critic dans A3C.
    Garantit que les sorties du réseau au départ sont dans une plage raisonnable, ni trop grandes, ni trop petite
    Dans A3C, la couche de sortie de l’Actor produit les logits des actions.
    Si les poids sont mal initialisés, certaines actions seront trop favorisées dès le départ → apprentissage instable.
    '''
    out = torch.randn(weights.size())
    # Grâce à cette initialisation, var(out) = std^2
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

# Initialisation des poids du réseau de manière optimale pour l'apprentissage
def weights_init(m):
    '''
    Initialise tous les autres modules du réseau (Convolutions et Linear) de manière classique et stable.
    Utilise la méthode de Glorot/Xavier, adaptée aux couches convolutives et fully-connected.
    Assure que la variance des activations reste stable à travers le réseau, surtout pour les couches cachées.
     Sans cette initialisation, certaines couches pourraient saturer (ReLU → 0) ou exploser → apprentissage instable.
    '''
    classname = m.__class__.__name__  # Astuce Python pour récupérer le type de connexion dans l’objet "m" (Conv ou fully-connected)
    
    # Si la connexion est une convolution
    if 'Conv' in classname:
        weight_shape = list(m.weight.shape)  # Liste contenant la forme des poids dans l’objet "m"
        fan_in = np.prod(weight_shape[1:4])  # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]  # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out))  # borne des poids

        # Génération de poids aléatoires proportionnels à la taille du tenseur
        with torch.no_grad():
            m.weight.uniform_(-w_bound, w_bound)
            # Initialisation des biais à zéro
            if m.bias is not None:
                m.bias.fill_(0)

    # Si la connexion est fully-connected
    elif 'Linear' in classname:
        weight_shape = list(m.weight.shape)  # Liste contenant la forme des poids dans l’objet "m"
        fan_in = weight_shape[1]  # dim1
        fan_out = weight_shape[0]  # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out))  # borne des poids

        # Génération de poids aléatoires proportionnels à la taille du tenseur
        with torch.no_grad():
            m.weight.uniform_(-w_bound, w_bound)
            # Initialisation des biais à zéro
            if m.bias is not None:
                m.bias.fill_(0)

# Création du "cerveau" A3C

class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, num_inputs, 42, 42) # taille de l'image
    
            conv_out_size = self.flatten(self.conv4(self.conv3(self.conv2(self.conv1(dummy))))).shape[1]

        self.lstm = nn.LSTMCell(conv_out_size, 256)
        
        self.fc1 = nn.Linear(256, 128)
        num_outputs = action_space.n
        
        self.critic_linear = nn.Linear(128, 1)  # output = V(S)
        self.actor_linear = nn.Linear(128, num_outputs)     # output = Q(S, A)
        
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.flatten(x)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = F.elu(self.fc1(hx))
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
