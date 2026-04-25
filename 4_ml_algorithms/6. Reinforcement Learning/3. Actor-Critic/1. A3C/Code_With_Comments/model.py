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
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # first convolution
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # second convolution
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # third convolution
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # fourth convolution
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # making an LSTM (Long Short Term Memory) to learn the temporal properties of the input - we obtain a big encoded vector S of size 256 that encodes an event of the game
        num_outputs = action_space.n # getting the number of possible actions
        self.critic_linear = nn.Linear(256, 1) # full connection of the critic: output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) # full connection of the actor: output = Q(S,A)
        self.apply(weights_init) # initilizing the weights of the model with random weights
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # setting the standard deviation of the critic tensor of weights to 0.01
        self.critic_linear.bias.data.fill_(0) # initializing the critic bias with zeros
        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms

    def forward(self, inputs):
        inputs, (hx, cx) = inputs # getting separately the input images to the tuple (hidden states, cell states)
        x = F.elu(self.conv1(inputs)) # forward propagating the signal from the input images to the 1st convolutional layer
        x = F.elu(self.conv2(x)) # forward propagating the signal from the 1st convolutional layer to the 2nd convolutional layer
        x = F.elu(self.conv3(x)) # forward propagating the signal from the 2nd convolutional layer to the 3rd convolutional layer
        x = F.elu(self.conv4(x)) # forward propagating the signal from the 3rd convolutional layer to the 4th convolutional layer
        x = x.view(-1, 32 * 3 * 3) # flattening the last convolutional layer into this 1D vector x
        hx, cx = self.lstm(x, (hx, cx)) # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
        return self.critic_linear(x), self.actor_linear(x), (hx, cx) # returning the output of the critic (V(S)), the output of the actor (Q(S,A)), and the new hidden & cell states ((hx, cx))
