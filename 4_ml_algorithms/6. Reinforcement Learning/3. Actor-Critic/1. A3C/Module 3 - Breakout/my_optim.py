# Optimiseur

import math
import torch
import torch.optim as optim

# Implémentation de l’optimiseur Adam avec des états partagés
# (nécessaire pour l’entraînement multi-processus de A3C)

class SharedAdam(optim.Adam):  
    # Classe héritant de optim.Adam

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # Initialisation de l’optimiseur Adam classique
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        # Initialisation manuelle des états internes de l’optimiseur
        # pour chaque paramètre du modèle
        for group in self.param_groups:
            # param_groups contient les groupes de paramètres à optimiser
            for p in group['params']:
                # Pour chaque paramètre (poids du réseau)
                state = self.state[p]
                # À ce stade, self.state est vide → on initialise les états

                state['step'] = torch.zeros((), dtype=torch.int64)
                # Compteur du nombre de mises à jour

                state['exp_avg'] = torch.zeros_like(p)
                # Moyenne mobile exponentielle du gradient (moment d’ordre 1)

                state['exp_avg_sq'] = torch.zeros_like(p)
                # Moyenne mobile exponentielle du gradient au carré (moment d’ordre 2)

    # Partage de la mémoire entre processus
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Les tenseurs internes de l’optimiseur sont placés
                # dans la mémoire partagée pour être accessibles
                # par tous les processus
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    # Exécution d’une étape d’optimisation Adam
    # (voir l’algorithme 1 : https://arxiv.org/pdf/1412.6980.pdf)
    def step(self):

        for group in self.param_groups:
            for p in group['params']:
                # Si aucun gradient n’est disponible, on ignore ce paramètre
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Incrément du compteur de pas
                state['step'] += 1

                # Régularisation L2 (weight decay)
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Mise à jour des moyennes mobiles
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Calcul du dénominateur
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Corrections de biais
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()

                # Calcul du pas de mise à jour
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Mise à jour finale des paramètres
                with torch.no_grad():
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return None
