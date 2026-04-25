# Entraînement de l'IA

import torch
import torch.nn.functional as F
from A3C.envs import create_atari_env
from A3C.model import ActorCritic

# Fonction pour s'assurer que les modèles partagent les mêmes gradients
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is None:
            shared_param._grad = param.grad

        

def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)  # décaler la seed avec le rang pour asynchroniser chaque agent
    
    env = create_atari_env(params.env_name)  # créer un environnement optimisé grâce à create_atari_env
 
    model = ActorCritic(env.observation_space.shape[0], env.action_space)  # créer le modèle à partir de la classe ActorCritic
    state, _ = env.reset(seed = params.seed + rank )  # état initial : tableau numpy de taille 1*42*42 en noir et blanc
    state = torch.from_numpy(state).float()  # convertir le tableau numpy en tenseur torch
    
    done = True  # flag indiquant si l'épisode est terminé
    episode_length = 0  # initialiser la longueur de l'épisode à 0
    total_frames = 0  # compteur total de frames pour ce processus

    while True:  # boucle principale
        episode_length += 1  # incrémenter la longueur de l'épisode
        model.load_state_dict(shared_model.state_dict())  # synchroniser avec le modèle partagé

        if done:  # si c'est la première itération ou si le jeu vient de se terminer
            cx = torch.zeros(1, 256)  # réinitialiser l'état de cellule du LSTM à zéro
            hx = torch.zeros(1, 256)  # réinitialiser l'état caché du LSTM à zéro
        else:  # sinon
            cx = cx.detach()  # conserver l'état de cellule précédent
            hx = hx.detach()  # conserver l'état caché précédent

        values = []  # liste des valeurs V(S)
        log_probs = []  # liste des log-probabilités des actions
        rewards = []  # liste des récompenses
        entropies = []  # liste des entropies

        for step in range(params.num_steps):  # parcourir les pas d'exploration
            # obtenir V(S), Q(S,A) et les nouveaux états caché et cellule
            value, action_values, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            
            prob = F.softmax(action_values, dim=1)  # distribution de probabilités des actions
            log_prob = F.log_softmax(action_values, dim=1)  # log-probabilités des actions
            entropy = -(log_prob * prob).sum(1)  # entropie de la politique
            entropies.append(entropy)  # stocker l'entropie de la politique
            
            total_frames += 1


            action = torch.multinomial(prob, num_samples=1)  # tirer une action selon la distribution
            log_prob = log_prob.gather(1, action)  # log-probabilité de l'action choisie

            values.append(value)  # stocker la valeur V(S)
            log_probs.append(log_prob)  # stocker le log-prob de l'action

            # jouer l'action, obtenir le nouvel état et la récompense
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated or (episode_length >= params.max_episode_length)  # si l'épisode est trop long, le terminer
            state = torch.from_numpy(next_state).float()  # convertir le nouvel état en tenseur
            reward = max(min(reward, 1), -1)  # clamp de la récompense entre -1 et 1
            
            if total_frames % 10000 == 0:
                print(f"[TRAIN] Process {rank} | Frames {total_frames} | Episode {episode_length} | Reward {sum(rewards)}")


            if done:  # si l'épisode est terminé
                episode_length = 0  # réinitialiser la longueur de l'épisode
                state, _ = env.reset(seed = params.seed + rank )  # réinitialiser l'environnement
                state = torch.from_numpy(state).float()  # convertir le nouvel état en tenseur
            
            
            rewards.append(reward)  # stocker la récompense

            if done:
                break  # arrêter l'exploration si l'épisode est terminé

        R = torch.zeros(1, 1)  # initialiser la récompense cumulative

        if not done:  # si l'épisode n'est pas terminé
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))  # estimer V(S) pour le dernier état
            R = value.detach()  # initialiser la récompense cumulative

        values.append(R)  # stocker V(S) du dernier état

        policy_loss = 0  # initialiser la loss de la politique
        value_loss = 0  # initialiser la loss de la valeur
        gae = torch.zeros(1, 1)  # initialiser l'Advantage Général (GAE)

        # boucle inversée pour calculer les pertes
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]  # R = gamma*R + r_t
            advantage = R - values[i]  # advantage = R(reward obtenu) - V(S)(état actuel)
            value_loss = value_loss + 0.5 * advantage.pow(2)  # perte du Critic

            # calculer la perte de l'Actor (r_t + gamma*V(S_t+1) - V(S_t)) 
            TD = rewards[i] + params.gamma * values[i + 1].detach() - values[i].detach()  # différence temporelle
            # calculer l'Advantage Général (GAE) : GAE = TD + gamma*tau*GAE
            gae = gae * params.gamma * params.tau + TD 
            # calculer la perte de l'Actor : -log(pi(a_t|s_t))*GAE - 0.01*H(pi)
            policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]  # perte de l'Actor

        optimizer.zero_grad()  # initialiser le gradient
        (policy_loss + 0.5 * value_loss).backward()  # backpropagation

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)  # clamp des gradients pour stabilité
        ensure_shared_grads(model, shared_model)  # partager les gradients avec le modèle global
        optimizer.step()  # mise à jour des poids
