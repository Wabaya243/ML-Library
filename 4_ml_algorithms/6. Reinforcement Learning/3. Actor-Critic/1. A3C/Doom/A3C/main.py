# Code principal

from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
import ale_py.roms  # parfois suffisant
from A3C.envs import create_atari_env
from A3C.model import ActorCritic
from A3C.train_agent import train
from A3C.test_agent import test
from A3C import my_optim
from A3C.params import Params

def main():
    # Lancement principal
    os.environ['OMP_NUM_THREADS'] = '1'  
    # Limite à 1 le nombre de threads OpenMP par cœur pour éviter les conflits
    
    params = Params()  
    # Création de l’objet paramètres à partir de la classe Params
    
    torch.manual_seed(params.seed)  
    # Fixe la graine aléatoire (utile pour la reproductibilité, mais non essentiel)
    
    env = create_atari_env(params.env_name)  
    # Création de l’environnement Atari optimisé
    
    shared_model = ActorCritic(
        env.observation_space.shape[0],
        env.action_space
    )  
    
    
    r'''
    checkpoint = torch.load("Save/a3c_breakout_best.pt")
    shared_model.load_state_dict(checkpoint["model_state_dict"])
    start_episode = checkpoint["episode"]
    
    print(f" Reprise depuis l'épisode {start_episode}")
    
    '''
    
    # Modèle Actor-Critic partagé entre tous les agents (processus)
    
    shared_model.share_memory()  
    # Stocke le modèle dans la mémoire partagée afin que tous les processus
    # puissent y accéder et le modifier
    
    optimizer = my_optim.SharedAdam(
        shared_model.parameters(),
        lr=params.lr
    )
    # Optimiseur partagé, agissant sur le modèle partagé
    
    optimizer.share_memory()
    # Stocke l’optimiseur dans la mémoire partagée pour un accès multi-processus
    
    processes = []
    # Liste qui contiendra tous les processus
    
    p = mp.Process(
        target=test,
        args=(params.num_processes, params, shared_model)
    )  
    # Création du processus de test :
    # - Il utilise le modèle partagé
    # - Il ne met PAS à jour les poids
    # - Il évalue simplement les performances
    
    p.start()  
    # Démarrage du processus de test
    
    processes.append(p)  
    # Ajout du processus de test à la liste
    
    for rank in range(0, params.num_processes):  
        # Boucle pour créer les processus d'entraînement
        # Chaque processus correspond à un agent différent
        p = mp.Process(
            target=train,
            args=(rank, params, shared_model, optimizer)
        )
        p.start()  
        # Démarrage du processus d'entraînement
        processes.append(p)  
        # Ajout du processus à la liste
    
    for p in processes:  
        # Attente de la fin de tous les processus
        # Permet un arrêt propre du programme
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')  # important sous Windows
    main()