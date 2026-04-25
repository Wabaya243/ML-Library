# Test Agent

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
import time
from collections import deque

import os
SAVE_DIR = "Save"
os.makedirs(SAVE_DIR, exist_ok=True)

def test(rank, params, shared_model):
    # pour la sauvegarde du model    
    
    episode = 0
    
    best_reward = -float("inf")

    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name, video=True)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    
    model.eval()
    state, _ = env.reset(seed=params.seed + rank)
    state = torch.from_numpy(state).float()

    reward_sum = 0
    done = True
    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0

    while True:
        episode_length += 1

        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, action_value, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(action_value, dim=1) # Application de softmax sur les actions
            action = prob.argmax(dim=1).item() # Sélection de l'action avec la probabilité maximale
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = torch.from_numpy(next_state).float()
        reward_sum += reward

        if done:
            
            episode += 1
            
            print(
                f"[TEST] Episode {episode} | "
                f"Reward {reward_sum} | "
                f"Length {episode_length}"
            )
            
            # Sauvegarde du meilleur modèle
            if reward_sum > best_reward:
                best_reward = reward_sum
                torch.save(
                    {
                        "episode": episode,
                        "reward": reward_sum,
                        "model_state_dict": shared_model.state_dict(),
                    },
                    os.path.join(SAVE_DIR, "a3c_breakout_best.pt")
                )
                print(f" Nouveau meilleur modèle sauvegardé (episode {episode})")
            
            print("Time {}, episode reward {}, episode length {}".format(time.strftime(
                "%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state,_ = env.reset()
            state = torch.from_numpy(state).float()   # <- Correction
            time.sleep(1)
