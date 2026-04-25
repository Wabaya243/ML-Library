import pygame
from env import GridEnv
from agent import QAgent
from settings import EPISODES, MAX_STEPS, FPS


def train():
    env = GridEnv()
    agent = QAgent()

    clock = pygame.time.Clock()

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:

            # Gestion fermeture fenêtre
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            # Choix d'action par l'agent
            action = agent.choose_action(state)

            # Transition environnement
            next_state, reward, done = env.step(action)

            # Mise à jour Q-table
            agent.update(state, action, reward, next_state)

            # Aller à l'état suivant
            state = next_state

            # Affichage pygame
            env.render(agent.q_table)
            clock.tick(FPS)

            steps += 1

        # Réduction epsilon (moins d'exploration)
        agent.decay_epsilon()
        print(f"Épisode {ep+1}/{EPISODES} terminé. epsilon={agent.epsilon:.3f}")

    pygame.quit()


if __name__ == "__main__":
    train()
