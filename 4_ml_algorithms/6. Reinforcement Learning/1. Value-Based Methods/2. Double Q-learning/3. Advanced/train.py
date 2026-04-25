import pygame
from env import GridEnv
from agent import DoubleQAgent
from settings import EPISODES, MAX_STEPS, FPS


def train():
    env = GridEnv()
    agent = DoubleQAgent()

    clock = pygame.time.Clock()

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:

            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            # Choix d'action par l'agent
            action = agent.choose_action(state)

            # Transition environnement
            next_state, reward, done = env.step(action)

            # Mise à jour Q-table
            agent.update(state, action, reward, next_state)

            # Aller à l'état suivant
            state = next_state

            # Affichage pygame
            env.render(agent.Q1 + agent.Q2)
            clock.tick(FPS)

            steps += 1

        # Réduction epsilon (moins d'exploration)
        agent.decay_epsilon()
        print(f"Épisode {ep+1}/{EPISODES} terminé. epsilon={agent.epsilon:.3f}")
        
        
    # POST-TRAINING : l'écran reste affiché 
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        env.render(agent.Q1 + agent.Q2)
        clock.tick(FPS)

    pygame.quit()

    pygame.quit()


if __name__ == "__main__":
    train()
