import pygame
import numpy as np
from settings import GRID_SIZE, CELL_SIZE, REWARD_GOAL, REWARD_STEP, REWARD_OBSTACLE, ACTIONS


class GridEnv:
    def __init__(self):
        pygame.init()

        # Fenêtre pygame dimensionnée selon la grille
        self.window = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        pygame.display.set_caption("Q-Learning Lab")

        # Position de départ de l'agent
        self.agent_pos = None

        # Position de la récompense
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)

        # Obstacles fixes (tu peux en ajouter)
        self.obstacles = {(3, 3), (4, 2), (5, 5)}

    def reset(self):
        # L'agent revient à la position de départ
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        # Action → vecteur directionnel
        ax, ay = self.agent_pos
        dx, dy = ACTIONS[action]
        nx, ny = ax + dx, ay + dy

        # Collision : hors de la grille
        if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
            return self.agent_pos, REWARD_STEP, False

        new_pos = (nx, ny)

        # Collision obstacle
        if new_pos in self.obstacles:
            return self.agent_pos, REWARD_OBSTACLE, False

        # Objectif atteint
        if new_pos == self.goal:
            return new_pos, REWARD_GOAL, True

        # Déplacement normal
        self.agent_pos = new_pos
        return new_pos, REWARD_STEP, False

    def render(self, q_table=None):
        self.window.fill((30, 30, 30))
    
        # Obstacles
        for (x, y) in self.obstacles:
            pygame.draw.rect(self.window, (200, 50, 50),
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
        # But
        gx, gy = self.goal
        pygame.draw.rect(self.window, (50, 200, 50),
                         (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
        # Agent
        ax, ay = self.agent_pos
        pygame.draw.rect(self.window, (50, 150, 255),
                         (ax * CELL_SIZE, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
        # Afficher Q-table numérique
        if q_table is not None:
            self.draw_q_numbers(q_table)
    
        pygame.display.flip()
        
    def draw_q_numbers(self, q_table):
        font = pygame.font.SysFont("Arial", 14)

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                cx = x * CELL_SIZE
                cy = y * CELL_SIZE

                q_up, q_down, q_left, q_right = q_table[x, y]

                # Texte formats (limiter décimales)
                up_txt    = font.render(f"{q_up:.2f}", True, (200, 200, 255))
                down_txt  = font.render(f"{q_down:.2f}", True, (200, 200, 255))
                left_txt  = font.render(f"{q_left:.2f}", True, (200, 200, 255))
                right_txt = font.render(f"{q_right:.2f}", True, (200, 200, 255))

                # Position : chaque valeur dans son coin
                self.window.blit(up_txt,    (cx + CELL_SIZE/3, cy + 2))
                self.window.blit(down_txt,  (cx + CELL_SIZE/3, cy + CELL_SIZE - 18))
                self.window.blit(left_txt,  (cx + 2, cy + CELL_SIZE/2 - 7))
                self.window.blit(right_txt, (cx + CELL_SIZE - 40, cy + CELL_SIZE/2 - 7))
