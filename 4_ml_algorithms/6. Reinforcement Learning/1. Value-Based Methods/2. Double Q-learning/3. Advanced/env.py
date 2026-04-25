import pygame
import numpy as np
from settings import GRID_SIZE, CELL_SIZE, REWARD_GOAL, REWARD_STEP, REWARD_OBSTACLE, ACTIONS

class GridEnv:
    def __init__(self):
        pygame.init()

        self.window = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        pygame.display.set_caption("Q-Learning Lab")

        # agent_pos = (row, col)
        self.agent_pos = None

        # But
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)  # (row, col)

        # Obstacles
        self.obstacles = {(3, 3), (4, 2), (5, 5)}

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        # 1. Transformation déterministe → NON DÉTERMINISTE
        slip_prob = 0.2   # par exemple, 20% de bruit

        import random
        if random.random() < slip_prob:
            # L’action change aléatoirement
            action = np.random.randint(4)
            
        row, col = self.agent_pos
        drow, dcol = ACTIONS[action]

        nrow = row + drow
        ncol = col + dcol

        # Hors limites
        if nrow < 0 or nrow >= GRID_SIZE or ncol < 0 or ncol >= GRID_SIZE:
            return self.agent_pos, REWARD_STEP, False

        new_pos = (nrow, ncol)

        # Obstacle
        if new_pos in self.obstacles:
            return self.agent_pos, REWARD_OBSTACLE, False

        # Goal
        if new_pos == self.goal:
            self.agent_pos = new_pos
            return new_pos, REWARD_GOAL, True

        # Déplacement normal
        self.agent_pos = new_pos
        return new_pos, REWARD_STEP, False

    def render(self, q_table=None):
        self.window.fill((30, 30, 30))

        # Obstacles
        for (r, c) in self.obstacles:
            pygame.draw.rect(self.window, (200, 50, 50),
                             (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # But
        gr, gc = self.goal
        pygame.draw.rect(self.window, (50, 200, 50),
                         (gc * CELL_SIZE, gr * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Agent
        ar, ac = self.agent_pos
        pygame.draw.rect(self.window, (50, 150, 255),
                         (ac * CELL_SIZE, ar * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        if q_table is not None:
            self.draw_q_numbers(q_table)

        pygame.display.flip()

    def draw_q_numbers(self, q_table):
        font = pygame.font.SysFont("Arial", 14)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cx = c * CELL_SIZE
                cy = r * CELL_SIZE

                q_up, q_down, q_left, q_right = q_table[r, c]

                self.window.blit(font.render(f"{q_up:.2f}", True, (200,200,255)),
                                 (cx + CELL_SIZE/3, cy + 2))
                self.window.blit(font.render(f"{q_down:.2f}", True, (200,200,255)),
                                 (cx + CELL_SIZE/3, cy + CELL_SIZE - 18))
                self.window.blit(font.render(f"{q_left:.2f}", True, (200,200,255)),
                                 (cx + 2, cy + CELL_SIZE/2 - 7))
                self.window.blit(font.render(f"{q_right:.2f}", True, (200,200,255)),
                                 (cx + CELL_SIZE - 40, cy + CELL_SIZE/2 - 7))
