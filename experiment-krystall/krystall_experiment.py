import pygame
import numpy as np
import random

# --- SIMULATION SETTINGS ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
NUM_AGENTS = 300      # Number of agents (N)
AGENT_RADIUS = 3
AGENT_SPEED = 2.0     # Average speed of agents
CHAOS_FACTOR = 0.5    # How strongly the direction changes (0 = straight lines)

# --- COLORS ---
BG_COLOR = (5, 5, 20)         # Dark blue background
AGENT_COLOR = (200, 220, 255) # Light blue agents
TEXT_COLOR = (255, 255, 255)  # White text

# --- Agent Class ---
class Agent:
    def __init__(self):
        # Initial position is a random point on the screen
        self.pos = np.array([
            random.uniform(0, SCREEN_WIDTH),
            random.uniform(0, SCREEN_HEIGHT)
        ], dtype=float)

        # Initial velocity is a random direction
        angle = random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float) * AGENT_SPEED

    def update(self):
        # Add some chaos to the movement
        chaos_vec = np.random.rand(2) * 2 - 1  # Random vector from -1 to 1
        self.vel += chaos_vec * CHAOS_FACTOR

        # Normalize velocity to keep it constant
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = self.vel / norm * AGENT_SPEED
        else:
            # If the velocity vector accidentally becomes zero, give it a random push
            angle = random.uniform(0, 2 * np.pi)
            self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float) * AGENT_SPEED

        # Move the agent
        self.pos += self.vel

        # Bounce off the walls with a "push-out" mechanism
        if self.pos[0] <= AGENT_RADIUS:
            self.pos[0] = AGENT_RADIUS
            self.vel[0] *= -1
        elif self.pos[0] >= SCREEN_WIDTH - AGENT_RADIUS:
            self.pos[0] = SCREEN_WIDTH - AGENT_RADIUS
            self.vel[0] *= -1

        if self.pos[1] <= AGENT_RADIUS:
            self.pos[1] = AGENT_RADIUS
            self.vel[1] *= -1
        elif self.pos[1] >= SCREEN_HEIGHT - AGENT_RADIUS:
            self.pos[1] = SCREEN_HEIGHT - AGENT_RADIUS
            self.vel[1] *= -1
            
    def draw(self, screen):
        pygame.draw.circle(screen, AGENT_COLOR, self.pos.astype(int), AGENT_RADIUS)

# --- Main Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Digital Primordial Broth v1.1")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 48)

    # Create N agents
    agents = [Agent() for _ in range(NUM_AGENTS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Update agent states ---
        for agent in agents:
            agent.update()

        # --- Calculate the Global Coherence Factor (GCF) ---
        velocities = np.array([agent.vel for agent in agents])

        # Protect against division by zero
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_velocities = velocities / norms

        mean_vector = np.mean(normalized_velocities, axis=0)
        gcf = np.linalg.norm(mean_vector) # Renamed from KGP to GCF for English

        # --- Drawing ---
        screen.fill(BG_COLOR)
        for agent in agents:
            agent.draw(screen)

        # Display the GCF on the screen
        gcf_text = font.render(f"GCF: {gcf:.4f}", True, TEXT_COLOR) # Changed КГП to GCF
        screen.blit(gcf_text, (20, 20))

        pygame.display.flip()
        clock.tick(60) # Limit to 60 frames per second

    pygame.quit()

if __name__ == "__main__":
    main()