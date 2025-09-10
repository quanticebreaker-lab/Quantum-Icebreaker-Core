import pygame
import numpy as np
import random

# --- НАСТРОЙКИ СИМУЛЯЦИИ ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
NUM_AGENTS = 300  # Количество агентов (N)
AGENT_RADIUS = 3
AGENT_SPEED = 2.0  # Средняя скорость агентов
CHAOS_FACTOR = 0.5 # Насколько сильно меняется направление (0 = прямые линии)

# --- ЦВЕТА ---
BG_COLOR = (5, 5, 20)        # Темно-синий фон
AGENT_COLOR = (200, 220, 255) # Светло-голубые агенты
TEXT_COLOR = (255, 255, 255) # Белый текст

# --- Класс для каждого агента ---
class Agent:
    def __init__(self):
        # Начальная позиция - случайная точка на экране
        self.pos = np.array([
            random.uniform(0, SCREEN_WIDTH),
            random.uniform(0, SCREEN_HEIGHT)
        ], dtype=float)

        # Начальная скорость - случайное направление
        angle = random.uniform(0, 2 * np.pi)
        self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float) * AGENT_SPEED

    def update(self):
        # Добавляем немного хаоса в движение
        chaos_vec = np.random.rand(2) * 2 - 1  # Случайный вектор от -1 до 1
        self.vel += chaos_vec * CHAOS_FACTOR

        # Нормализуем скорость, чтобы она оставалась постоянной
        norm = np.linalg.norm(self.vel)
        if norm > 0:
            self.vel = self.vel / norm * AGENT_SPEED
        else:
            # Если вектор скорости случайно стал нулевым, даём случайный толчок
            angle = random.uniform(0, 2 * np.pi)
            self.vel = np.array([np.cos(angle), np.sin(angle)], dtype=float) * AGENT_SPEED

        # Двигаем агента
        self.pos += self.vel

        # Отскок от стен с "выталкиванием"
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
        pygame.draw.circle(screen, AGENT_COLOR, (int(self.pos[0]), int(self.pos[1])), AGENT_RADIUS)

# --- Основная функция ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Цифровой Первичный Бульон v1.1")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 48)

    # Создаем N агентов
    agents = [Agent() for _ in range(NUM_AGENTS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Обновление состояния агентов ---
        for agent in agents:
            agent.update()

        # --- Расчет Коэффициента Глобального Порядка (КГП) ---
        velocities = np.array([agent.vel for agent in agents])

        # Защита от деления на ноль
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_velocities = velocities / norms

        mean_vector = np.mean(normalized_velocities, axis=0)
        kgp = np.linalg.norm(mean_vector)

        # --- Отрисовка ---
        screen.fill(BG_COLOR)
        for agent in agents:
            agent.draw(screen)

        # Отображаем КГП на экране
        kgp_text = font.render(f"КГП: {kgp:.4f}", True, TEXT_COLOR)
        screen.blit(kgp_text, (20, 20))

        pygame.display.flip()
        clock.tick(60) # Ограничение до 60 кадров в секунду

    pygame.quit()

if __name__ == "__main__":
    main()