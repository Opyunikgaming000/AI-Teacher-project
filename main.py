import random
import sys

import pygame


WIDTH, HEIGHT = 800, 600
PLAYER_SPEED = 6
BULLET_SPEED = 9
ENEMY_MIN_SPEED = 2
ENEMY_MAX_SPEED = 5
ENEMY_SPAWN_MS = 700


def reset_game_state() -> dict:
    return {
        "player": pygame.Rect(WIDTH // 2 - 20, HEIGHT - 60, 40, 30),
        "bullets": [],
        "enemies": [],
        "score": 0,
        "game_over": False,
    }


def move_player(player: pygame.Rect) -> None:
    keys = pygame.key.get_pressed()
    dx = 0
    dy = 0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx -= PLAYER_SPEED
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx += PLAYER_SPEED
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        dy -= PLAYER_SPEED
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        dy += PLAYER_SPEED

    player.x += dx
    player.y += dy
    player.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))


def draw_ship(screen: pygame.Surface, player: pygame.Rect) -> None:
    pygame.draw.polygon(
        screen,
        (80, 200, 255),
        [
            (player.centerx, player.top),
            (player.left, player.bottom),
            (player.right, player.bottom),
        ],
    )


def draw_text(screen: pygame.Surface, text: str, size: int, color: tuple[int, int, int], pos: tuple[int, int]) -> None:
    font = pygame.font.SysFont("arial", size)
    screen.blit(font.render(text, True, color), pos)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simple 2D Shooter")
    clock = pygame.time.Clock()

    state = reset_game_state()
    fire_cooldown = 0

    spawn_event = pygame.USEREVENT + 1
    pygame.time.set_timer(spawn_event, ENEMY_SPAWN_MS)

    while True:
        dt = clock.tick(60)
        fire_cooldown = max(0, fire_cooldown - dt)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == spawn_event and not state["game_over"]:
                enemy_w = random.randint(30, 45)
                enemy_h = random.randint(20, 35)
                enemy_x = random.randint(0, WIDTH - enemy_w)
                enemy = pygame.Rect(enemy_x, -enemy_h, enemy_w, enemy_h)
                speed = random.randint(ENEMY_MIN_SPEED, ENEMY_MAX_SPEED)
                state["enemies"].append((enemy, speed))

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not state["game_over"] and fire_cooldown == 0:
                    bullet = pygame.Rect(state["player"].centerx - 3, state["player"].top - 10, 6, 12)
                    state["bullets"].append(bullet)
                    fire_cooldown = 180
                if event.key == pygame.K_r and state["game_over"]:
                    state = reset_game_state()
                    fire_cooldown = 0

        if not state["game_over"]:
            move_player(state["player"])

            for bullet in state["bullets"]:
                bullet.y -= BULLET_SPEED
            state["bullets"] = [b for b in state["bullets"] if b.bottom > 0]

            moved_enemies = []
            for enemy, speed in state["enemies"]:
                enemy.y += speed
                if enemy.top < HEIGHT:
                    moved_enemies.append((enemy, speed))
            state["enemies"] = moved_enemies

            hit_enemy_indexes = set()
            remaining_bullets = []
            for bullet in state["bullets"]:
                hit = False
                for index, (enemy, _) in enumerate(state["enemies"]):
                    if index in hit_enemy_indexes:
                        continue
                    if bullet.colliderect(enemy):
                        state["score"] += 1
                        hit_enemy_indexes.add(index)
                        hit = True
                        break
                if not hit:
                    remaining_bullets.append(bullet)

            state["bullets"] = remaining_bullets
            if hit_enemy_indexes:
                state["enemies"] = [
                    enemy_data
                    for index, enemy_data in enumerate(state["enemies"])
                    if index not in hit_enemy_indexes
                ]

            for enemy, _ in state["enemies"]:
                if enemy.colliderect(state["player"]):
                    state["game_over"] = True
                    break

        screen.fill((14, 18, 30))

        draw_ship(screen, state["player"])

        for bullet in state["bullets"]:
            pygame.draw.rect(screen, (255, 230, 100), bullet)

        for enemy, _ in state["enemies"]:
            pygame.draw.rect(screen, (255, 90, 90), enemy)

        draw_text(screen, f"Score: {state['score']}", 28, (255, 255, 255), (12, 10))
        draw_text(screen, "Move: Arrow Keys/WASD | Shoot: Space", 22, (180, 220, 255), (12, HEIGHT - 34))

        if state["game_over"]:
            draw_text(screen, "GAME OVER", 56, (255, 120, 120), (WIDTH // 2 - 165, HEIGHT // 2 - 80))
            draw_text(screen, "Press R to restart", 32, (240, 240, 240), (WIDTH // 2 - 130, HEIGHT // 2 - 20))

        pygame.display.flip()


if __name__ == "__main__":
    main()
