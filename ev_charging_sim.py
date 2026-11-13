"""Mô phỏng xe điện tìm đường đến trạm sạc bằng Pygame.

Điều khiển:
    Mũi tên: di chuyển xe
    Space: thực hiện lệnh Charge tại ô hiện tại
    R: reset mô phỏng
    Esc hoặc đóng cửa sổ: thoát
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pygame


# ---------------------------------------------------------------------------
# Cấu hình mô phỏng
# ---------------------------------------------------------------------------
GRID_SIZE = 15
WINDOW_WIDTH = 950
WINDOW_HEIGHT = 680
MAP_LEFT = 42
MAP_TOP = 48
CELL_SIZE = 39
MAP_SIZE = GRID_SIZE * CELL_SIZE
SIDEBAR_LEFT = MAP_LEFT + MAP_SIZE + 32
FPS = 30

ROAD = 0
OBSTACLE = 1
NORMAL = 2
FAST = 3
BUSY = 4

START_POSITION = (13, 1)

COLORS = {
    "background": (241, 245, 249),
    "road": (248, 250, 252),
    "grid": (203, 213, 225),
    "wall": (20, 83, 45),
    "obstacle": (30, 41, 59),
    "normal": (134, 239, 172),
    "fast": (125, 211, 252),
    "busy": (248, 113, 113),
    "car": (37, 99, 235),
    "white": (255, 255, 255),
    "text": (15, 23, 42),
    "muted": (71, 85, 105),
    "trail": (124, 58, 237),
    "panel": (255, 255, 255),
}


@dataclass
class Simulation:
    """Trạng thái và luật mô phỏng của xe điện."""

    grid: list[list[int]] = field(default_factory=list)
    position: tuple[int, int] = START_POSITION
    trail: list[tuple[int, int]] = field(default_factory=list)
    step: int = 0
    reward: int = 0
    collisions: int = 0
    status: str = "Đang tìm đường"
    finished: bool = False
    busy_positions: set[tuple[int, int]] = field(default_factory=set)
    station_positions: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.reset()

    def station_type_layout(self) -> dict[tuple[int, int], int]:
        """Trả về loại cố định của từng vị trí trạm, chưa áp dụng trạng thái bận."""
        station_types = {
            (3, 5): NORMAL,
            (4, 5): FAST,
            (5, 5): NORMAL,
            (6, 5): FAST,
            (3, 7): FAST,
            (4, 7): NORMAL,
            (5, 7): NORMAL,
            (6, 7): FAST,
            (3, 10): NORMAL,
            (4, 10): FAST,
            (5, 10): NORMAL,
            (6, 10): FAST,
            (3, 12): FAST,
            (4, 12): NORMAL,
            (5, 12): FAST,
            (6, 12): NORMAL,
        }

        # Hai vùng trạm mới 3 hàng x 3 cột tại các vị trí được đánh dấu V.
        extra_station_pattern = (
            (NORMAL, FAST, NORMAL),
            (FAST, NORMAL, FAST),
            (NORMAL, FAST, NORMAL),
        )
        for y_offset, row in enumerate(extra_station_pattern):
            for x_offset, station_code in enumerate(row):
                if y_offset != 1:
                    station_types[(7 + x_offset, 5 + y_offset)] = station_code
                    station_types[(7 + x_offset, 10 + y_offset)] = station_code
        return station_types

    def create_map(self) -> list[list[int]]:
        """Tạo bản đồ 15x15 và áp dụng trạng thái bận hiện tại."""
        grid = [[ROAD for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Tường bao; mở hai ô cửa vào tại (12, 0) và (13, 0).
        for x in range(GRID_SIZE):
            grid[0][x] = OBSTACLE
            grid[GRID_SIZE - 1][x] = OBSTACLE
        for y in range(GRID_SIZE):
            grid[y][0] = OBSTACLE
            grid[y][GRID_SIZE - 1] = OBSTACLE
        grid[0][12] = ROAD
        grid[0][13] = ROAD

        # Khối điều hành lớn phía trên trái, đặt sát tường trên. Hàng 4
        # để trống làm làn đường.
        for y in range(1, 4):
            for x in range(1, 9):
                grid[y][x] = OBSTACLE

        # Hai dải phân cách ngang giữa các làn trạm.
        for x in range(3, 7):
            grid[6][x] = OBSTACLE
            grid[11][x] = OBSTACLE
        for x in range(7, 10):
            grid[6][x] = OBSTACLE
            grid[11][x] = OBSTACLE

        station_types = self.station_type_layout()
        self.station_positions = tuple(station_types)
        for position, station_code in station_types.items():
            if position in self.busy_positions:
                station_code = BUSY
            x, y = position
            grid[y][x] = station_code

        return grid

    def reset(self) -> None:
        """Đưa xe về lối vào và tạo trạng thái trạm ngẫu nhiên mới."""
        station_positions = tuple(self.station_type_layout())
        busy_count = random.randint(1, max(1, len(station_positions) - 2))
        self.busy_positions = set(random.sample(station_positions, busy_count))
        self.grid = self.create_map()
        self.position = START_POSITION
        self.trail = [START_POSITION]
        self.step = 0
        self.reward = 0
        self.collisions = 0
        self.status = "Đang tìm đường"
        self.finished = False

    def reload_stations(self) -> None:
        """Reload bản đồ trạm, luôn giữ lại ít nhất hai trụ đang trống."""
        self.reset()
        self.status = "Đã reload trạng thái trụ sạc"

    def move(self, dx: int, dy: int) -> None:
        """Di chuyển xe; va chạm giữ nguyên vị trí và phạt -100."""
        if self.finished:
            return

        self.step += 1
        x, y = self.position
        target = (x + dx, y + dy)
        valid = 0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE

        if not valid or self.grid[target[1]][target[0]] in (OBSTACLE, BUSY):
            self.reward -= 100
            self.collisions += 1
            self.status = "Va chạm - đường bị chặn"
            return

        self.position = target
        self.trail.append(target)
        self.reward -= 5
        self.status = f"Đang di chuyển ({target[1]}, {target[0]})"

    def charge(self) -> None:
        """Sạc tại vị trí hiện tại và kết thúc lượt mô phỏng."""
        if self.finished:
            return

        self.step += 1
        code = self.grid[self.position[1]][self.position[0]]
        if code == NORMAL:
            self.reward += 100
            self.status = "Sạc thường thành công (+100)"
        elif code == FAST:
            self.reward += 120
            self.status = "Sạc nhanh thành công (+120)"
        elif code == BUSY:
            self.reward -= 100
            self.status = "Trụ đang bận (-100)"
        else:
            self.reward -= 50
            self.status = "Charge sai vị trí (-50)"
        self.finished = True


class PygameRenderer:
    """Bộ vẽ giao diện, tách biệt khỏi luật mô phỏng."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("EV Charging Station - Mô phỏng Pygame")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 16)
        self.small_font = pygame.font.SysFont("segoeui", 13)
        self.bold_font = pygame.font.SysFont("segoeui", 16, bold=True)
        self.title_font = pygame.font.SysFont("segoeui", 22, bold=True)

    def text(
        self,
        value: str,
        position: tuple[int, int],
        font: pygame.font.Font | None = None,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        surface = (font or self.font).render(
            str(value), True, color or COLORS["text"]
        )
        self.screen.blit(surface, position)

    def cell_rect(self, x: int, y: int, inset: int = 2) -> pygame.Rect:
        return pygame.Rect(
            MAP_LEFT + x * CELL_SIZE + inset,
            MAP_TOP + y * CELL_SIZE + inset,
            CELL_SIZE - inset * 2,
            CELL_SIZE - inset * 2,
        )

    def draw_grid(self, simulation: Simulation) -> None:
        """Vẽ ô bản đồ, tọa độ, trạm sạc và các khối vật cản."""
        for x in range(GRID_SIZE):
            self.text(
                str(x),
                (MAP_LEFT + x * CELL_SIZE + 14, MAP_TOP - 25),
                self.small_font,
                COLORS["muted"],
            )
        for y in range(GRID_SIZE):
            self.text(
                str(y),
                (MAP_LEFT - 25, MAP_TOP + y * CELL_SIZE + 11),
                self.small_font,
                COLORS["muted"],
            )

        for y, row in enumerate(simulation.grid):
            for x, code in enumerate(row):
                color = {
                    ROAD: COLORS["road"],
                    OBSTACLE: COLORS["wall"],
                    NORMAL: COLORS["normal"],
                    FAST: COLORS["fast"],
                    BUSY: COLORS["busy"],
                }[code]
                rect = self.cell_rect(x, y)
                pygame.draw.rect(
                    self.screen,
                    color,
                    rect,
                    border_radius=6 if code != ROAD else 0,
                )
                pygame.draw.rect(self.screen, COLORS["grid"], rect, width=1)

                if code == NORMAL:
                    self.text(
                        "N",
                        (rect.centerx - 5, rect.centery - 9),
                        self.bold_font,
                        COLORS["text"],
                    )
                elif code == FAST:
                    self.text(
                        "F",
                        (rect.centerx - 5, rect.centery - 9),
                        self.bold_font,
                        COLORS["text"],
                    )
                elif code == BUSY:
                    self.text(
                        "N/F",
                        (rect.centerx - 11, rect.centery - 7),
                        self.small_font,
                        COLORS["text"],
                    )

        # Vệt đường đi.
        if len(simulation.trail) > 1:
            points = [
                (
                    MAP_LEFT + x * CELL_SIZE + CELL_SIZE // 2,
                    MAP_TOP + y * CELL_SIZE + CELL_SIZE // 2,
                )
                for x, y in simulation.trail
            ]
            pygame.draw.lines(self.screen, COLORS["trail"], False, points, 4)

        # Nhãn lối vào và xe điện.
        entrance_label = "LỐI VÀO"
        entrance_center_x = MAP_LEFT + 13 * CELL_SIZE
        entrance_x = entrance_center_x - self.small_font.size(entrance_label)[0] // 2
        self.text(entrance_label, (entrance_x, MAP_TOP + 11), self.small_font, COLORS["car"])
        car_x, car_y = simulation.position
        center = (
            MAP_LEFT + car_x * CELL_SIZE + CELL_SIZE // 2,
            MAP_TOP + car_y * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.circle(self.screen, COLORS["white"], center, 14)
        pygame.draw.circle(self.screen, COLORS["car"], center, 10)

    def draw_legend_item(
        self, y: int, color: tuple[int, int, int], label: str, symbol: str = ""
    ) -> None:
        swatch = pygame.Rect(SIDEBAR_LEFT + 20, y, 22, 22)
        pygame.draw.rect(self.screen, color, swatch, border_radius=5)
        if symbol:
            self.text(symbol, (swatch.x + 3, swatch.y + 3), self.small_font, COLORS["text"])
        self.text(label, (SIDEBAR_LEFT + 52, y + 3), self.small_font, COLORS["muted"])

    def draw_sidebar(self, simulation: Simulation) -> None:
        panel = pygame.Rect(SIDEBAR_LEFT, 20, WINDOW_WIDTH - SIDEBAR_LEFT - 20, 640)
        pygame.draw.rect(self.screen, COLORS["panel"], panel, border_radius=14)
        self.text("EV CHARGING", (SIDEBAR_LEFT + 20, 38), self.title_font)
        self.text(
            "Xe điện tự động tìm trạm sạc tối ưu",
            (SIDEBAR_LEFT + 20, 70),
            self.small_font,
            COLORS["muted"],
        )

        self.text("CHÚ THÍCH", (SIDEBAR_LEFT + 20, 108), self.bold_font)
        y = 138
        self.draw_legend_item(y, COLORS["car"], "Xe điện")
        self.draw_legend_item(y + 29, COLORS["trail"], "Vệt đường đi")
        self.draw_legend_item(y + 58, COLORS["wall"], "Vật cản / tường bao")
        self.draw_legend_item(y + 87, COLORS["normal"], "Sạc thường trống  +100", "N")
        self.draw_legend_item(y + 116, COLORS["fast"], "Sạc nhanh trống  +120", "F")
        self.draw_legend_item(y + 145, COLORS["busy"], "Trụ đang bận  -100", "N/F")

        y += 205
        self.text("THÔNG SỐ MÔ PHỎNG", (SIDEBAR_LEFT + 20, y), self.bold_font)
        y += 34
        values = (
            ("Step", simulation.step),
            ("Reward", simulation.reward),
            ("Collisions", simulation.collisions),
        )
        for label, value in values:
            self.text(label, (SIDEBAR_LEFT + 22, y), self.small_font, COLORS["muted"])
            self.text(str(value), (SIDEBAR_LEFT + 165, y), self.bold_font)
            y += 32

        self.text("TRẠNG THÁI", (SIDEBAR_LEFT + 20, y + 14), self.bold_font)
        self.text(
            simulation.status,
            (SIDEBAR_LEFT + 22, y + 47),
            self.small_font,
            COLORS["muted"],
        )
        button = self.reload_button_rect()
        pygame.draw.rect(self.screen, COLORS["car"], button, border_radius=8)
        button_text = "Reload Trụ Sạc (R)"
        text_width = self.bold_font.size(button_text)[0]
        self.text(
            button_text,
            (button.centerx - text_width // 2, button.y + 9),
            self.bold_font,
            COLORS["white"],
        )
        self.text(
            "Mũi tên: di chuyển    Space: Charge    R: Reload",
            (SIDEBAR_LEFT + 20, 625),
            self.small_font,
            COLORS["muted"],
        )

    def reload_button_rect(self) -> pygame.Rect:
        return pygame.Rect(SIDEBAR_LEFT + 20, 570, 231, 36)

    def draw(self, simulation: Simulation) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill(COLORS["background"])
        self.draw_grid(simulation)
        self.draw_sidebar(simulation)
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        pygame.quit()


def handle_key(simulation: Simulation, key: int) -> None:
    """Ánh xạ phím điều khiển sang hành động mô phỏng."""
    actions = {
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
    }
    if key in actions:
        dx, dy = actions[key]
        simulation.move(dx, dy)
    elif key == pygame.K_SPACE:
        simulation.charge()
    elif key == pygame.K_r:
        simulation.reload_stations()


def main() -> None:
    simulation = Simulation()
    renderer = PygameRenderer()
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        handle_key(simulation, event.key)
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and renderer.reload_button_rect().collidepoint(event.pos)
                ):
                    simulation.reload_stations()
            renderer.screen.fill(COLORS["background"])
            renderer.draw_grid(simulation)
            renderer.draw_sidebar(simulation)
            pygame.display.flip()
            renderer.clock.tick(FPS)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
