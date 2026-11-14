"""Bộ hiển thị giao diện đồ họa Pygame cho môi trường EVChargingEnv.

Giữ nguyên hoàn toàn phong cách thẩm mỹ, màu sắc, bo góc, ký hiệu,
nhãn thông số và thanh bên (sidebar) của bản mô phỏng gốc.
Đồng thời khắc phục:
1. Nhãn 'LỐI VÀO' được dời lên trên lưới (MAP_TOP - 42) tránh đè ô đường đi.
2. Phương thức render() chỉ vẽ khung hình và gọi pygame.event.pump()
   thay vì tiêu thụ pygame.event.get() để không gây xung đột sự kiện.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pygame

from config import (
    BUSY,
    COLORS,
    FAST,
    FPS,
    MAP_LEFT,
    MAP_TOP,
    NORMAL,
    OBSTACLE,
    ROAD,
    SIDEBAR_LEFT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

if TYPE_CHECKING:
    from env.ev_charging_env import EVChargingEnv


class PygameRenderer:
    """Bộ vẽ giao diện Pygame cho môi trường EVChargingEnv."""

    def __init__(self, env: Optional["EVChargingEnv"] = None) -> None:
        """Khởi tạo cửa sổ Pygame và phông chữ."""
        pygame.init()
        pygame.display.set_caption("EV Charging Station - Gymnasium RL Environment")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Khởi tạo hệ thống phông chữ giao diện
        self.font = pygame.font.SysFont("segoeui", 16)
        self.small_font = pygame.font.SysFont("segoeui", 13)
        self.bold_font = pygame.font.SysFont("segoeui", 16, bold=True)
        self.title_font = pygame.font.SysFont("segoeui", 22, bold=True)

    def text(
        self,
        value: str,
        position: tuple[int, int],
        font: Optional[pygame.font.Font] = None,
        color: Optional[tuple[int, int, int]] = None,
    ) -> None:
        """Hỗ trợ vẽ văn bản nhanh lên màn hình."""
        surface = (font or self.font).render(
            str(value), True, color or COLORS["text"]
        )
        self.screen.blit(surface, position)

    def _get_cell_size(self, env: "EVChargingEnv") -> int:
        """Tính toán kích thước ô co giãn linh hoạt phù hợp với 10x10 và 15x15."""
        max_dim = max(env.grid_width, env.grid_height)
        return 585 // max_dim

    def cell_rect(self, x: int, y: int, cell_size: int, inset: int = 2) -> pygame.Rect:
        """Trả về hình chữ nhật của ô tại tọa độ lưới (x, y)."""
        return pygame.Rect(
            MAP_LEFT + x * cell_size + inset,
            MAP_TOP + y * cell_size + inset,
            cell_size - inset * 2,
            cell_size - inset * 2,
        )

    def draw_grid(self, env: "EVChargingEnv") -> None:
        """Vẽ toàn bộ lưới bản đồ, tọa độ, trạm sạc, vệt đường và xe điện."""
        cell_size = self._get_cell_size(env)

        # 1. Vẽ chỉ số tọa độ X (cột) và Y (hàng)
        for x in range(env.grid_width):
            self.text(
                str(x),
                (MAP_LEFT + x * cell_size + cell_size // 2 - 4, MAP_TOP - 25),
                self.small_font,
                COLORS["muted"],
            )
        for y in range(env.grid_height):
            self.text(
                str(y),
                (MAP_LEFT - 25, MAP_TOP + y * cell_size + cell_size // 2 - 8),
                self.small_font,
                COLORS["muted"],
            )

        # 2. Vẽ từng ô trong lưới ma trận
        for y, row in enumerate(env.grid):
            for x, code in enumerate(row):
                color = {
                    ROAD: COLORS["road"],
                    OBSTACLE: COLORS["wall"],
                    NORMAL: COLORS["normal"],
                    FAST: COLORS["fast"],
                    BUSY: COLORS["busy"],
                }.get(code, COLORS["road"])

                rect = self.cell_rect(x, y, cell_size)
                pygame.draw.rect(
                    self.screen,
                    color,
                    rect,
                    border_radius=6 if code != ROAD else 0,
                )
                pygame.draw.rect(self.screen, COLORS["grid"], rect, width=1)

                # Vẽ ký hiệu trên từng loại trụ sạc
                if code == NORMAL:
                    self.text(
                        "N",
                        (rect.centerx - 6, rect.centery - 10),
                        self.bold_font,
                        COLORS["text"],
                    )
                elif code == FAST:
                    self.text(
                        "F",
                        (rect.centerx - 5, rect.centery - 10),
                        self.bold_font,
                        COLORS["text"],
                    )
                elif code == BUSY:
                    self.text(
                        "N/F",
                        (rect.centerx - 12, rect.centery - 8),
                        self.small_font,
                        COLORS["text"],
                    )

        # 3. Vẽ vệt đường đi của xe (Trail)
        if len(env.trail) > 1:
            points = [
                (
                    MAP_LEFT + tx * cell_size + cell_size // 2,
                    MAP_TOP + ty * cell_size + cell_size // 2,
                )
                for tx, ty in env.trail
            ]
            pygame.draw.lines(self.screen, COLORS["trail"], False, points, 4)

        # 4. Vẽ nhãn 'LỐI VÀO' dời lên trên lưới tại MAP_TOP - 42 để không đè ô đường
        entrance_col = getattr(env, "entrance_col", env.grid_width - 1)
        entrance_label = "LỐI VÀO"
        entrance_center_x = MAP_LEFT + entrance_col * cell_size + cell_size // 2
        entrance_text_width = self.small_font.size(entrance_label)[0]
        entrance_x = entrance_center_x - entrance_text_width // 2
        self.text(
            entrance_label,
            (entrance_x, MAP_TOP - 42),
            self.small_font,
            COLORS["car"],
        )

        # 5. Vẽ xe điện tại vị trí hiện tại
        car_x, car_y = env.agent_pos
        center = (
            MAP_LEFT + car_x * cell_size + cell_size // 2,
            MAP_TOP + car_y * cell_size + cell_size // 2,
        )
        outer_radius = min(14, cell_size // 2 - 3)
        inner_radius = min(10, cell_size // 2 - 7)
        pygame.draw.circle(self.screen, COLORS["white"], center, outer_radius)
        pygame.draw.circle(self.screen, COLORS["car"], center, inner_radius)

    def draw_legend_item(
        self, y: int, color: tuple[int, int, int], label: str, symbol: str = ""
    ) -> None:
        """Vẽ một dòng chú thích trong bảng chú thích."""
        swatch = pygame.Rect(SIDEBAR_LEFT + 20, y, 22, 22)
        pygame.draw.rect(self.screen, color, swatch, border_radius=5)
        if symbol:
            self.text(symbol, (swatch.x + 3, swatch.y + 3), self.small_font, COLORS["text"])
        self.text(label, (SIDEBAR_LEFT + 52, y + 3), self.small_font, COLORS["muted"])

    def reload_button_rect(self) -> pygame.Rect:
        """Vị trí hình chữ nhật của nút 'Reload Trụ Sạc (R)'."""
        return pygame.Rect(SIDEBAR_LEFT + 20, 570, 231, 36)

    def draw_sidebar(self, env: "EVChargingEnv") -> None:
        """Vẽ thanh bên phải hiển thị bảng chú thích, thống số và trạng thái."""
        panel = pygame.Rect(SIDEBAR_LEFT, 20, WINDOW_WIDTH - SIDEBAR_LEFT - 20, 640)
        pygame.draw.rect(self.screen, COLORS["panel"], panel, border_radius=14)

        # Tiêu đề giao diện
        self.text("EV CHARGING", (SIDEBAR_LEFT + 20, 38), self.title_font)
        self.text(
            "Xe điện tự động tìm trạm sạc tối ưu",
            (SIDEBAR_LEFT + 20, 70),
            self.small_font,
            COLORS["muted"],
        )

        # Khối chú thích
        self.text("CHÚ THÍCH", (SIDEBAR_LEFT + 20, 106), self.bold_font)
        y = 134
        self.draw_legend_item(y, COLORS["car"], "Xe điện")
        self.draw_legend_item(y + 28, COLORS["trail"], "Vệt đường đi")
        self.draw_legend_item(y + 56, COLORS["wall"], "Vật cản / tường bao")
        self.draw_legend_item(y + 84, COLORS["normal"], "Sạc thường trống  +100", "N")
        self.draw_legend_item(y + 112, COLORS["fast"], "Sạc nhanh trống  +120", "F")
        self.draw_legend_item(y + 140, COLORS["busy"], "Trụ đang bận  -100", "N/F")

        # Khối thông số mô phỏng
        y += 196
        self.text("THÔNG SỐ MÔ PHỎNG", (SIDEBAR_LEFT + 20, y), self.bold_font)
        y += 32
        reward_display = int(env.cumulative_reward) if env.cumulative_reward.is_integer() else f"{env.cumulative_reward:.1f}"
        values = (
            ("Step", f"{env.step_count}/{env.max_steps}"),
            ("Reward", reward_display),
            ("Collisions", env.collision_attempts),
            ("Kịch bản", env.current_scenario_id),
        )
        for label, val in values:
            self.text(label, (SIDEBAR_LEFT + 22, y), self.small_font, COLORS["muted"])
            self.text(str(val), (SIDEBAR_LEFT + 165, y), self.bold_font)
            y += 28

        # Khối hiển thị trạng thái
        y += 8
        self.text("TRẠNG THÁI", (SIDEBAR_LEFT + 20, y), self.bold_font)
        self.text(
            env.status,
            (SIDEBAR_LEFT + 22, y + 30),
            self.small_font,
            COLORS["muted"],
        )

        # Nút Reload Trụ Sạc
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

        # Hướng dẫn điều khiển
        self.text(
            "Mũi tên: di chuyển    Space: Charge    R: Reload",
            (SIDEBAR_LEFT + 20, 625),
            self.small_font,
            COLORS["muted"],
        )

    def render(self, env: "EVChargingEnv") -> Any:
        """Vẽ khung hình hiện tại và cập nhật màn hình mà không tiêu thụ event queue."""
        pygame.event.pump()
        self.screen.fill(COLORS["background"])
        self.draw_grid(env)
        self.draw_sidebar(env)
        pygame.display.flip()
        self.clock.tick(FPS)

        if env.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    def close(self) -> None:
        """Giải phóng tài nguyên hiển thị Pygame."""
        pygame.quit()

