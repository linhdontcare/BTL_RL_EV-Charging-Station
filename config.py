"""Cấu hình dùng chung cho môi trường mô phỏng xe điện tìm trạm sạc.

Tệp này chứa toàn bộ các hằng số, luật thưởng phạt, cấu hình bản đồ
và tham số đồ họa Pygame để đảm bảo tính nhất quán trên toàn bộ dự án.
"""

from __future__ import annotations

from typing import Any

# ===========================================================================
# 1. MÃ HÓA Ô LƯỚI (GRID CELL ENCODING)
# ===========================================================================
ROAD: int = 0       # Đường đi cho phép xe di chuyển
OBSTACLE: int = 1   # Vật cản, tường bao không thể đi qua
NORMAL: int = 2     # Trụ sạc thường đang trống
FAST: int = 3       # Trụ sạc nhanh đang trống
BUSY: int = 4       # Trụ sạc đang có xe chiếm dụng (bận), đóng vai trò như vật cản

CELL_TYPES = {
    ROAD: "Đường đi",
    OBSTACLE: "Vật cản / Tường",
    NORMAL: "Trụ sạc thường (Trống)",
    FAST: "Trụ sạc nhanh (Trống)",
    BUSY: "Trụ đang bận",
}

# ===========================================================================
# 2. HỆ THỐNG THƯỞNG / PHẠT (REWARDS)
# ===========================================================================
REWARD_FAST: float = 120.0          # Sạc nhanh thành công tại ô trống
REWARD_NORMAL: float = 100.0        # Sạc thường thành công tại ô trống
REWARD_BUSY_CHARGE: float = -100.0  # Phạt sạc tại trụ đang bận
REWARD_INVALID_CHARGE: float = -50.0 # Phạt sạc sai vị trí (trên đường đi)
REWARD_COLLISION: float = -100.0    # Phạt va chạm biên/vật cản/trụ bận (giữ nguyên vị trí)
REWARD_STEP: float = -5.0           # Phạt chi phí di chuyển mỗi bước hợp lệ

# ===========================================================================
# 3. THAM SỐ HUẤN LUYỆN VÀ ĐIỀU KIỆN DỪNG
# ===========================================================================
MAX_STEPS: int = 100                # Giới hạn chân trời (horizon limit)
GAMMA: float = 0.95                 # Hệ số chiết khấu chuẩn cho MDP
SEEDS: list[int] = [42, 100, 2026, 7, 99]  # Bộ seed chuẩn phục vụ tái lập thí nghiệm

# ===========================================================================
# 4. KHÔNG GIAN HÀNH ĐỘNG (ACTION SPACE)
# ===========================================================================
ACTION_UP: int = 0
ACTION_DOWN: int = 1
ACTION_LEFT: int = 2
ACTION_RIGHT: int = 3
ACTION_CHARGE: int = 4

NUM_ACTIONS: int = 5

ACTION_DELTAS: dict[int, tuple[int, int]] = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

ACTION_NAMES: dict[int, str] = {
    ACTION_UP: "Lên (Up)",
    ACTION_DOWN: "Xuống (Down)",
    ACTION_LEFT: "Trái (Left)",
    ACTION_RIGHT: "Phải (Right)",
    ACTION_CHARGE: "Sạc (Charge)",
}

# ===========================================================================
# 5. CẤU HÌNH GIAO DIỆN ĐỒ HỌA PYGAME
# ===========================================================================
WINDOW_WIDTH: int = 950
WINDOW_HEIGHT: int = 680
MAP_LEFT: int = 42
MAP_TOP: int = 48
CELL_SIZE_15: int = 39   # Kích thước ô chuẩn của bản đồ 15x15
MAP_SIZE: int = 585      # Vùng hiển thị bản đồ lưới (15 * 39 = 585)
SIDEBAR_LEFT: int = MAP_LEFT + MAP_SIZE + 32  # Tọa độ X bắt đầu của sidebar (659)
FPS: int = 30

# Bảng màu sắc giữ nguyên phong cách gốc của ev_charging_sim.py
COLORS: dict[str, tuple[int, int, int]] = {
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

# ===========================================================================
# 6. CẤU HÌNH BẢN ĐỒ (MAP CONFIGURATIONS)
# ===========================================================================

def _generate_default_10x10() -> dict[str, Any]:
    """Tạo cấu hình bản đồ cơ sở 10x10.
    
    Đặc điểm kỹ thuật:
    - Kích thước: 10x10 ô.
    - Tường bao có chừa lối vào tại ô (8, 0).
    - Vật cản nội bộ chiếm 18% diện tích (18 ô):
      + Khối nhà điều hành: x in [1..4], y in [1..3] (12 ô).
      + Dải phân cách giữa: x in [2..7], y = 6 (6 ô).
    - Đúng k = 4 trụ sạc cố định (2 NORMAL, 2 FAST):
      + (2, 5): NORMAL, (4, 5): FAST
      + (2, 7): NORMAL, (4, 7): FAST
      Q-table kích thước hữu hạn: 10 * 10 * 2^4 = 1600 trạng thái.
    - Điểm xuất phát mặc định: (8, 1) ngay lối vào.
    """
    width, height = 10, 10
    obstacles: set[tuple[int, int]] = set()

    # 1. Tường bao (ngoại trừ ô lối vào (8, 0))
    for x in range(width):
        if x != 8:
            obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    # 2. Khối vật cản nhà điều hành (12 ô)
    for y in range(1, 4):
        for x in range(1, 5):
            obstacles.add((x, y))

    # 3. Dải phân cách ngang giữa hai làn trạm (6 ô)
    for x in range(2, 8):
        obstacles.add((x, 6))

    # 4. Danh sách k = 4 trụ sạc cố định
    stations: list[dict[str, Any]] = [
        {"id": 0, "pos": (2, 5), "type": NORMAL, "type_name": "normal"},
        {"id": 1, "pos": (4, 5), "type": FAST, "type_name": "fast"},
        {"id": 2, "pos": (2, 7), "type": NORMAL, "type_name": "normal"},
        {"id": 3, "pos": (4, 7), "type": FAST, "type_name": "fast"},
    ]

    return {
        "name": "default_10x10",
        "width": width,
        "height": height,
        "start_pos": (8, 1),
        "entrance": (8, 0),
        "entrance_col": 8,
        "obstacles": obstacles,
        "stations": stations,
        "num_stations": 4,
    }


def _generate_extended_15x15() -> dict[str, Any]:
    """Tạo cấu hình bản đồ mở rộng 15x15 kế thừa bố cục ev_charging_sim.py.
    
    Đặc điểm kỹ thuật:
    - Kích thước: 15x15 ô.
    - Tường bao có mở hai lối vào tại (12, 0) và (13, 0).
    - Khối nhà điều hành tại x in [1..8], y in [1..3] (24 ô).
    - Dải phân cách hàng 6 và hàng 11 tại x in [3..9].
    - Rút gọn danh sách trụ sạc xuống k = 8 trụ sạc (4 NORMAL, 4 FAST):
      + Làn 1 (y = 5):  (3, 5) NORMAL, (5, 5) FAST
      + Làn 2 (y = 7):  (3, 7) FAST,   (5, 7) NORMAL
      + Làn 3 (y = 10): (3, 10) NORMAL, (5, 10) FAST
      + Làn 4 (y = 12): (3, 12) FAST,   (5, 12) NORMAL
    - Điểm xuất phát mặc định: (13, 1).
    """
    width, height = 15, 15
    obstacles: set[tuple[int, int]] = set()

    # 1. Tường bao (mở lối vào tại (12, 0) và (13, 0))
    for x in range(width):
        if x not in (12, 13):
            obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    # 2. Khối điều hành phía trên trái (24 ô)
    for y in range(1, 4):
        for x in range(1, 9):
            obstacles.add((x, y))

    # 3. Hai dải phân cách ngang
    for x in range(3, 10):
        obstacles.add((x, 6))
        obstacles.add((x, 11))

    # 4. Danh sách k = 8 trụ sạc cố định
    stations: list[dict[str, Any]] = [
        {"id": 0, "pos": (3, 5), "type": NORMAL, "type_name": "normal"},
        {"id": 1, "pos": (5, 5), "type": FAST, "type_name": "fast"},
        {"id": 2, "pos": (3, 7), "type": FAST, "type_name": "fast"},
        {"id": 3, "pos": (5, 7), "type": NORMAL, "type_name": "normal"},
        {"id": 4, "pos": (3, 10), "type": NORMAL, "type_name": "normal"},
        {"id": 5, "pos": (5, 10), "type": FAST, "type_name": "fast"},
        {"id": 6, "pos": (3, 12), "type": FAST, "type_name": "fast"},
        {"id": 7, "pos": (5, 12), "type": NORMAL, "type_name": "normal"},
    ]

    return {
        "name": "extended_15x15",
        "width": width,
        "height": height,
        "start_pos": (13, 1),
        "entrance": (13, 0),
        "entrance_col": 13,
        "obstacles": obstacles,
        "stations": stations,
        "num_stations": 8,
    }


MAP_CONFIGS: dict[str, dict[str, Any]] = {
    "default_10x10": _generate_default_10x10(),
    "extended_15x15": _generate_extended_15x15(),
}
