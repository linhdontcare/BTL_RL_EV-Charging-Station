"""Mô phỏng xe điện tìm đường đến trạm sạc - Điểm chuyển tiếp (Entry Point).

Tệp này đóng vai trò tương thích ngược (Backward Compatibility) sau khi
dự án đã được tái cấu trúc (refactored) sang kiến trúc chuẩn Gymnasium:
- Cấu hình chung & Hằng số : config.py
- Môi trường chuẩn Gymnasium : env/ev_charging_env.py
- Bộ sinh 8 kịch bản C01-C08 : env/scenarios.py
- Bộ hiển thị đồ họa Pygame   : gui/pygame_renderer.py
- Điểm chạy thử nghiệm thủ công: demo_manual.py

Điều khiển (khi chạy trực tiếp):
    Mũi tên (Up, Down, Left, Right): Di chuyển xe điện
    Space: Sạc (Charge) tại ô hiện tại
    R: Reset lại episode
    Phím 1-8: Đổi nhanh kịch bản C01-C08
    M: Đổi bản đồ 10x10 và 15x15
    Esc: Thoát
"""

from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (
    BUSY,
    COLORS,
    FAST,
    GAMMA,
    MAP_CONFIGS,
    MAX_STEPS,
    NORMAL,
    OBSTACLE,
    ROAD,
    SEEDS,
)
from demo_manual import main
from env.ev_charging_env import EVChargingEnv
from env.scenarios import ScenarioData, setup_scenario
from gui.pygame_renderer import PygameRenderer

__all__ = [
    "ROAD",
    "OBSTACLE",
    "NORMAL",
    "FAST",
    "BUSY",
    "MAX_STEPS",
    "GAMMA",
    "SEEDS",
    "COLORS",
    "MAP_CONFIGS",
    "EVChargingEnv",
    "ScenarioData",
    "setup_scenario",
    "PygameRenderer",
    "main",
]

if __name__ == "__main__":
    main()
