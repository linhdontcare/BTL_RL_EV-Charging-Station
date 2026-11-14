"""Chương trình chạy thử nghiệm thủ công môi trường EVChargingEnv bằng bàn phím.

Điều khiển:
    Mũi tên (Lên, Xuống, Trái, Phải): Di chuyển xe điện
    Space: Thực hiện lệnh Sạc (Charge) tại ô hiện tại
    R: Reset lại episode với kịch bản hiện tại
    Phím số 1 đến 8: Chuyển nhanh giữa 8 kịch bản chuẩn C01 đến C08
    M: Chuyển đổi giữa bản đồ cơ sở 10x10 và bản đồ mở rộng 15x15
    Esc hoặc đóng cửa sổ: Thoát chương trình
"""

from __future__ import annotations

import argparse
import os
import sys

# Đảm bảo đường dẫn import hoạt động độc lập
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import pygame

from config import (
    ACTION_CHARGE,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_UP,
)
from env.ev_charging_env import EVChargingEnv


def parse_arguments() -> argparse.Namespace:
    """Xử lý tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm thủ công EVChargingEnv")
    parser.add_argument(
        "--map",
        type=str,
        default="default_10x10",
        choices=["default_10x10", "extended_15x15"],
        help="Chọn bản đồ mô phỏng ('default_10x10' hoặc 'extended_15x15')",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="C06",
        help="Mã kịch bản thử nghiệm: C01 đến C08 (mặc định C06 - ngẫu nhiên)",
    )
    parser.add_argument(
        "--obs_type",
        type=str,
        default="tuple",
        choices=["tuple", "box"],
        help="Định dạng quan sát ('tuple' cho Tabular RL, 'box' cho DQN)",
    )
    return parser.parse_args()


def print_step_summary(
    action_name: str,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    cumulative_reward: float,
) -> None:
    """In thông tin chi tiết từng bước hành động ra màn hình console."""
    status_tag = ""
    if terminated:
        status_tag = " -> [TERMINATED]"
    elif truncated:
        status_tag = " -> [TRUNCATED]"

    print(
        f"[Bước {info['step_count']:03d}] {action_name:<16} | "
        f"Thưởng bước: {reward:>+6.1f} | "
        f"Tổng tích lũy: {cumulative_reward:>+6.1f} | "
        f"Va chạm: {info['collision_attempts']}{status_tag}"
    )

    if terminated or truncated:
        print("-" * 65)
        print("=== KẾT THÚC EPISODE ===")
        print(f"  * Kịch bản       : {info['scenario_id']}")
        print(f"  * Kết quả        : {'THÀNH CÔNG (+)' if info['is_success'] else 'THẤT BẠI (-)'}")
        print(f"  * Loại trụ đã sạc: {info['charged_type'] or 'Chưa sạc / Sai vị trí'}")
        print(f"  * Tổng điểm thưởng: {cumulative_reward:+.1f}")
        print(f"  * Tổng số bước   : {info['step_count']}")
        print(f"  * Tổng số va chạm: {info['collision_attempts']}")
        print("  -> Nhấn phím 'R' để chơi lại hoặc phím 1-8 để đổi kịch bản C01-C08.")
        print("-" * 65)


def main() -> None:
    """Vòng lặp chính chạy thử nghiệm giao diện."""
    args = parse_arguments()
    current_map = args.map
    current_scenario = args.scenario.upper()

    print("=" * 65)
    print(" KHỞI CHẠY MÔ PHỎNG THỦ CÔNG: EV CHARGING GYMNASIUM ENV")
    print(f"  * Bản đồ: {current_map}")
    print(f"  * Kịch bản ban đầu: {current_scenario}")
    print(f"  * Định dạng quan sát: {args.obs_type}")
    print("=" * 65)
    print("Phím điều khiển:")
    print("  [Mũi tên] Di chuyển xe điện")
    print("  [Space]   Sạc điện (Charge)")
    print("  [R]       Reset lại episode hiện tại")
    print("  [1 - 8]   Chuyển nhanh sang các kịch bản chuẩn C01 - C08")
    print("  [M]       Đổi bản đồ giữa 10x10 và 15x15")
    print("  [Esc]     Thoát chương trình")
    print("=" * 65)

    env = EVChargingEnv(
        map_name=current_map,
        obs_type=args.obs_type,
        render_mode="human",
    )

    obs, info = env.reset(options={"scenario": current_scenario})
    print(f"[RESET] Bắt đầu kịch bản {current_scenario} trên bản đồ {current_map}.")

    key_action_map = {
        pygame.K_UP: (ACTION_UP, ACTION_NAMES[ACTION_UP]),
        pygame.K_DOWN: (ACTION_DOWN, ACTION_NAMES[ACTION_DOWN]),
        pygame.K_LEFT: (ACTION_LEFT, ACTION_NAMES[ACTION_LEFT]),
        pygame.K_RIGHT: (ACTION_RIGHT, ACTION_NAMES[ACTION_RIGHT]),
        pygame.K_SPACE: (ACTION_CHARGE, ACTION_NAMES[ACTION_CHARGE]),
    }

    scenario_keys = {
        pygame.K_1: "C01",
        pygame.K_2: "C02",
        pygame.K_3: "C03",
        pygame.K_4: "C04",
        pygame.K_5: "C05",
        pygame.K_6: "C06",
        pygame.K_7: "C07",
        pygame.K_8: "C08",
    }

    running = True
    try:
        while running:
            # Thu nhận toàn bộ sự kiện Pygame tại đây (không bị PygameRenderer nuốt mất)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    # Phím Reset (R)
                    elif event.key == pygame.K_r:
                        obs, info = env.reset(options={"scenario": current_scenario})
                        print(f"\n[RESET] Đã làm mới kịch bản {current_scenario}.")

                    # Phím đổi kịch bản C01 đến C08
                    elif event.key in scenario_keys:
                        current_scenario = scenario_keys[event.key]
                        obs, info = env.reset(options={"scenario": current_scenario})
                        print(f"\n[CHUYỂN KỊCH BẢN] Đã chuyển sang {current_scenario}.")

                    # Phím đổi bản đồ (M)
                    elif event.key == pygame.K_m:
                        current_map = "extended_15x15" if current_map == "default_10x10" else "default_10x10"
                        env.close()
                        env = EVChargingEnv(
                            map_name=current_map,
                            obs_type=args.obs_type,
                            render_mode="human",
                        )
                        obs, info = env.reset(options={"scenario": current_scenario})
                        print(f"\n[ĐỔI BẢN ĐỒ] Đã chuyển sang bản đồ {current_map}.")

                    # Phím thực thi hành động (nếu episode chưa kết thúc)
                    elif event.key in key_action_map:
                        if not env.terminated and not env.truncated:
                            action_code, action_name = key_action_map[event.key]
                            obs, reward, terminated, truncated, info = env.step(action_code)
                            print_step_summary(
                                action_name,
                                reward,
                                terminated,
                                truncated,
                                info,
                                env.cumulative_reward,
                            )

                # Click chuột vào nút Reload Trụ Sạc
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and env.renderer is not None
                    and env.renderer.reload_button_rect().collidepoint(event.pos)
                ):
                    obs, info = env.reset(options={"scenario": current_scenario})
                    print(f"\n[CLICK RELOAD] Đã làm mới kịch bản {current_scenario}.")

            # Vẽ màn hình hiện tại
            env.render()

    finally:
        env.close()
        print("\nĐã đóng chương trình mô phỏng an toàn.")


if __name__ == "__main__":
    main()

