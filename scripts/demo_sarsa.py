"""Chương trình trình diễn trực quan thuật toán SARSA trên giao diện Pygame.

Tính năng:
- Khởi tạo môi trường EVChargingEnv với render_mode="human".
- Tự động nạp Q-table đã huấn luyện từ models/sarsa_table_{scenario}_seed42.npy.
- Điều khiển xe điện tự động bằng chính sách khai thác tham lam (eval_mode=True).
- Tạm nghỉ giữa mỗi bước đi (mặc định 0.35s) kết hợp vòng lặp sự kiện Pygame 30 FPS
  giúp quan sát trực quan và tránh treo cửa sổ trên macOS.
- Phím tắt điều khiển:
    [1 - 8] : Chuyển nhanh kịch bản C01-C08 và tự động nạp model tương ứng.
    [R]     : Chạy lại episode hiện tại từ đầu.
    [Esc] hoặc nút đóng cửa sổ : Thoát an toàn.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

# Đảm bảo đường dẫn import hoạt động từ mọi thư mục thực thi
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pygame

from agents.sarsa import SarsaAgent
from config import ACTION_NAMES, MAP_CONFIGS
from env.ev_charging_env import EVChargingEnv


def parse_args() -> argparse.Namespace:
    """Xử lý tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Trình diễn trực quan tác tử SARSA trên môi trường EVChargingEnv."
    )
    parser.add_argument(
        "--scenario",
        type=str.upper,
        default="C03",
        choices=[f"C0{i}" for i in range(1, 9)],
        help="Mã kịch bản trình diễn (C01 đến C08, mặc định: C03).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Thời gian nghỉ giữa mỗi bước tính bằng giây (mặc định: 0.35).",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="default_10x10",
        choices=list(MAP_CONFIGS.keys()),
        help="Tên bản đồ mô phỏng (mặc định: default_10x10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed của model cần nạp (mặc định: 42).",
    )
    return parser.parse_args()


def load_sarsa_model(agent: SarsaAgent, scenario_id: str, seed: int = 42) -> bool:
    """Nạp Q-table của kịch bản tương ứng vào SarsaAgent.

    Trả về True nếu nạp thành công, False nếu chưa có tệp model (không làm crash).
    """
    rel_path = os.path.join("models", f"sarsa_table_{scenario_id}_seed{seed}.npy")
    abs_path = os.path.join(ROOT_DIR, rel_path)

    if not os.path.isfile(abs_path):
        # Xóa Q-table cũ để tránh dùng nhầm tri thức từ kịch bản trước đó
        agent.q_table.clear()
        print("\n" + "!" * 72)
        print(f"[THÔNG BÁO] Chưa tìm thấy file model cho kịch bản {scenario_id}: {rel_path}")
        print("Hãy huấn luyện model trước bằng lệnh:")
        print(f"    python3 scripts/train_sarsa.py --scenario {scenario_id} --seed {seed}")
        print("Nhấn phím [1-8] để chuyển sang kịch bản khác hoặc [Esc] để thoát.")
        print("!" * 72)
        return False

    agent.load(abs_path)
    print(f"\n[OK] Đã nạp Q-table SARSA từ: {rel_path} ({len(agent.q_table)} trạng thái)")
    return True


def build_scenario_keymap() -> dict[int, str]:
    """Tạo bảng ánh xạ phím số 1-8 (bao gồm hàng phím số và Numpad) sang mã C01-C08."""
    keymap: dict[int, str] = {}
    for i in range(1, 9):
        scenario_id = f"C0{i}"
        main_key = getattr(pygame, f"K_{i}", None)
        kp_key = getattr(pygame, f"K_KP{i}", None)
        if main_key is not None:
            keymap[main_key] = scenario_id
        if kp_key is not None:
            keymap[kp_key] = scenario_id
    return keymap


def reset_demo_episode(
    env: EVChargingEnv,
    scenario_id: str,
    model_loaded: bool,
    seed: int | None = None,
) -> tuple[Any, dict[str, Any], float, bool, float]:
    """Đặt lại episode cho kịch bản hiện tại và cập nhật trạng thái hiển thị."""
    obs, info = env.reset(seed=seed, options={"scenario": scenario_id})
    if not model_loaded:
        env.status = f"Chưa có model {scenario_id} (cần train)"
        env.render()

    print(f"\n--- Bắt đầu trình diễn kịch bản {scenario_id} | Vị trí đầu: {obs[:2]} | Trạng thái trụ: {obs[2]} ---")
    return obs, info, 0.0, False, time.perf_counter()


def print_episode_summary(scenario_id: str, info: dict[str, Any], total_reward: float) -> None:
    """In tóm tắt kết quả sau khi episode kết thúc."""
    status_text = "THÀNH CÔNG" if info["is_success"] else "THẤT BẠI"
    print(
        f"-> Kết thúc {scenario_id}: [{status_text}] | "
        f"Loại sạc: {info['charged_type'] or 'Không'} | "
        f"Tổng bước: {info['step_count']} | "
        f"Tổng reward: {total_reward:.1f} | "
        f"Va chạm: {info['collision_attempts']}"
    )
    print("   (Nhấn [R] để chạy lại, [1-8] để đổi kịch bản, [Esc] để thoát)")


def main() -> None:
    """Điểm khởi chạy chương trình trình diễn trực quan SARSA."""
    args = parse_args()
    step_delay = max(0.0, float(args.delay))
    current_scenario = args.scenario
    seed = args.seed

    print("=" * 72)
    print("TRÌNH DIỄN TRỰC QUAN THUẬT TOÁN SARSA - EV CHARGING STATION")
    print("=" * 72)
    print(f"Kịch bản ban đầu : {current_scenario}")
    print(f"Bản đồ           : {args.map}")
    print(f"Độ trễ mỗi bước  : {step_delay:.2f}s")
    print("Phím điều khiển  : [1-8] Đổi kịch bản | [R] Chạy lại | [Esc] Thoát")
    print("=" * 72)

    # Khởi tạo EVChargingEnv với render_mode="human" chuẩn Gymnasium
    env = EVChargingEnv(
        map_name=args.map,
        obs_type="tuple",
        render_mode="human",
    )

    # Khởi tạo SarsaAgent (eval_mode=True sẽ chạy hoàn toàn tham lam theo Q-table)
    agent = SarsaAgent(action_space=env.action_space.n, seed=seed)
    scenario_keymap = build_scenario_keymap()
    clock = pygame.time.Clock()

    try:
        model_loaded = load_sarsa_model(agent, current_scenario, seed=seed)
        obs, info, total_reward, done, last_step_time = reset_demo_episode(
            env, current_scenario, model_loaded, seed=seed
        )

        running = True
        while running:
            rendered_this_frame = False

            # 1. Xử lý liên tục hàng đợi sự kiện Pygame để tránh treo cửa sổ trên macOS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

                    if event.key == pygame.K_r:
                        obs, info, total_reward, done, last_step_time = reset_demo_episode(
                            env, current_scenario, model_loaded
                        )
                        rendered_this_frame = True
                        continue

                    if event.key in scenario_keymap:
                        new_scenario = scenario_keymap[event.key]
                        current_scenario = new_scenario
                        model_loaded = load_sarsa_model(agent, current_scenario, seed=seed)
                        obs, info, total_reward, done, last_step_time = reset_demo_episode(
                            env, current_scenario, model_loaded
                        )
                        rendered_this_frame = True
                        continue

            if not running:
                break

            # 2. Tự động cho xe thực hiện bước tiếp theo sau khoảng nghỉ step_delay
            now = time.perf_counter()
            if model_loaded and not done and (now - last_step_time >= step_delay):
                action = agent.select_action(obs, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                rendered_this_frame = True
                total_reward += reward
                done = terminated or truncated
                last_step_time = time.perf_counter()

                action_label = ACTION_NAMES.get(action, str(action))
                print(
                    f"  Bước {info['step_count']:2d} | {action_label:<14} | "
                    f"Vị trí: {obs[:2]} | Reward: {reward:6.1f} | Tổng: {total_reward:6.1f}"
                )

                if done:
                    print_episode_summary(current_scenario, info, total_reward)

            # 3. Duy trì vẽ lại giao diện mượt mà ở ~30 FPS khi chưa render trong khung hình này
            if not rendered_this_frame:
                frame_start = time.perf_counter()
                env.render()
                if time.perf_counter() - frame_start < 0.02:
                    clock.tick(30)

    except KeyboardInterrupt:
        print("\nĐã dừng trình diễn theo yêu cầu người dùng.")
    finally:
        # Đóng tài nguyên môi trường và Pygame an toàn trên macOS
        env.close()
        if pygame.get_init():
            pygame.quit()
        print("Đã đóng cửa sổ Pygame an toàn.")


if __name__ == "__main__":
    main()
