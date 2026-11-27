"""Trình diễn trực quan thuật toán Q-Learning (và so sánh Random Policy) trên Pygame.

Cách chạy:
    .venv\\Scripts\\python.exe scripts\\demo_q_learning.py --scenario C03

Phím điều khiển trong cửa sổ Pygame:
    [1 - 8]  Chuyển nhanh giữa 8 kịch bản C01 - C08 (tự nạp Q-table tương ứng)
    [Tab]    Đổi qua lại giữa Q-Learning và Random Policy để so sánh trực quan
    [Space]  Tạm dừng / Chạy tiếp (hoặc chạy tập mới nếu tập cũ đã kết thúc)
    [R]      Chạy lại kịch bản hiện tại (hoặc bấm nút 'Reload Trụ Sạc (R)')
    [+ / -]  Tăng / giảm tốc độ di chuyển của xe
    [Esc]    Thoát chương trình
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pygame

from agents.q_learning import QLearningAgent
from agents.random_agent import RandomAgent
import config
from env.ev_charging_env import EVChargingEnv
from scripts.train_q_learning import train_q_learning

ALL_SCENARIOS = [f"C0{i}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo trực quan Q-Learning & Random Policy trên giao diện Pygame."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="C03",
        choices=ALL_SCENARIOS,
        help="Kịch bản mở đầu (C01-C08, mặc định: C03).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed của Q-table cần nạp (mặc định: 42).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="q_learning",
        choices=["q_learning", "random"],
        help="Tác tử khởi đầu ('q_learning' hoặc 'random', mặc định: q_learning).",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="default_10x10",
        choices=list(config.MAP_CONFIGS.keys()),
        help="Bản đồ hiển thị (mặc định: default_10x10).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Thời gian trễ giữa mỗi bước di chuyển tính bằng giây (mặc định: 0.35s).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Thư mục chứa tệp Q-table .npy đã huấn luyện (mặc định: models).",
    )
    return parser.parse_args()


def load_or_train_q_agent(
    map_name: str,
    scenario_id: str,
    seed: int,
    models_dir: str,
) -> QLearningAgent:
    """Nạp Q-table đã lưu theo kịch bản; nếu chưa có sẽ tự huấn luyện nhanh 3000 tập."""
    agent = QLearningAgent(action_space=config.NUM_ACTIONS, seed=seed)
    specific_path = os.path.join(models_dir, f"q_learning_table_{scenario_id}_seed{seed}.npy")
    default_path = os.path.join(models_dir, "q_learning_table.npy")

    if map_name == "default_10x10" and os.path.isfile(specific_path):
        agent.load(specific_path)
        print(f"[MODEL] Đã nạp Q-table: {specific_path} ({len(agent.q_table)} trạng thái)")
        return agent

    if map_name == "default_10x10" and os.path.isfile(default_path):
        agent.load(default_path)
        print(f"[MODEL] Đã nạp Q-table mặc định: {default_path} ({len(agent.q_table)} trạng thái)")
        return agent

    print(f"[TRAIN] Chưa có Q-table cho {scenario_id} ({map_name}), đang huấn luyện nhanh 3000 episodes...")
    train_env = EVChargingEnv(map_name=map_name, obs_type="tuple", render_mode=None)
    train_q_learning(train_env, agent, episodes=3000, scenario_id=scenario_id, seed=seed, quiet=True)
    train_env.close()
    print(f"[TRAIN] Hoàn tất! ({len(agent.q_table)} trạng thái)")
    return agent


def main() -> None:
    args = parse_args()
    current_scenario = args.scenario.upper()
    agent_mode = args.agent
    step_delay = max(0.05, args.delay)
    episode_idx = 0

    print("=" * 72)
    print(" DEMO TRỰC QUAN PYGAME: Q-LEARNING vs RANDOM POLICY")
    print("  [1-8] Đổi kịch bản C01-C08 | [Tab] Đổi Q-Learning <-> Random")
    print("  [Space] Tạm dừng / Chạy lại | [R] Reload | [+/-] Tốc độ | [Esc] Thoát")
    print("=" * 72)

    env = EVChargingEnv(map_name=args.map, obs_type="tuple", render_mode="human")
    q_agent = load_or_train_q_agent(args.map, current_scenario, args.seed, args.models_dir)
    random_agent = RandomAgent(action_space=config.NUM_ACTIONS, seed=args.seed)

    def reset_episode(reload_model: bool = False) -> Any:
        nonlocal q_agent, episode_idx
        episode_idx += 1
        if reload_model:
            q_agent = load_or_train_q_agent(args.map, current_scenario, args.seed, args.models_dir)
        obs, _ = env.reset(
            seed=args.seed * 1000 + episode_idx,
            options={"scenario": current_scenario},
        )
        algo_label = "Q-Learning" if agent_mode == "q_learning" else "Random Policy"
        pygame.display.set_caption(
            f"EV Charging Demo | {algo_label} | Kịch bản {current_scenario} | Delay {step_delay:.2f}s"
        )
        print(f"\n=== TẬP #{episode_idx} | {algo_label} | Kịch bản {current_scenario} ===")
        return obs

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

    obs = reset_episode(reload_model=False)
    paused = False
    running = True
    last_step_time = time.perf_counter()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs = reset_episode(reload_model=False)
                        paused = False
                        last_step_time = time.perf_counter()
                    elif event.key == pygame.K_SPACE:
                        if env.terminated or env.truncated:
                            obs = reset_episode(reload_model=False)
                            paused = False
                        else:
                            paused = not paused
                            print("[PAUSE]" if paused else "[RESUME]")
                        last_step_time = time.perf_counter()
                    elif event.key == pygame.K_TAB:
                        agent_mode = "random" if agent_mode == "q_learning" else "q_learning"
                        obs = reset_episode(reload_model=False)
                        paused = False
                        last_step_time = time.perf_counter()
                    elif event.key in scenario_keys:
                        current_scenario = scenario_keys[event.key]
                        obs = reset_episode(reload_model=True)
                        paused = False
                        last_step_time = time.perf_counter()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        step_delay = max(0.05, step_delay - 0.05)
                        print(f"[SPEED] Delay mỗi bước: {step_delay:.2f}s")
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        step_delay = min(1.50, step_delay + 0.05)
                        print(f"[SPEED] Delay mỗi bước: {step_delay:.2f}s")
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and env.renderer is not None
                    and env.renderer.reload_button_rect().collidepoint(event.pos)
                ):
                    obs = reset_episode(reload_model=False)
                    paused = False
                    last_step_time = time.perf_counter()

            now = time.perf_counter()
            if not paused and not (env.terminated or env.truncated) and (now - last_step_time >= step_delay):
                active_agent = q_agent if agent_mode == "q_learning" else random_agent
                action = active_agent.select_action(obs, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                last_step_time = now

                print(
                    f"  Bước {info['step_count']:02d} | {config.ACTION_NAMES[action]:<14} | "
                    f"Thưởng: {reward:>+6.1f} | Tổng: {env.cumulative_reward:>+6.1f} | "
                    f"Va chạm: {info['collision_attempts']}"
                )
                if terminated or truncated:
                    result_txt = "THÀNH CÔNG" if info["is_success"] else "THẤT BẠI / HẾT BƯỚC"
                    print(
                        f"-> KẾT THÚC: {result_txt} | Trụ: {info['charged_type']} | "
                        f"Tổng điểm: {env.cumulative_reward:+.1f} | Nhấn Space/R để chạy tiếp hoặc 1-8 đổi kịch bản."
                    )

            env.render()
    finally:
        env.close()
        print("\nĐã đóng cửa sổ demo Pygame.")


if __name__ == "__main__":
    main()
