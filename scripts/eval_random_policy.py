"""Script chạy mốc tham chiếu cơ sở Random Policy trên các kịch bản C01-C08.

Bám sát Mục 2.6, Mục 3.1 và Thuật toán 4 trong báo cáo:
- Chạy RandomAgent (chọn ngẫu nhiên đồng nhất 5 hành động) trên cùng 8 kịch bản
  C01-C08 và bộ 5 hạt giống chuẩn [42, 100, 2026, 7, 99].
- Mỗi (kịch bản, seed) đánh giá N_eval = 100 episodes với hạt giống reset
  seed * 1000 + ep.
- Xuất bảng tổng hợp 5 chỉ số chuẩn hóa (Mean +/- Std) và lưu CSV để đối chiếu
  mức độ cải thiện của Q-Learning, SARSA và DQN.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from agents.random_agent import RandomAgent
import config
from env.ev_charging_env import EVChargingEnv

ALL_SCENARIOS: list[str] = [f"C0{i}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    """Xử lý đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Đánh giá mốc tham chiếu cơ sở Random Policy trên C01-C08."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=ALL_SCENARIOS,
        help="Chỉ định 1 kịch bản cụ thể (bỏ trống để chạy toàn bộ C01-C08).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Số episodes đánh giá mỗi kịch bản / seed (mặc định: 100).",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="default_10x10",
        choices=list(config.MAP_CONFIGS.keys()),
        help="Bản đồ mô phỏng ('default_10x10' hoặc 'extended_15x15').",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Thư mục lưu bảng kết quả CSV (mặc định: logs).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Mở cửa sổ Pygame trực quan hóa Random Policy chạy trên môi trường.",
    )
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=3,
        help="Số tập trình diễn trên Pygame khi bật --render (mặc định: 3).",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.25,
        help="Độ trễ (giây) giữa mỗi bước khi hiển thị Pygame (mặc định: 0.25).",
    )
    return parser.parse_args()


def evaluate_random_policy_seed(
    map_name: str,
    scenario_id: str,
    seed: int,
    eval_episodes: int = 100,
) -> dict[str, Any]:
    """Đánh giá RandomAgent cho một cặp (kịch bản, seed)."""
    env = EVChargingEnv(map_name=map_name, obs_type="tuple", render_mode=None)
    agent = RandomAgent(action_space=int(env.action_space.n), seed=seed)

    rewards: list[float] = []
    all_steps: list[int] = []
    success_steps: list[int] = []
    collisions: list[int] = []
    successes: list[int] = []

    for ep in range(1, eval_episodes + 1):
        s, info = env.reset(seed=seed * 1000 + ep, options={"scenario": scenario_id})
        ep_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            a = agent.select_action(s, eval_mode=True)
            s_next, r, terminated, truncated, info = env.step(a)
            ep_reward += r
            s = s_next

        rewards.append(ep_reward)
        all_steps.append(info["step_count"])
        collisions.append(info["collision_attempts"])
        if info["is_success"]:
            successes.append(1)
            success_steps.append(info["step_count"])
        else:
            successes.append(0)

    env.close()
    n_succ = len(success_steps)
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_success_steps": float(sum(success_steps) / max(1, n_succ)),
        "mean_all_steps": float(np.mean(all_steps)),
        "mean_reward": float(np.mean(rewards)),
        "mean_collisions": float(np.mean(collisions)),
        "train_time": 0.0,
    }


def save_csv(rows: list[dict[str, Any]], filepath: str) -> None:
    """Lưu danh sách bản ghi ra tệp CSV."""
    if not rows:
        return
    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Chạy đánh giá mốc Random Policy và in bảng tổng hợp."""
    args = parse_args()
    scenarios = [args.scenario] if args.scenario else ALL_SCENARIOS
    seeds = config.SEEDS

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for sc_id in scenarios:
        sc_results: list[dict[str, Any]] = []
        for seed in seeds:
            res = evaluate_random_policy_seed(
                map_name=args.map,
                scenario_id=sc_id,
                seed=seed,
                eval_episodes=args.eval_episodes,
            )
            sc_results.append(res)
            detail_rows.append(res)

        summary_rows.append(
            {
                "scenario_id": sc_id,
                "num_seeds": len(seeds),
                "sr_mean": float(np.mean([m["success_rate"] for m in sc_results])),
                "sr_std": float(np.std([m["success_rate"] for m in sc_results])),
                "steps_mean": float(np.mean([m["mean_success_steps"] for m in sc_results])),
                "steps_std": float(np.std([m["mean_success_steps"] for m in sc_results])),
                "all_steps_mean": float(np.mean([m["mean_all_steps"] for m in sc_results])),
                "reward_mean": float(np.mean([m["mean_reward"] for m in sc_results])),
                "reward_std": float(np.std([m["mean_reward"] for m in sc_results])),
                "coll_mean": float(np.mean([m["mean_collisions"] for m in sc_results])),
                "coll_std": float(np.std([m["mean_collisions"] for m in sc_results])),
                "time_mean": 0.0,
                "time_std": 0.0,
            }
        )

    print("\n" + "=" * 104)
    print(f"BẢNG MỐC THAM CHIẾU CƠ SỞ RANDOM POLICY ({args.map} | {len(seeds)} seeds x {args.eval_episodes} eps)")
    print("=" * 104)
    print(
        f"{'Kịch bản':<9} | {'Success Rate (%)':<18} | {'Success Steps':<16} | "
        f"{'Avg Reward':<20} | {'Collisions':<15} | {'Train Time (s)':<14}"
    )
    print("-" * 104)
    for row in summary_rows:
        print(
            f"{row['scenario_id']:<9} | "
            f"{row['sr_mean']:6.2f} +/- {row['sr_std']:<7.2f} | "
            f"{row['steps_mean']:6.2f} +/- {row['steps_std']:<5.2f} | "
            f"{row['reward_mean']:8.2f} +/- {row['reward_std']:<7.2f} | "
            f"{row['coll_mean']:5.2f} +/- {row['coll_std']:<5.2f} | "
            f"{row['time_mean']:6.3f} +/- {row['time_std']:<5.3f}"
        )
    print("=" * 104)

    summary_path = os.path.join(args.log_dir, "summary_random_policy.csv")
    detail_path = os.path.join(args.log_dir, "detail_random_policy_seeds.csv")
    save_csv(summary_rows, summary_path)
    save_csv(detail_rows, detail_path)
    print(f"-> Đã lưu bảng tổng hợp Random Policy vào: {os.path.abspath(summary_path)}")

    if args.render:
        import time
        import pygame

        demo_sc = scenarios[0]
        demo_seed = seeds[0]
        env = EVChargingEnv(map_name=args.map, obs_type="tuple", render_mode="human")
        agent = RandomAgent(action_space=int(env.action_space.n), seed=demo_seed)
        print(f"\n[PYGAME DEMO] Trình diễn Random Policy trên {demo_sc} ({args.demo_episodes} tập, nhấn Esc để đóng)...")

        stop_requested = False
        try:
            for ep in range(1, args.demo_episodes + 1):
                if stop_requested:
                    break
                s, _ = env.reset(seed=demo_seed * 1000 + ep, options={"scenario": demo_sc})
                pygame.display.set_caption(f"Random Policy Demo | {demo_sc} | Tập {ep}/{args.demo_episodes}")
                terminated = False
                truncated = False
                last_step = time.perf_counter()

                while not (terminated or truncated) and not stop_requested:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                        ):
                            stop_requested = True
                            break
                    if time.perf_counter() - last_step >= args.step_delay:
                        a = agent.select_action(s, eval_mode=True)
                        s, _, terminated, truncated, _ = env.step(a)
                        last_step = time.perf_counter()
                    env.render()

                end_wait = time.perf_counter()
                while time.perf_counter() - end_wait < 1.0 and not stop_requested:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                        ):
                            stop_requested = True
                            break
                    env.render()
        finally:
            env.close()


if __name__ == "__main__":
    main()
