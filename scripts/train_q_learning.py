"""Script huấn luyện và đánh giá thuật toán Q-Learning cho bài toán EV Charging Station.

Bám sát Thuật toán 1 (Mục 3.2), Bảng 10 (5 chỉ số đánh giá), Bảng 12 (siêu tham số)
và Thuật toán 4 (Pipeline đánh giá đa hạt giống) trong báo cáo:
- Vòng lặp huấn luyện Off-policy TD Control:
    1. Khởi tạo s, info = env.reset(...)
    2. Trong mỗi bước của episode:
       - Chọn hành động a = agent.select_action(s, eval_mode=False)
       - Thực thi s_next, r, terminated, truncated, info = env.step(a)
       - Cập nhật Off-policy: agent.update(s, a, r, s_next, terminated=terminated, truncated=truncated)
       - Chuyển trạng thái: s = s_next
    3. Suy giảm epsilon sau mỗi episode.
- Huấn luyện mặc định 3.000 episodes ở chế độ render_mode=None.
- Hỗ trợ chạy 1 kịch bản hoặc toàn bộ 8 kịch bản C01-C08, 1 seed hoặc 5 seeds chuẩn.
- Ghi log CSV đồng nhất định dạng với SARSA, lưu Q-table .npy và xuất bảng Mean +/- Std.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any

# Đảm bảo import được các module từ thư mục gốc dự án
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from agents.q_learning import QLearningAgent
import config
from env.ev_charging_env import EVChargingEnv

ALL_SCENARIOS: list[str] = [f"C0{i}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    """Xử lý các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá thuật toán Q-Learning trên EVChargingEnv."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="C01",
        choices=ALL_SCENARIOS,
        help="Kịch bản huấn luyện và đánh giá (C01 đến C08, mặc định: C01).",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Chạy tuần tự trên toàn bộ 8 kịch bản chuẩn C01-C08.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed ngẫu nhiên cho một lần chạy (mặc định: 42).",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Chạy trên toàn bộ 5 seeds chuẩn [42, 100, 2026, 7, 99] và tổng hợp Mean +/- Std.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="Số episodes huấn luyện mỗi seed (mặc định: 3000).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Số episodes đánh giá với chính sách tham lam (mặc định: 100).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Tốc độ học alpha (mặc định: 0.10).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config.GAMMA,
        help="Hệ số chiết khấu gamma (mặc định: 0.95).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Tỷ lệ thăm dò ban đầu (mặc định: 1.0).",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Tỷ lệ thăm dò tối thiểu (mặc định: 0.05).",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Hệ số suy giảm epsilon sau mỗi episode (mặc định: 0.995).",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="default_10x10",
        choices=list(config.MAP_CONFIGS.keys()),
        help="Bản đồ mô phỏng ('default_10x10' hoặc 'extended_15x15').",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Tần suất in log tiến độ huấn luyện (mặc định: 500, 0 để tắt log định kỳ).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Thư mục lưu Q-table .npy (mặc định: models).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Thư mục lưu log CSV (mặc định: logs).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Chỉ in bảng tổng hợp, ẩn log chi tiết từng seed.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Mở cửa sổ Pygame trực quan hóa xe điện sau khi huấn luyện xong.",
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
        default=0.35,
        help="Độ trễ (giây) giữa mỗi bước khi hiển thị Pygame (mặc định: 0.35).",
    )
    return parser.parse_args()


def train_q_learning(
    env: EVChargingEnv,
    agent: QLearningAgent,
    episodes: int,
    scenario_id: str,
    seed: int,
    log_interval: int = 500,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Thực thi vòng lặp huấn luyện Q-Learning theo Thuật toán 1."""
    history: list[dict[str, Any]] = []
    if not quiet:
        print(
            f"\n--- HUẤN LUYỆN Q-LEARNING | Scenario: {scenario_id} | "
            f"Seed: {seed} | Episodes: {episodes} ---"
        )

    start_time = time.perf_counter()

    for ep in range(1, episodes + 1):
        if ep == 1:
            s, info = env.reset(seed=seed, options={"scenario": scenario_id})
        else:
            s, info = env.reset(options={"scenario": scenario_id})

        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            a = agent.select_action(s, eval_mode=False)
            s_next, r, terminated, truncated, info = env.step(a)
            episode_reward += r

            agent.update(
                s=s,
                a=a,
                r=r,
                s_next=s_next,
                terminated=terminated,
                truncated=truncated,
            )
            s = s_next

        current_eps = agent.epsilon
        agent.decay_epsilon()

        history.append(
            {
                "episode": ep,
                "reward": episode_reward,
                "step_count": info["step_count"],
                "collision_attempts": info["collision_attempts"],
                "is_success": 1 if info["is_success"] else 0,
                "charged_type": info["charged_type"] or "none",
                "epsilon": current_eps,
            }
        )

        if not quiet and log_interval > 0 and (ep % log_interval == 0 or ep == 1 or ep == episodes):
            window = history[-min(len(history), 50) :]
            avg_rew = float(np.mean([h["reward"] for h in window]))
            avg_steps = float(np.mean([h["step_count"] for h in window]))
            avg_cols = float(np.mean([h["collision_attempts"] for h in window]))
            sr = float(np.mean([h["is_success"] for h in window]) * 100.0)
            elapsed = time.perf_counter() - start_time
            print(
                f"Ep {ep:4d}/{episodes} | "
                f"Reward: {episode_reward:7.1f} (avg50: {avg_rew:6.1f}) | "
                f"Steps: {info['step_count']:2d} (avg50: {avg_steps:4.1f}) | "
                f"Cols: {info['collision_attempts']:2d} (avg50: {avg_cols:3.1f}) | "
                f"Success: {sr:5.1f}% | "
                f"Eps: {current_eps:5.3f} | "
                f"Time: {elapsed:5.2f}s"
            )

    total_train_time = time.perf_counter() - start_time
    if not quiet:
        print(
            f"-> Hoàn thành huấn luyện trong {total_train_time:.3f}s. "
            f"Số trạng thái trong Q-table: {len(agent.q_table)}"
        )
    return history, total_train_time


def evaluate_q_learning(
    env: EVChargingEnv,
    agent: QLearningAgent,
    eval_episodes: int,
    scenario_id: str,
    seed: int,
    quiet: bool = False,
) -> dict[str, Any]:
    """Đánh giá tác tử Q-Learning theo chính sách tham lam tất định (Thuật toán 4)."""
    eval_start = time.perf_counter()
    rewards: list[float] = []
    all_steps: list[int] = []
    success_steps: list[int] = []
    collisions: list[int] = []
    successes: list[int] = []
    fast_charges = 0
    normal_charges = 0

    for ep in range(1, eval_episodes + 1):
        # Đúng quy ước hạt giống đánh giá trong Thuật toán 4: seed * 1000 + ep
        s, info = env.reset(seed=seed * 1000 + ep, options={"scenario": scenario_id})
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            a = agent.select_action(s, eval_mode=True)
            s_next, r, terminated, truncated, info = env.step(a)
            episode_reward += r
            s = s_next

        rewards.append(episode_reward)
        all_steps.append(info["step_count"])
        collisions.append(info["collision_attempts"])
        is_succ = bool(info["is_success"])
        successes.append(1 if is_succ else 0)

        if is_succ:
            success_steps.append(info["step_count"])
            if info["charged_type"] == "fast":
                fast_charges += 1
            elif info["charged_type"] == "normal":
                normal_charges += 1

    eval_time = time.perf_counter() - eval_start
    n_succ = len(success_steps)
    # Đúng công thức Bảng 10 & Thuật toán 4: sum_succ_steps / max(1, n_succ)
    avg_success_steps = float(sum(success_steps) / max(1, n_succ))

    metrics = {
        "scenario_id": scenario_id,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_success_steps": avg_success_steps,
        "mean_all_steps": float(np.mean(all_steps)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_collisions": float(np.mean(collisions)),
        "fast_charge_rate": float(fast_charges / eval_episodes * 100.0),
        "normal_charge_rate": float(normal_charges / eval_episodes * 100.0),
        "eval_time": float(eval_time),
    }

    if not quiet:
        print(
            f"\n[ĐÁNH GIÁ {eval_episodes} EPISODES - {scenario_id} - SEED {seed}]\n"
            f"  * Tỷ lệ thành công (SR)          : {metrics['success_rate']:5.1f}% "
            f"(Fast: {metrics['fast_charge_rate']:.1f}%, Normal: {metrics['normal_charge_rate']:.1f}%)\n"
            f"  * Số bước TB thành công (Steps)  : {metrics['mean_success_steps']:5.2f} "
            f"(Tất cả eps: {metrics['mean_all_steps']:5.2f})\n"
            f"  * Phần thưởng TB (Reward)        : {metrics['mean_reward']:6.2f} +/- {metrics['std_reward']:5.2f} "
            f"[Min: {metrics['min_reward']:.1f}, Max: {metrics['max_reward']:.1f}]\n"
            f"  * Số lần va chạm TB (Collisions) : {metrics['mean_collisions']:5.2f}\n"
            f"  * Thời gian đánh giá             : {metrics['eval_time']:5.3f}s"
        )
    return metrics


def save_rows_to_csv(rows: list[dict[str, Any]], filepath: str) -> None:
    """Lưu danh sách từ điển ra tệp CSV."""
    if not rows:
        return
    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(abs_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment_single_seed(
    scenario_id: str,
    seed: int,
    episodes: int,
    eval_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    map_name: str,
    log_interval: int,
    save_dir: str,
    log_dir: str,
    save_default_model: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Chạy trọn gói huấn luyện và đánh giá Q-Learning trên 1 kịch bản và 1 seed."""
    env = EVChargingEnv(map_name=map_name, obs_type="tuple", render_mode=None)

    agent = QLearningAgent(
        action_space=int(env.action_space.n),
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    history, train_time = train_q_learning(
        env=env,
        agent=agent,
        episodes=episodes,
        scenario_id=scenario_id,
        seed=seed,
        log_interval=log_interval,
        quiet=quiet,
    )

    eval_metrics = evaluate_q_learning(
        env=env,
        agent=agent,
        eval_episodes=eval_episodes,
        scenario_id=scenario_id,
        seed=seed,
        quiet=quiet,
    )
    eval_metrics["train_time"] = train_time
    eval_metrics["q_table_states"] = len(agent.q_table)

    # Lưu Q-table theo kịch bản và seed
    model_path = os.path.join(save_dir, f"q_learning_table_{scenario_id}_seed{seed}.npy")
    agent.save(model_path)

    # Lưu thêm tệp mặc định models/q_learning_table.npy theo báo cáo
    if save_default_model:
        default_model_path = os.path.join(save_dir, "q_learning_table.npy")
        agent.save(default_model_path)

    # Lưu lịch sử huấn luyện CSV (cùng định dạng với SARSA)
    csv_path = os.path.join(log_dir, f"train_q_learning_{scenario_id}_seed{seed}.csv")
    save_rows_to_csv(history, csv_path)

    env.close()
    return eval_metrics


def demo_agent_on_pygame(
    map_name: str,
    model_path: str,
    scenario_id: str,
    seed: int,
    demo_episodes: int = 3,
    step_delay: float = 0.35,
) -> None:
    """Hiển thị trực quan tác tử Q-Learning đã huấn luyện trên cửa sổ Pygame."""
    import pygame

    agent = QLearningAgent(action_space=config.NUM_ACTIONS, seed=seed)
    agent.load(model_path)

    env = EVChargingEnv(map_name=map_name, obs_type="tuple", render_mode="human")
    print(f"\n[PYGAME DEMO] Trình diễn Q-Learning trên {scenario_id} ({demo_episodes} tập, nhấn Esc để đóng)...")

    stop_requested = False
    try:
        for ep in range(1, demo_episodes + 1):
            if stop_requested:
                break
            s, info = env.reset(seed=seed * 1000 + ep, options={"scenario": scenario_id})
            pygame.display.set_caption(f"Q-Learning Demo | {scenario_id} | Tập {ep}/{demo_episodes}")
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

                if time.perf_counter() - last_step >= step_delay:
                    a = agent.select_action(s, eval_mode=True)
                    s, _, terminated, truncated, info = env.step(a)
                    last_step = time.perf_counter()
                env.render()

            # Dừng nhẹ 1.2 giây ở cuối tập để quan sát vị trí sạc và vệt đường đi
            end_wait = time.perf_counter()
            while time.perf_counter() - end_wait < 1.2 and not stop_requested:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        stop_requested = True
                        break
                env.render()
    finally:
        env.close()


def aggregate_scenario_metrics(scenario_id: str, seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp Mean +/- Std qua các seeds cho một kịch bản."""
    return {
        "scenario_id": scenario_id,
        "num_seeds": len(seed_metrics),
        "sr_mean": float(np.mean([m["success_rate"] for m in seed_metrics])),
        "sr_std": float(np.std([m["success_rate"] for m in seed_metrics])),
        "steps_mean": float(np.mean([m["mean_success_steps"] for m in seed_metrics])),
        "steps_std": float(np.std([m["mean_success_steps"] for m in seed_metrics])),
        "all_steps_mean": float(np.mean([m["mean_all_steps"] for m in seed_metrics])),
        "reward_mean": float(np.mean([m["mean_reward"] for m in seed_metrics])),
        "reward_std": float(np.std([m["mean_reward"] for m in seed_metrics])),
        "coll_mean": float(np.mean([m["mean_collisions"] for m in seed_metrics])),
        "coll_std": float(np.std([m["mean_collisions"] for m in seed_metrics])),
        "time_mean": float(np.mean([m["train_time"] for m in seed_metrics])),
        "time_std": float(np.std([m["train_time"] for m in seed_metrics])),
        "fast_rate_mean": float(np.mean([m["fast_charge_rate"] for m in seed_metrics])),
        "normal_rate_mean": float(np.mean([m["normal_charge_rate"] for m in seed_metrics])),
    }


def print_summary_table(summary_rows: list[dict[str, Any]], seeds: list[int]) -> None:
    """In bảng tổng hợp 5 chỉ số chuẩn hóa (Mean +/- Std)."""
    print("\n" + "=" * 116)
    print(f"BẢNG TỔNG HỢP Q-LEARNING (Mean +/- Std qua {len(seeds)} seeds: {seeds})")
    print("=" * 116)
    print(
        f"{'Kịch bản':<9} | {'Success Rate (%)':<18} | {'Success Steps':<16} | "
        f"{'Avg Reward':<20} | {'Collisions':<15} | {'Train Time (s)':<16} | {'Fast/Norm (%)':<13}"
    )
    print("-" * 116)
    for row in summary_rows:
        print(
            f"{row['scenario_id']:<9} | "
            f"{row['sr_mean']:6.2f} +/- {row['sr_std']:<7.2f} | "
            f"{row['steps_mean']:6.2f} +/- {row['steps_std']:<5.2f} | "
            f"{row['reward_mean']:8.2f} +/- {row['reward_std']:<7.2f} | "
            f"{row['coll_mean']:5.2f} +/- {row['coll_std']:<5.2f} | "
            f"{row['time_mean']:6.3f} +/- {row['time_std']:<5.3f} | "
            f"{row['fast_rate_mean']:5.1f}/{row['normal_rate_mean']:<5.1f}"
        )
    print("=" * 116)


def main() -> None:
    """Điểm chạy chính của chương trình."""
    args = parse_args()

    scenarios_to_run = ALL_SCENARIOS if args.all_scenarios else [args.scenario]
    seeds_to_run = config.SEEDS if args.all_seeds else [args.seed]

    print("=" * 75)
    print("CHƯƠNG TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ Q-LEARNING CHO EV CHARGING")
    print("=" * 75)
    print(f"Kịch bản        : {scenarios_to_run}")
    print(f"Bản đồ lưới     : {args.map}")
    print(f"Số episodes     : {args.episodes} (đánh giá: {args.eval_episodes})")
    print(f"Siêu tham số    : alpha={args.alpha}, gamma={args.gamma}")
    print(f"Epsilon         : init={args.epsilon}, min={args.epsilon_min}, decay={args.epsilon_decay}")
    print(f"Danh sách seeds : {seeds_to_run}")
    print("=" * 75)

    summary_rows: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []

    for sc_id in scenarios_to_run:
        sc_metrics: list[dict[str, Any]] = []
        for idx, seed in enumerate(seeds_to_run):
            # Lưu q_learning_table.npy cho C06 seed đầu tiên (tổng quát nhất) hoặc cấu hình đơn lẻ
            save_default = (
                (len(scenarios_to_run) == 1 and idx == 0)
                or (sc_id == "C06" and idx == 0)
            )
            metrics = run_experiment_single_seed(
                scenario_id=sc_id,
                seed=seed,
                episodes=args.episodes,
                eval_episodes=args.eval_episodes,
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon=args.epsilon,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                map_name=args.map,
                log_interval=args.log_interval,
                save_dir=args.save_dir,
                log_dir=args.log_dir,
                save_default_model=save_default,
                quiet=args.quiet,
            )
            sc_metrics.append(metrics)
            all_seed_rows.append(metrics)

        summary_rows.append(aggregate_scenario_metrics(sc_id, sc_metrics))

    print_summary_table(summary_rows, seeds_to_run)

    # Lưu bảng tổng hợp CSV khi chạy nhiều kịch bản hoặc nhiều seeds
    if len(scenarios_to_run) > 1 or len(seeds_to_run) > 1:
        summary_csv = os.path.join(args.log_dir, "summary_q_learning.csv")
        detail_csv = os.path.join(args.log_dir, "detail_q_learning_seeds.csv")
        save_rows_to_csv(summary_rows, summary_csv)
        save_rows_to_csv(all_seed_rows, detail_csv)
        print(f"-> Đã lưu bảng tổng hợp vào: {os.path.abspath(summary_csv)}")
        print(f"-> Đã lưu chi tiết từng seed vào: {os.path.abspath(detail_csv)}")

    # Nếu bật --render, mở cửa sổ Pygame trực quan hóa tác tử đã huấn luyện
    if args.render:
        demo_sc = scenarios_to_run[0]
        demo_seed = seeds_to_run[0]
        model_path = os.path.join(args.save_dir, f"q_learning_table_{demo_sc}_seed{demo_seed}.npy")
        demo_agent_on_pygame(
            map_name=args.map,
            model_path=model_path,
            scenario_id=demo_sc,
            seed=demo_seed,
            demo_episodes=args.demo_episodes,
            step_delay=args.step_delay,
        )


if __name__ == "__main__":
    main()
