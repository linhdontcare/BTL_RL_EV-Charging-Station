"""Script huấn luyện và đánh giá thuật toán SARSA cho bài toán EV Charging Station.

Tính năng:
- Cài đặt chuẩn vòng lặp SARSA On-policy TD Control:
    1. Chọn trước a tại s bằng epsilon-greedy.
    2. Trong vòng lặp mỗi bước:
       - Thực thi s_next, r, terminated, truncated, info = env.step(a)
       - Chọn trước a_next tại s_next bằng epsilon-greedy
       - Gọi agent.update(s, a, r, s_next, a_next, terminated)
       - Gán s, a = s_next, a_next
    3. Giảm epsilon sau mỗi episode.
- Huấn luyện mặc định 3000 episodes với render_mode=None.
- Hỗ trợ chạy theo seed đơn lẻ (mặc định 42) hoặc bộ 5 seeds chuẩn [42, 100, 2026, 7, 99].
- Hỗ trợ chọn kịch bản bất kỳ từ C01 đến C08.
- Ghi log đầy đủ các chỉ số: tổng reward, số bước, collision_attempts, is_success,
  thời gian huấn luyện (time.perf_counter()).
- Tự động chạy đánh giá 100 episodes với eval_mode=True sau khi huấn luyện xong.
- Lưu Q-table ra tệp .npy và lưu lịch sử huấn luyện ra tệp .csv trong thư mục logs/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any

# Đảm bảo import được module từ thư mục gốc
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from agents.sarsa import SarsaAgent
import config
from env.ev_charging_env import EVChargingEnv


def parse_args() -> argparse.Namespace:
    """Xử lý các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá thuật toán SARSA trên môi trường EVChargingEnv."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="C01",
        choices=[f"C0{i}" for i in range(1, 9)],
        help="Kịch bản huấn luyện và đánh giá (C01 đến C08, mặc định: C01).",
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
        help="Chạy tuần tự trên toàn bộ 5 seeds chuẩn [42, 100, 2026, 7, 99] và tổng hợp thống kê.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="Số episodes huấn luyện (mặc định: 3000).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Số episodes đánh giá sau huấn luyện (mặc định: 100).",
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
        default=0.95,
        help="Hệ số chiết khấu gamma (mặc định: 0.95).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Epsilon ban đầu (mặc định: 1.0).",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Epsilon tối thiểu (mặc định: 0.05).",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Hệ số suy giảm epsilon (mặc định: 0.995).",
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
        default=200,
        help="Khoảng cách episode in log tiến độ huấn luyện (mặc định: 200).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Thư mục lưu trữ Q-table (mặc định: models).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Thư mục lưu trữ tệp log CSV (mặc định: logs).",
    )
    return parser.parse_args()


def train_sarsa(
    env: EVChargingEnv,
    agent: SarsaAgent,
    episodes: int,
    scenario_id: str,
    seed: int,
    log_interval: int = 200,
) -> tuple[list[dict[str, Any]], float]:
    """Thực thi vòng lặp huấn luyện SARSA theo đúng đặc tả kỹ thuật.

    Tham số:
        env: Môi trường EVChargingEnv.
        agent: Tác tử SarsaAgent.
        episodes: Số lượng episodes cần train.
        scenario_id: Mã kịch bản (C01-C08).
        seed: Hạt giống ngẫu nhiên.
        log_interval: Tần suất in log tiến độ.

    Trả về:
        (training_history, total_train_time)
    """
    history: list[dict[str, Any]] = []
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN SARSA | Scenario: {scenario_id} | Seed: {seed} | Episodes: {episodes} ---")

    start_time = time.perf_counter()

    for ep in range(1, episodes + 1):
        # Thiết lập seed ở episode đầu tiên để đảm bảo tái lập
        if ep == 1:
            s, info = env.reset(seed=seed, options={"scenario": scenario_id})
        else:
            s, info = env.reset(options={"scenario": scenario_id})

        # 1. Chọn trước a tại s bằng chính sách epsilon-greedy
        a = agent.select_action(s, eval_mode=False)

        episode_reward = 0.0
        done = False

        while not done:
            # 2. Thực thi bước hành động trong môi trường
            s_next, r, terminated, truncated, info = env.step(a)
            episode_reward += r

            # 3. Chọn trước a_next tại s_next bằng chính sách epsilon-greedy
            a_next = agent.select_action(s_next, eval_mode=False)

            # 4. Cập nhật Q-table theo công thức SARSA On-policy
            agent.update(s, a, r, s_next, a_next, terminated, truncated)

            # 5. Chuyển tiếp trạng thái và hành động
            s, a = s_next, a_next
            done = terminated or truncated

        # 6. Suy giảm epsilon sau mỗi episode
        current_eps = agent.epsilon
        agent.decay_epsilon()

        # Lưu lại bản ghi episode
        history.append({
            "episode": ep,
            "reward": episode_reward,
            "step_count": info["step_count"],
            "collision_attempts": info["collision_attempts"],
            "is_success": 1 if info["is_success"] else 0,
            "charged_type": info["charged_type"] or "none",
            "epsilon": current_eps,
        })

        # In log tiến độ định kỳ
        if ep % log_interval == 0 or ep == 1 or ep == episodes:
            window = history[-min(len(history), 50):]
            avg_rew = np.mean([h["reward"] for h in window])
            avg_steps = np.mean([h["step_count"] for h in window])
            avg_cols = np.mean([h["collision_attempts"] for h in window])
            sr = np.mean([h["is_success"] for h in window]) * 100.0
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
    print(f"-> Hoàn thành huấn luyện trong {total_train_time:.3f} giây. Số trạng thái trong Q-table: {len(agent.q_table)}")
    return history, total_train_time


def evaluate_sarsa(
    env: EVChargingEnv,
    agent: SarsaAgent,
    eval_episodes: int,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    """Đánh giá tác tử SARSA với chính sách tham lam tuyệt đối (eval_mode=True).

    Tham số:
        env: Môi trường EVChargingEnv.
        agent: Tác tử SarsaAgent.
        eval_episodes: Số episodes đánh giá (mặc định 100).
        scenario_id: Mã kịch bản.
        seed: Hạt giống ngẫu nhiên.

    Trả về:
        Từ điển chứa các chỉ số thống kê tổng hợp.
    """
    eval_start = time.perf_counter()
    rewards: list[float] = []
    steps: list[int] = []
    collisions: list[int] = []
    successes: list[int] = []

    for ep in range(1, eval_episodes + 1):
        # Đảm bảo seed độc lập và có thể tái lập cho giai đoạn đánh giá
        if ep == 1:
            s, info = env.reset(seed=seed + 10000, options={"scenario": scenario_id})
        else:
            s, info = env.reset(options={"scenario": scenario_id})

        episode_reward = 0.0
        done = False

        while not done:
            # Chọn hành động tham lam tuyệt đối (eval_mode=True)
            a = agent.select_action(s, eval_mode=True)
            s_next, r, terminated, truncated, info = env.step(a)
            episode_reward += r
            s = s_next
            done = terminated or truncated

        rewards.append(episode_reward)
        steps.append(info["step_count"])
        collisions.append(info["collision_attempts"])
        successes.append(1 if info["is_success"] else 0)

    eval_time = time.perf_counter() - eval_start

    metrics = {
        "scenario_id": scenario_id,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "success_rate": float(np.mean(successes) * 100.0),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_collisions": float(np.mean(collisions)),
        "eval_time": float(eval_time),
    }

    print(
        f"\n[KẾT QUẢ ĐÁNH GIÁ {eval_episodes} EPISODES - {scenario_id} - SEED {seed}]\n"
        f"  * Tỷ lệ thành công (Success Rate) : {metrics['success_rate']:5.1f}%\n"
        f"  * Phần thưởng trung bình (Reward)  : {metrics['mean_reward']:6.2f} +/- {metrics['std_reward']:5.2f} "
        f"[Min: {metrics['min_reward']:.1f}, Max: {metrics['max_reward']:.1f}]\n"
        f"  * Số bước trung bình (Steps)     : {metrics['mean_steps']:5.2f} +/- {metrics['std_steps']:4.2f}\n"
        f"  * Số lần va chạm TB (Collisions) : {metrics['mean_collisions']:5.2f}\n"
        f"  * Thời gian đánh giá             : {metrics['eval_time']:5.3f}s"
    )
    return metrics


def save_history_to_csv(history: list[dict[str, Any]], filepath: str) -> None:
    """Lưu lịch sử huấn luyện ra tệp CSV."""
    if not history:
        return
    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    keys = list(history[0].keys())
    with open(abs_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"-> Đã lưu lịch sử huấn luyện vào: {abs_path}")


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
) -> dict[str, Any]:
    """Chạy trọn gói huấn luyện và đánh giá trên 1 seed cụ thể."""
    # Khởi tạo môi trường với obs_type='tuple' và render_mode=None để tối ưu hiệu năng
    env = EVChargingEnv(map_name=map_name, obs_type="tuple", render_mode=None)

    # Khởi tạo tác tử SARSA
    agent = SarsaAgent(
        action_space=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    # 1. Huấn luyện
    history, train_time = train_sarsa(
        env=env,
        agent=agent,
        episodes=episodes,
        scenario_id=scenario_id,
        seed=seed,
        log_interval=log_interval,
    )

    # 2. Đánh giá 100 episodes
    eval_metrics = evaluate_sarsa(
        env=env,
        agent=agent,
        eval_episodes=eval_episodes,
        scenario_id=scenario_id,
        seed=seed,
    )
    eval_metrics["train_time"] = train_time

    # 3. Lưu model Q-table
    model_filename = f"sarsa_table_{scenario_id}_seed{seed}.npy"
    model_path = os.path.join(save_dir, model_filename)
    agent.save(model_path)
    print(f"-> Đã lưu Q-table vào: {model_path}")

    if save_default_model:
        default_model_path = os.path.join(save_dir, "sarsa_table.npy")
        agent.save(default_model_path)
        print(f"-> Đã lưu Q-table mặc định vào: {default_model_path}")

    # 4. Lưu log huấn luyện CSV
    csv_filename = f"train_sarsa_{scenario_id}_seed{seed}.csv"
    csv_path = os.path.join(log_dir, csv_filename)
    save_history_to_csv(history, csv_path)

    env.close()
    return eval_metrics


def main() -> None:
    """Hàm thực thi chính."""
    args = parse_args()

    print("=" * 70)
    print("CHƯƠNG TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ SARSA CHO EV CHARGING")
    print("=" * 70)
    print(f"Kịch bản        : {args.scenario}")
    print(f"Bản đồ lưới     : {args.map}")
    print(f"Số episodes     : {args.episodes}")
    print(f"Số eps đánh giá : {args.eval_episodes}")
    print(f"Siêu tham số    : alpha={args.alpha}, gamma={args.gamma}")
    print(f"Epsilon         : init={args.epsilon}, min={args.epsilon_min}, decay={args.epsilon_decay}")

    seeds_to_run = config.SEEDS if args.all_seeds else [args.seed]
    print(f"Danh sách seeds : {seeds_to_run}")
    print("=" * 70)

    all_metrics: list[dict[str, Any]] = []

    for idx, seed in enumerate(seeds_to_run):
        save_default = (len(seeds_to_run) == 1) or (idx == 0)
        metrics = run_experiment_single_seed(
            scenario_id=args.scenario,
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
        )
        all_metrics.append(metrics)

    # Nếu chạy nhiều seeds, in bảng tổng hợp thống kê
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print(f"BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ TRÊN {len(all_metrics)} SEEDS ({args.scenario})")
        print("=" * 80)
        print(f"{'Seed':<8} | {'Success Rate (%)':<18} | {'Mean Reward':<16} | {'Mean Steps':<12} | {'Mean Cols':<11} | {'Train Time':<10}")
        print("-" * 80)
        for m in all_metrics:
            print(
                f"{m['seed']:<8} | "
                f"{m['success_rate']:<18.1f} | "
                f"{m['mean_reward']:>7.2f} +/- {m['std_reward']:<5.2f} | "
                f"{m['mean_steps']:<12.2f} | "
                f"{m['mean_collisions']:<11.2f} | "
                f"{m['train_time']:>8.2f}s"
            )
        print("-" * 80)

        # Thống kê trung bình và độ lệch chuẩn gộp
        avg_sr = np.mean([m["success_rate"] for m in all_metrics])
        std_sr = np.std([m["success_rate"] for m in all_metrics])
        avg_rew = np.mean([m["mean_reward"] for m in all_metrics])
        std_rew = np.std([m["mean_reward"] for m in all_metrics])
        avg_stp = np.mean([m["mean_steps"] for m in all_metrics])
        std_stp = np.std([m["mean_steps"] for m in all_metrics])
        avg_col = np.mean([m["mean_collisions"] for m in all_metrics])
        std_col = np.std([m["mean_collisions"] for m in all_metrics])
        avg_time = np.mean([m["train_time"] for m in all_metrics])

        print(
            f"{'TRUNG BÌNH':<8} | "
            f"{avg_sr:>5.1f} +/- {std_sr:<10.1f} | "
            f"{avg_rew:>7.2f} +/- {std_rew:<5.2f} | "
            f"{avg_stp:<12.2f} | "
            f"{avg_col:<11.2f} | "
            f"{avg_time:>8.2f}s"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
