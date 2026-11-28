import argparse
import os
import sys
import time
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pygame

from agents.dqn_pytorch import DQNAgent
from agents.random_agent import RandomAgent
import config
from env.ev_charging_env import EVChargingEnv

ALL_SCENARIOS = [f"C0{i}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo trực quan DQN & Random Policy trên giao diện Pygame.")
    parser.add_argument("--scenario", type=str, default="C03", choices=ALL_SCENARIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "random"])
    parser.add_argument("--map", type=str, default="default_10x10", choices=list(config.MAP_CONFIGS.keys()))
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--models-dir", type=str, default="models")
    return parser.parse_args()


def load_or_train_dqn_agent(map_name: str, scenario_id: str, seed: int, models_dir: str) -> DQNAgent:
    agent = DQNAgent(action_dim=config.NUM_ACTIONS, seed=seed)
    specific_path = os.path.join(models_dir, f"dqn_weights_{scenario_id}_seed{seed}.pth")
    default_path = os.path.join(models_dir, "dqn_weights.pth")

    if map_name == "default_10x10" and os.path.isfile(specific_path):
        agent.load(specific_path)
        print(f"[MODEL] Đã nạp trọng số DQN: {specific_path}")
        return agent

    if map_name == "default_10x10" and os.path.isfile(default_path):
        agent.load(default_path)
        print(f"[MODEL] Đã nạp trọng số DQN mặc định: {default_path}")
        return agent

    print(
        f"[WARNING] Không tìm thấy file trọng số DQN (.pth) cho kịch bản {scenario_id} ({map_name}). "
        f"Vui lòng chạy lệnh: python scripts/train_dqn.py --scenario {scenario_id} trước."
    )
    sys.exit(1)


def main() -> None:
    args = parse_args()
    current_scenario = args.scenario.upper()
    agent_mode = args.agent
    step_delay = max(0.05, args.delay)
    episode_idx = 0

    env = EVChargingEnv(map_name=args.map, obs_type="box", render_mode="human")
    dqn_agent = load_or_train_dqn_agent(args.map, current_scenario, args.seed, args.models_dir)
    random_agent = RandomAgent(action_space=config.NUM_ACTIONS, seed=args.seed)

    def reset_episode(reload_model: bool = False) -> Any:
        nonlocal dqn_agent, episode_idx
        episode_idx += 1
        if reload_model:
            dqn_agent = load_or_train_dqn_agent(args.map, current_scenario, args.seed, args.models_dir)
        obs, _ = env.reset(seed=args.seed * 1000 + episode_idx, options={"scenario": current_scenario})
        algo_label = "DQN" if agent_mode == "dqn" else "Random Policy"
        pygame.display.set_caption(f"EV Charging Demo | {algo_label} | Kịch bản {current_scenario} | Delay {step_delay:.2f}s")
        return obs

    scenario_keys = {
        pygame.K_1: "C01", pygame.K_2: "C02", pygame.K_3: "C03", pygame.K_4: "C04",
        pygame.K_5: "C05", pygame.K_6: "C06", pygame.K_7: "C07", pygame.K_8: "C08",
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
                    elif event.key == pygame.K_SPACE:
                        if env.terminated or env.truncated:
                            obs = reset_episode(reload_model=False)
                            paused = False
                        else:
                            paused = not paused
                    elif event.key == pygame.K_TAB:
                        agent_mode = "random" if agent_mode == "dqn" else "dqn"
                        obs = reset_episode(reload_model=False)
                        paused = False
                    elif event.key in scenario_keys:
                        current_scenario = scenario_keys[event.key]
                        obs = reset_episode(reload_model=True)
                        paused = False
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        step_delay = max(0.05, step_delay - 0.05)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        step_delay = min(1.50, step_delay + 0.05)

            now = time.perf_counter()
            if not paused and not (env.terminated or env.truncated) and (now - last_step_time >= step_delay):
                active_agent = dqn_agent if agent_mode == "dqn" else random_agent
                action = active_agent.select_action(obs, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                last_step_time = now
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
