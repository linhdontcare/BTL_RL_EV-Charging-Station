"""Kiểm thử đơn vị tự động cho QLearningAgent và RandomAgent."""

from __future__ import annotations

from collections import defaultdict
import os
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from agents.base_agent import BaseAgent
from agents.q_learning import QLearningAgent
from agents.random_agent import RandomAgent
from agents.sarsa import SarsaAgent
from config import GAMMA, NUM_ACTIONS, REWARD_FAST, REWARD_STEP


def test_q_learning_init() -> None:
    """Kiểm tra khởi tạo siêu tham số chuẩn theo Bảng 12."""
    print("-> Kiểm tra khởi tạo QLearningAgent...")
    agent = QLearningAgent()
    assert isinstance(agent, BaseAgent)
    assert agent.action_space == NUM_ACTIONS == 5
    assert agent.alpha == 0.10
    assert agent.gamma == GAMMA == 0.95
    assert agent.epsilon == 1.0
    assert agent.epsilon_min == 0.05
    assert agent.epsilon_decay == 0.995
    assert isinstance(agent.q_table, defaultdict)

    dummy_key = (0, 0, (0, 0, 0, 0))
    assert np.all(agent.q_table[dummy_key] == 0.0)
    assert agent.q_table[dummy_key].shape == (5,)
    print("   [OK] Khởi tạo và siêu tham số đúng chuẩn Bảng 12.")


def test_state_key_conversion() -> None:
    """Kiểm tra chuyển đổi state sang khóa tuple hashable."""
    print("-> Kiểm tra get_state_key...")
    key1 = QLearningAgent.get_state_key((2, 5, (0, 1, 1, 0)))
    assert key1 == (2, 5, (0, 1, 1, 0))
    assert isinstance(key1, tuple) and isinstance(key1[2], tuple)

    key2 = QLearningAgent.get_state_key((3, 7, [1, 0, 0, 1]))
    assert key2 == (3, 7, (1, 0, 0, 1))

    key3 = QLearningAgent.get_state_key(np.array([4, 5, 0, 1, 0, 1]))
    assert key3 == (4, 5, (0, 1, 0, 1))
    print("   [OK] get_state_key chuẩn xác và tương thích với EVChargingEnv.")


def test_select_action_and_tie_breaking() -> None:
    """Kiểm tra chọn hành động epsilon-greedy, greedy và random tie-breaking."""
    print("-> Kiểm tra select_action và tie-breaking...")
    agent = QLearningAgent(seed=42)
    state = (2, 4, (0, 1, 1, 1))

    # 1. Tất cả Q = 0 -> random tie-breaking trong eval_mode=True
    actions = [agent.select_action(state, eval_mode=True) for _ in range(50)]
    assert len(set(actions)) > 1, "Cần random tie-breaking khi các Q-value bằng nhau."

    # 2. Một giá trị max duy nhất
    key = QLearningAgent.get_state_key(state)
    agent.q_table[key][1] = 80.0
    for _ in range(20):
        assert agent.select_action(state, eval_mode=True) == 1

    # 3. Hai giá trị max bằng nhau (Action 1 và Action 3)
    agent.q_table[key][3] = 80.0
    tied = {agent.select_action(state, eval_mode=True) for _ in range(60)}
    assert tied == {1, 3}, f"Kỳ vọng tie-breaking giữa {{1, 3}}, nhận {tied}"

    # 4. Khi epsilon = 1.0 và eval_mode = False -> thăm dò toàn bộ 5 hành động
    agent.epsilon = 1.0
    explored = {agent.select_action(state, eval_mode=False) for _ in range(100)}
    assert explored == {0, 1, 2, 3, 4}
    print("   [OK] select_action (exploration/exploitation/tie-breaking) hoạt động chuẩn.")


def test_update_off_policy_formulas() -> None:
    """Kiểm tra công thức cập nhật Off-policy TD Control và khác biệt với SARSA."""
    print("-> Kiểm tra công thức update Off-policy TD (Q-Learning)...")
    q_agent = QLearningAgent(alpha=0.10, gamma=0.95, seed=42)
    sarsa_agent = SarsaAgent(alpha=0.10, gamma=0.95, seed=42)

    s = (2, 4, (0, 1, 1, 1))
    a = 1
    r = REWARD_STEP  # -5.0
    s_next = (2, 5, (0, 1, 1, 1))
    a_next_exploratory = 0  # Hành động thăm dò kém tại s_next

    s_key = QLearningAgent.get_state_key(s)
    s_next_key = QLearningAgent.get_state_key(s_next)

    # Tại s_next: hành động 4 (Charge) có Q = 100.0, hành động 0 có Q = -50.0
    for ag in (q_agent, sarsa_agent):
        ag.q_table[s_next_key] = np.array([-50.0, -5.0, -5.0, -5.0, 100.0])

    # 1. Non-terminated: Q-Learning phải dùng max_a' Q(s_next, a') = 100.0, bỏ qua a_next_exploratory = 0
    # y_target = -5.0 + 0.95 * 100.0 = 90.0 -> Q[s][a] = 0.0 + 0.10 * 90.0 = 9.0
    q_agent.update(s, a, r, s_next, a_next=a_next_exploratory, terminated=False, truncated=False)
    assert np.isclose(q_agent.q_table[s_key][a], 9.0), (
        f"Kỳ vọng 9.0, nhận {q_agent.q_table[s_key][a]}"
    )

    # Đối chiếu SARSA: dùng đúng Q(s_next, 0) = -50.0 -> y_target = -5.0 + 0.95 * (-50.0) = -52.5 -> -5.25
    sarsa_agent.update(s, a, r, s_next, a_next=a_next_exploratory, terminated=False, truncated=False)
    assert np.isclose(sarsa_agent.q_table[s_key][a], -5.25)
    assert q_agent.q_table[s_key][a] > sarsa_agent.q_table[s_key][a]

    # 2. Truncated == True, terminated == False: vẫn phải bootstrap theo max Q(s_next, .)
    # y_target = 90.0 -> Q mới = 9.0 + 0.10 * (90.0 - 9.0) = 17.1
    q_agent.update(s, a, r, s_next, terminated=False, truncated=True)
    assert np.isclose(q_agent.q_table[s_key][a], 17.1), (
        f"Kỳ vọng 17.1 khi truncated=True, nhận {q_agent.q_table[s_key][a]}"
    )

    # 3. Terminated == True: không bootstrap (y_target = r)
    term_s = (4, 5, (0, 0, 1, 1))
    term_key = QLearningAgent.get_state_key(term_s)
    q_agent.update(term_s, 4, REWARD_FAST, s_next, terminated=True, truncated=False)
    # y_target = 120.0 -> Q = 0.0 + 0.10 * 120.0 = 12.0
    assert np.isclose(q_agent.q_table[term_key][4], 12.0), (
        f"Kỳ vọng 12.0 khi terminated=True, nhận {q_agent.q_table[term_key][4]}"
    )
    print("   [OK] Công thức Off-policy TD (terminated vs truncated & max_a') chính xác 100%.")


def test_decay_epsilon() -> None:
    """Kiểm tra lịch suy giảm epsilon."""
    print("-> Kiểm tra decay_epsilon...")
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5)
    expected = [0.5, 0.25, 0.125, 0.0625, 0.05, 0.05]
    for exp_val in expected:
        agent.decay_epsilon()
        assert np.isclose(agent.epsilon, exp_val)
    print("   [OK] decay_epsilon dừng chính xác tại epsilon_min.")


def test_save_load() -> None:
    """Kiểm tra lưu và nạp Q-table định dạng .npy."""
    print("-> Kiểm tra save và load Q-table...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "q_learning_table.npy")
        agent1 = QLearningAgent(seed=42)
        key = (4, 5, (0, 0, 1, 1))
        agent1.q_table[key] = np.array([10.0, 20.0, 30.0, 40.0, 120.0])
        agent1.save(model_path)
        assert os.path.isfile(model_path)

        agent2 = QLearningAgent(seed=99)
        agent2.load(model_path)
        assert isinstance(agent2.q_table, defaultdict)
        assert np.allclose(agent2.q_table[key], [10.0, 20.0, 30.0, 40.0, 120.0])
        # Trạng thái mới chưa từng gặp vẫn trả về zeros(5)
        assert np.allclose(agent2.q_table[(8, 1, (1, 1, 1, 1))], np.zeros(5))
    print("   [OK] save và load hoạt động chính xác.")


def test_random_agent() -> None:
    """Kiểm tra mốc tham chiếu cơ sở RandomAgent."""
    print("-> Kiểm tra RandomAgent (Random Policy Baseline)...")
    agent1 = RandomAgent(action_space=5, seed=42)
    agent2 = RandomAgent(action_space=5, seed=42)
    assert isinstance(agent1, BaseAgent)

    seq1 = [agent1.select_action((1, 1, (0, 0, 0, 0)), eval_mode=True) for _ in range(500)]
    seq2 = [agent2.select_action((1, 1, (0, 0, 0, 0)), eval_mode=False) for _ in range(500)]
    assert seq1 == seq2, "RandomAgent cùng seed phải sinh chuỗi hành động giống hệt nhau."
    assert set(seq1) == {0, 1, 2, 3, 4}, "RandomAgent phải phủ đủ 5 hành động."

    # Kiểm tra phương thức update / decay_epsilon / save / load không lỗi
    agent1.update((0, 0, (0,)), 0, -5.0, (0, 1, (0,)), None, False, False)
    agent1.decay_epsilon()
    with tempfile.TemporaryDirectory() as temp_dir:
        meta_path = os.path.join(temp_dir, "random_meta.npy")
        agent1.save(meta_path)
        agent2.load(meta_path)
        assert agent2.action_space == 5
    print("   [OK] RandomAgent phân phối đều, tái lập theo seed và tương thích BaseAgent.")


def main() -> None:
    print("=" * 65)
    print("KIỂM THỬ ĐƠN VỊ CHO Q-LEARNING AGENT & RANDOM POLICY BASELINE")
    print("=" * 65)
    test_q_learning_init()
    test_state_key_conversion()
    test_select_action_and_tie_breaking()
    test_update_off_policy_formulas()
    test_decay_epsilon()
    test_save_load()
    test_random_agent()
    print("=" * 65)
    print("TẤT CẢ CÁC BÀI TEST Q-LEARNING & RANDOM AGENT ĐÃ VƯỢT QUA (PASSED)!")
    print("=" * 65)


if __name__ == "__main__":
    main()
