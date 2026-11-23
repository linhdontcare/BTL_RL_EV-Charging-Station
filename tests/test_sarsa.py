"""Kiểm thử tự động cho thuật toán SarsaAgent."""

from collections import defaultdict
import os
import shutil
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np

from agents.sarsa import SarsaAgent
from config import NUM_ACTIONS, REWARD_NORMAL, REWARD_STEP


def test_sarsa_init():
    """Kiểm tra khởi tạo và siêu tham số mặc định."""
    print("-> Kiểm tra khởi tạo SarsaAgent...")
    agent = SarsaAgent()
    assert agent.alpha == 0.10
    assert agent.gamma == 0.95
    assert agent.epsilon == 1.0
    assert agent.epsilon_min == 0.05
    assert agent.epsilon_decay == 0.995
    assert agent.action_space == NUM_ACTIONS == 5
    assert isinstance(agent.q_table, defaultdict)
    # Kiểm tra giá trị mặc định của state mới là vector 5 số 0
    dummy_key = (0, 0, (0, 0, 0, 0))
    assert np.all(agent.q_table[dummy_key] == 0.0)
    assert len(agent.q_table[dummy_key]) == 5
    print("   [OK] Khởi tạo và siêu tham số chính xác.")


def test_state_key_conversion():
    """Kiểm tra hàm chuyển đổi state sang key tuple hashable."""
    print("-> Kiểm tra get_state_key...")
    # Tuple chuẩn từ Gymnasium
    obs_tuple = (2, 5, (0, 1, 1, 0))
    key1 = SarsaAgent.get_state_key(obs_tuple)
    assert key1 == (2, 5, (0, 1, 1, 0))
    assert isinstance(key1, tuple)
    assert isinstance(key1[2], tuple)

    # Dạng list bên trong
    obs_list = (3, 7, [1, 0, 0, 1])
    key2 = SarsaAgent.get_state_key(obs_list)
    assert key2 == (3, 7, (1, 0, 0, 1))

    # Hashable để làm key trong dict
    test_dict = {}
    test_dict[key1] = 10.0
    assert test_dict[(2, 5, (0, 1, 1, 0))] == 10.0
    print("   [OK] get_state_key chuẩn xác và hashable.")


def test_select_action_greedy_and_random_tie_breaking():
    """Kiểm tra select_action với eval_mode=True và tie-breaking."""
    print("-> Kiểm tra select_action và tie-breaking...")
    agent = SarsaAgent(seed=42)
    state = (2, 4, (0, 1, 1, 1))

    # 1. Khi tất cả Q bằng 0, chọn ngẫu nhiên trong các hành động hòa (tie-breaking)
    actions = [agent.select_action(state, eval_mode=True) for _ in range(50)]
    assert len(set(actions)) > 1, "Cần random tie-breaking khi tất cả Q-values bằng nhau"

    # 2. Khi có 1 giá trị max duy nhất
    key = SarsaAgent.get_state_key(state)
    agent.q_table[key][1] = 50.0  # Action 1 là Down
    for _ in range(20):
        a = agent.select_action(state, eval_mode=True)
        assert a == 1, f"Khai thác phải chọn action 1, nhận {a}"

    # 3. Khi có 2 giá trị max bằng nhau (Action 1 và Action 4)
    agent.q_table[key][4] = 50.0
    tied_actions = {agent.select_action(state, eval_mode=True) for _ in range(50)}
    assert tied_actions == {1, 4}, f"Tie-breaking phải chia đều giữa {1, 4}, nhận {tied_actions}"
    print("   [OK] select_action khai thác và random tie-breaking hoạt động hoàn hảo.")


def test_update_on_policy_formulas():
    """Kiểm tra công thức cập nhật On-policy TD (SARSA)."""
    print("-> Kiểm tra công thức update On-policy TD...")
    agent = SarsaAgent(alpha=0.10, gamma=0.95, seed=42)

    s = (2, 4, (0, 1, 1, 1))
    a = 1  # Down
    r = REWARD_STEP  # -5.0
    s_next = (2, 5, (0, 1, 1, 1))
    a_next = 4  # Charge
    terminated = False
    truncated = False

    s_key = SarsaAgent.get_state_key(s)
    s_next_key = SarsaAgent.get_state_key(s_next)

    # Giả lập Q[s_next][a_next] = 100.0
    agent.q_table[s_next_key][a_next] = 100.0

    # Bước chưa terminated: y_target = r + gamma * Q[s_next][a_next] = -5.0 + 0.95 * 100.0 = 90.0
    # Q[s][a] mới = 0.0 + 0.10 * (90.0 - 0.0) = 9.0
    agent.update(s, a, r, s_next, a_next, terminated=False, truncated=False)
    assert np.isclose(agent.q_table[s_key][a], 9.0), f"Expected 9.0, got {agent.q_table[s_key][a]}"

    # Bước truncated: y_target vẫn là r + gamma * Q[s_next][a_next]
    agent.update(s, a, r, s_next, a_next, terminated=False, truncated=True)
    # y_target = 90.0, Q_cu = 9.0 -> Q_moi = 9.0 + 0.10 * (90.0 - 9.0) = 17.1
    assert np.isclose(agent.q_table[s_key][a], 17.1), f"Expected 17.1, got {agent.q_table[s_key][a]}"

    # Bước terminated == True: y_target = r (không phụ thuộc Q[s_next])
    r_terminal = REWARD_NORMAL  # 100.0
    agent.update(s_next, a_next, r_terminal, (0, 0, (0, 0, 0, 0)), 0, terminated=True)
    # y_target = 100.0, Q_cu = 100.0 -> td_error = 0 -> Q vẫn là 100.0
    assert np.isclose(agent.q_table[s_next_key][a_next], 100.0)

    # Thử từ 0 lên:
    dummy_s = (1, 1, (0, 0, 0, 0))
    dummy_key = SarsaAgent.get_state_key(dummy_s)
    agent.update(dummy_s, 2, 80.0, s_next, 0, terminated=True)
    # y_target = 80.0, Q_cu = 0.0 -> Q_moi = 0.0 + 0.10 * 80.0 = 8.0
    assert np.isclose(agent.q_table[dummy_key][2], 8.0)
    print("   [OK] Công thức update On-policy (terminated vs truncated/non-terminated) chính xác 100%.")


def test_decay_epsilon():
    """Kiểm tra suy giảm epsilon."""
    print("-> Kiểm tra decay_epsilon...")
    agent = SarsaAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == 0.5
    agent.decay_epsilon()
    assert agent.epsilon == 0.25
    agent.decay_epsilon()
    assert agent.epsilon == 0.125
    agent.decay_epsilon()
    assert agent.epsilon == 0.0625
    agent.decay_epsilon()
    assert agent.epsilon == 0.05  # Đạt min
    agent.decay_epsilon()
    assert agent.epsilon == 0.05  # Không xuống dưới min
    print("   [OK] decay_epsilon dừng chính xác tại epsilon_min.")


def test_save_load():
    """Kiểm tra lưu và nạp Q-table."""
    print("-> Kiểm tra save và load...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_sarsa.npy")
        agent1 = SarsaAgent(seed=42)
        key1 = (2, 5, (0, 1, 1, 0))
        agent1.q_table[key1] = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        agent1.save(model_path)

        assert os.path.exists(model_path)

        agent2 = SarsaAgent(seed=100)
        agent2.load(model_path)
        assert isinstance(agent2.q_table, defaultdict)
        assert np.allclose(agent2.q_table[key1], [1.1, 2.2, 3.3, 4.4, 5.5])
        # Kiểm tra state chưa thấy vẫn sinh zeros(5)
        new_key = (9, 9, (1, 1, 1, 1))
        assert np.allclose(agent2.q_table[new_key], np.zeros(5))
    print("   [OK] save và load hoạt động trơn tru.")


def main():
    print("=" * 60)
    print("KIỂM THỬ ĐƠN VỊ CHO SARSA AGENT")
    print("=" * 60)
    test_sarsa_init()
    test_state_key_conversion()
    test_select_action_greedy_and_random_tie_breaking()
    test_update_on_policy_formulas()
    test_decay_epsilon()
    test_save_load()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST SARSA ĐÃ VƯỢT QUA XUẤT SẮC (PASSED)!")
    print("=" * 60)


if __name__ == "__main__":
    main()
