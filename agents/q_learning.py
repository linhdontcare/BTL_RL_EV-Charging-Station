"""Thuật toán Q-Learning (Off-policy TD Control) cho bài toán điều hướng xe điện tìm trạm sạc.

Cài đặt bám sát Thuật toán 1 và Bảng 12 trong báo cáo:
- Kế thừa lớp BaseAgent.
- Quản lý Q-table dạng defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)).
- Khóa trạng thái tuple hashable: (x, y, tuple(station_status)) khớp với EVChargingEnv.
- Lựa chọn hành động epsilon-greedy khi huấn luyện; tham lam tuyệt đối với
  random tie-breaking khi đánh giá (eval_mode=True).
- Cập nhật Off-policy TD Control:
    y_target = r                                   nếu terminated == True
    y_target = r + gamma * max_{a'} Q[s_next][a']  nếu ngược lại (kể cả truncated == True)
    Q[s][a] += alpha * (y_target - Q[s][a])
- Lưu và nạp Q-table dưới định dạng .npy tương thích toàn bộ hệ thống.
"""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Optional

import numpy as np

from agents.base_agent import BaseAgent
from config import GAMMA, NUM_ACTIONS


class QLearningAgent(BaseAgent):
    """Tác tử Q-Learning học tăng cường Off-policy dạng bảng (Tabular Q-Learning)."""

    def __init__(
        self,
        action_space: int = NUM_ACTIONS,
        alpha: float = 0.10,
        gamma: float = GAMMA,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        """Khởi tạo tác tử Q-Learning.

        Tham số:
            action_space: Số lượng hành động rời rạc (mặc định 5).
            alpha: Tốc độ học (learning rate), mặc định 0.10.
            gamma: Hệ số chiết khấu (discount factor), mặc định 0.95.
            epsilon: Tỷ lệ thăm dò ban đầu, mặc định 1.0.
            epsilon_min: Tỷ lệ thăm dò tối thiểu, mặc định 0.05.
            epsilon_decay: Hệ số suy giảm epsilon sau mỗi episode, mặc định 0.995.
            seed: Hạt giống ngẫu nhiên phục vụ tái lập thực nghiệm.
        """
        super().__init__()
        self.action_space: int = int(action_space)
        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(epsilon)
        self.epsilon_min: float = float(epsilon_min)
        self.epsilon_decay: float = float(epsilon_decay)

        # Bộ sinh số ngẫu nhiên riêng cho tác tử để đảm bảo tái lập khoa học
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Bảng Q-table động chỉ cấp phát cho các trạng thái thực tế xuất hiện
        self.q_table: defaultdict[tuple[int, int, tuple[int, ...]], np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_space, dtype=np.float64)
        )

    @staticmethod
    def get_state_key(state: Any) -> tuple[int, int, tuple[int, ...]]:
        """Chuyển đổi quan sát từ môi trường sang khóa tuple hashable chuẩn.

        Khóa có dạng: (x, y, tuple(station_status)) đồng bộ với EVChargingEnv.

        Tham số:
            state: Quan sát trả về từ env.reset() hoặc env.step().

        Trả về:
            Tuple hashable (x, y, tuple(station_status)).
        """
        if isinstance(state, tuple) and len(state) == 3 and isinstance(state[2], tuple):
            return (int(state[0]), int(state[1]), tuple(int(c) for c in state[2]))

        if isinstance(state, (tuple, list)):
            if len(state) == 3:
                return (int(state[0]), int(state[1]), tuple(int(c) for c in state[2]))
            if len(state) > 3:
                return (int(state[0]), int(state[1]), tuple(int(c) for c in state[2:]))

        if isinstance(state, np.ndarray):
            flat = state.ravel()
            return (int(flat[0]), int(flat[1]), tuple(int(c) for c in flat[2:]))

        raise TypeError(f"Không thể chuyển đổi trạng thái '{state}' sang khóa Q-table.")

    def select_action(self, state: Any, eval_mode: bool = False) -> int:
        """Chọn hành động theo chính sách epsilon-greedy hoặc tham lam tuyệt đối.

        Tham số:
            state: Trạng thái hiện tại từ môi trường.
            eval_mode:
                - False: Chính sách thăm dò epsilon-greedy.
                - True: Khai thác tham lam tuyệt đối (epsilon = 0) với random tie-breaking.

        Trả về:
            Mã hành động nguyên trong [0, action_space - 1].
        """
        key = self.get_state_key(state)
        q_values = self.q_table[key]

        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_space))

        # Khai thác với random tie-breaking khi có nhiều hành động đạt cùng giá trị max
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q, atol=1e-8))

        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def update(
        self,
        s: Any,
        a: int,
        r: float,
        s_next: Any,
        a_next: Optional[int] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Cập nhật bảng Q-table theo công thức Off-policy TD Control (Q-Learning).

        Công thức mục tiêu (Thuật toán 1):
            - Nếu terminated == True (đã thực hiện lệnh Charge):
                  y_target = r
            - Ngược lại (kể cả khi truncated == True do chạm max_steps):
                  y_target = r + gamma * max_{a'} Q[s_next][a']
            - Cập nhật:
                  Q[s][a] <- Q[s][a] + alpha * (y_target - Q[s][a])

        Lưu ý: Tham số `a_next` được giữ để tương thích hoàn toàn với giao diện
        BaseAgent (dùng chung với SARSA), nhưng Q-Learning luôn dùng max_{a'} Q(s', a').
        """
        del a_next, truncated  # Off-policy dùng max_a' và vẫn bootstrap khi truncated=True

        s_key = self.get_state_key(s)

        if terminated:
            y_target = float(r)
        else:
            s_next_key = self.get_state_key(s_next)
            max_next_q = float(np.max(self.q_table[s_next_key]))
            y_target = float(r) + self.gamma * max_next_q

        td_error = y_target - self.q_table[s_key][a]
        self.q_table[s_key][a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Giảm tỷ lệ thăm dò epsilon sau mỗi episode xuống tối thiểu epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Lưu Q-table dạng dict(self.q_table) ra tệp .npy.

        Tham số:
            path: Đường dẫn tệp đích (ví dụ: 'models/q_learning_table.npy').
        """
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        q_dict = dict(self.q_table)
        np.save(abs_path, q_dict)

    def load(self, path: str) -> None:
        """Nạp Q-table từ tệp .npy vào lại defaultdict.

        Tham số:
            path: Đường dẫn tệp nguồn (ví dụ: 'models/q_learning_table.npy').
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {abs_path}")

        loaded_data = np.load(abs_path, allow_pickle=True).item()
        if not isinstance(loaded_data, dict):
            raise ValueError(
                f"Dữ liệu trong {abs_path} không hợp lệ (cần dict, nhận {type(loaded_data)})."
            )

        self.q_table = defaultdict(
            lambda: np.zeros(self.action_space, dtype=np.float64),
            loaded_data,
        )
