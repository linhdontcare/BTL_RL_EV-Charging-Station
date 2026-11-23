"""Thuật toán SARSA (On-policy TD Control) cho bài toán điều hướng xe điện tìm trạm sạc.

Cài đặt chi tiết:
- Kế thừa lớp BaseAgent.
- Quản lý Q-table dưới dạng defaultdict(lambda: np.zeros(NUM_ACTIONS)).
- Chuyển đổi trạng thái sang key dạng tuple hashable: (x, y, tuple(station_status)).
- Lựa chọn hành động epsilon-greedy khi huấn luyện, tham lam tuyệt đối với random tie-breaking khi đánh giá.
- Cập nhật On-policy TD:
    y_target = r nếu terminated == True
    y_target = r + gamma * Q[s_next][a_next] nếu ngược lại (kể cả khi truncated == True)
    Q[s][a] += alpha * (y_target - Q[s][a])
- Hỗ trợ lưu và nạp Q-table ra/từ định dạng .npy.
"""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any, Optional

import numpy as np

from agents.base_agent import BaseAgent
from config import NUM_ACTIONS


class SarsaAgent(BaseAgent):
    """Tác tử SARSA học tăng cường On-policy dạng bảng (Tabular SARSA)."""

    def __init__(
        self,
        action_space: int = NUM_ACTIONS,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        """Khởi tạo tác tử SARSA.

        Tham số:
            action_space: Số lượng hành động rời rạc (mặc định 5).
            alpha: Tốc độ học (learning rate), mặc định 0.10.
            gamma: Hệ số chiết khấu (discount factor), mặc định 0.95.
            epsilon: Tỷ lệ khám phá ban đầu, mặc định 1.0.
            epsilon_min: Tỷ lệ khám phá tối thiểu, mặc định 0.05.
            epsilon_decay: Hệ số suy giảm epsilon sau mỗi episode, mặc định 0.995.
            seed: Hạt giống ngẫu nhiên phục vụ tái lập thực nghiệm.
        """
        super().__init__()
        self.action_space: int = action_space
        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(epsilon)
        self.epsilon_min: float = float(epsilon_min)
        self.epsilon_decay: float = float(epsilon_decay)

        # Trình tạo số ngẫu nhiên cho tác tử
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Cấu trúc Q-table: ánh xạ state_key -> np.ndarray kích thước (action_space,)
        self.q_table: defaultdict[tuple[int, int, tuple[int, ...]], np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_space, dtype=np.float64)
        )

    @staticmethod
    def get_state_key(state: Any) -> tuple[int, int, tuple[int, ...]]:
        """Chuyển đổi quan sát từ môi trường thành khóa tuple hashable chuẩn.

        Khóa có cấu trúc: (x, y, tuple(station_status)) khớp chính xác với EVChargingEnv.

        Tham số:
            state: Quan sát nhận từ env.reset() hoặc env.step().

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
        """Chọn hành động dựa trên trạng thái hiện tại.

        Tham số:
            state: Quan sát trạng thái hiện tại từ môi trường.
            eval_mode:
                - False: Chính sách epsilon-greedy (ngẫu nhiên với xác suất epsilon,
                  ngược lại chọn argmax Q(s, a) với random tie-breaking).
                - True: Chính sách khai thác tham lam tuyệt đối argmax Q(s, a)
                  (với random tie-breaking nếu có nhiều hành động cùng giá trị max).

        Trả về:
            Mã hành động được lựa chọn (0 đến action_space - 1).
        """
        key = self.get_state_key(state)
        q_values = self.q_table[key]

        if not eval_mode and self.rng.random() < self.epsilon:
            # Khám phá ngẫu nhiên (Exploration)
            return int(self.rng.integers(0, self.action_space))

        # Khai thác (Exploitation): argmax Q(s, a) với random tie-breaking
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
        a_next: int,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Cập nhật Q-value theo công thức On-policy TD Control (SARSA).

        Công thức mục tiêu:
            Nếu terminated == True:
                y_target = r
            Ngược lại (kể cả khi truncated == True):
                y_target = r + gamma * Q[s_next][a_next]

        Cập nhật trọng số:
            Q[s][a] += alpha * (y_target - Q[s][a])
        """
        s_key = self.get_state_key(s)

        if terminated:
            y_target = float(r)
        else:
            s_next_key = self.get_state_key(s_next)
            y_target = float(r) + self.gamma * float(self.q_table[s_next_key][a_next])

        td_error = y_target - self.q_table[s_key][a]
        self.q_table[s_key][a] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Giảm epsilon theo hệ số epsilon_decay cho đến khi chạm epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Lưu Q-table dạng dict(self.q_table) ra file .npy.

        Tham số:
            path: Đường dẫn tệp lưu (ví dụ: 'models/sarsa_table.npy').
        """
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        q_dict = dict(self.q_table)
        np.save(abs_path, q_dict)

    def load(self, path: str) -> None:
        """Nạp Q-table từ file .npy vào lại defaultdict.

        Tham số:
            path: Đường dẫn tệp nạp (ví dụ: 'models/sarsa_table.npy').
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình tại: {abs_path}")

        loaded_data = np.load(abs_path, allow_pickle=True).item()
        if not isinstance(loaded_data, dict):
            raise ValueError(f"Dữ liệu trong {abs_path} không hợp lệ (cần dict, nhận {type(loaded_data)}).")

        self.q_table = defaultdict(
            lambda: np.zeros(self.action_space, dtype=np.float64),
            loaded_data,
        )
