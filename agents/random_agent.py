"""Tác tử mốc tham chiếu cơ sở Random Policy (Chính sách ngẫu nhiên đồng nhất).

Theo đặc tả báo cáo (Mục 2.6 & Mục 3.1):
- Random Policy KHÔNG phải thuật toán Học tăng cường được huấn luyện,
  mà là mốc tham chiếu tính toán cơ sở (Baseline) để định lượng mức độ
  cải thiện của Q-Learning, SARSA và DQN.
- Tại mỗi bước thời gian t, chọn ngẫu nhiên đồng nhất:
    a_t ~ Uniform({0: Up, 1: Down, 2: Left, 3: Right, 4: Charge})
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from agents.base_agent import BaseAgent
from config import NUM_ACTIONS


class RandomAgent(BaseAgent):
    """Mốc tham chiếu cơ sở chọn hành động ngẫu nhiên đồng nhất."""

    def __init__(
        self,
        action_space: int = NUM_ACTIONS,
        seed: Optional[int] = None,
    ) -> None:
        """Khởi tạo tác tử Random Policy.

        Tham số:
            action_space: Số lượng hành động rời rạc (mặc định 5).
            seed: Hạt giống ngẫu nhiên phục vụ tái lập kết quả thực nghiệm.
        """
        super().__init__()
        self.action_space: int = int(action_space)
        self.seed: Optional[int] = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def select_action(self, state: Any, eval_mode: bool = False) -> int:
        """Chọn ngẫu nhiên đồng nhất một hành động trong {0, ..., action_space - 1}.

        Lưu ý: Cả khi eval_mode=True hay False, Random Policy luôn chọn
        ngẫu nhiên đồng nhất theo đúng Thuật toán 4 trong báo cáo.
        """
        del state, eval_mode  # Không sử dụng trạng thái hay chế độ khai thác
        return int(self.rng.integers(0, self.action_space))

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
        """Random Policy không học nên phương thức cập nhật không làm gì."""
        del s, a, r, s_next, a_next, terminated, truncated

    def decay_epsilon(self) -> None:
        """Random Policy không có tham số epsilon để suy giảm."""
        return None

    def save(self, path: str) -> None:
        """Lưu thông tin cấu hình cơ bản (tương thích giao diện BaseAgent)."""
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        metadata = {
            "agent_type": "RandomAgent",
            "action_space": self.action_space,
            "seed": self.seed,
        }
        np.save(abs_path, metadata)

    def load(self, path: str) -> None:
        """Nạp thông tin cấu hình cơ bản (tương thích giao diện BaseAgent)."""
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại: {abs_path}")
        loaded_data = np.load(abs_path, allow_pickle=True).item()
        if not isinstance(loaded_data, dict):
            raise ValueError(
                f"Dữ liệu trong {abs_path} không hợp lệ (cần dict, nhận {type(loaded_data)})."
            )
        if "action_space" in loaded_data:
            self.action_space = int(loaded_data["action_space"])
