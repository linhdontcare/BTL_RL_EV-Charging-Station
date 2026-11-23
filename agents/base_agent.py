"""Lớp cơ sở trừu tượng cho tất cả các tác tử Học tăng cường (RL Agents).

Tệp này định nghĩa giao diện (Interface/Abstract Base Class) mà mọi thuật toán
(SARSA, Q-Learning, DQN, ...) trong dự án EV Charging Station cần tuân thủ.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Lớp tác tử trừu tượng định nghĩa các phương thức chuẩn."""

    @abstractmethod
    def select_action(self, state: Any, eval_mode: bool = False) -> int:
        """Chọn hành động dựa trên quan sát trạng thái hiện tại.

        Tham số:
            state: Trạng thái quan sát nhận được từ môi trường.
            eval_mode: Nếu False, sử dụng chiến lược thăm dò (exploration);
                       Nếu True, sử dụng chính sách khai thác tham lam tuyệt đối (greedy).

        Trả về:
            Mã hành động số nguyên hợp lệ trong không gian hành động.
        """
        pass

    @abstractmethod
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
        """Cập nhật tri thức học tập của tác tử sau mỗi bước chuyển đổi môi trường.

        Tham số:
            s: Trạng thái hiện tại.
            a: Hành động đã thực thi tại trạng thái s.
            r: Phần thưởng nhận được sau khi thực thi hành động a.
            s_next: Trạng thái kế tiếp.
            a_next: Hành động kế tiếp được chọn tại trạng thái s_next.
            terminated: Trạng thái kết thúc tự nhiên của bài toán (đạt đích hoặc hỏng).
            truncated: Trạng thái dừng do hết thời gian tối đa (horizon limit).
        """
        pass

    @abstractmethod
    def decay_epsilon(self) -> None:
        """Suy giảm chỉ số khám phá epsilon sau mỗi episode huấn luyện."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Lưu trọng số hoặc Q-table của mô hình ra tệp."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Nạp trọng số hoặc Q-table của mô hình từ tệp đã lưu."""
        pass
