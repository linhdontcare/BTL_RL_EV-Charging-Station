"""Lớp môi trường chuẩn Gymnasium cho bài toán Xe điện tìm trạm sạc.

Kế thừa gymnasium.Env, hỗ trợ:
- Hai chế độ quan sát: obs_type='tuple' (Tabular RL) và obs_type='box' (DQN).
- 5 hành động rời rạc: Up, Down, Left, Right, Charge.
- 8 kịch bản chuẩn C01-C08 qua tham số options={'scenario': ...}.
- Render linh hoạt (None cho huấn luyện tốc độ cao, 'human', 'rgb_array').
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    ACTION_CHARGE,
    ACTION_DELTAS,
    BUSY,
    FAST,
    MAP_CONFIGS,
    MAX_STEPS,
    NORMAL,
    NUM_ACTIONS,
    OBSTACLE,
    REWARD_BUSY_CHARGE,
    REWARD_COLLISION,
    REWARD_FAST,
    REWARD_INVALID_CHARGE,
    REWARD_NORMAL,
    REWARD_STEP,
    ROAD,
)
from env.scenarios import ScenarioData, setup_scenario


class EVChargingEnv(gym.Env):
    """Môi trường mô phỏng xe điện tìm trạm sạc tuân thủ chuẩn Gymnasium."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_name: str = "default_10x10",
        obs_type: str = "tuple",
        max_steps: int = MAX_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        """Khởi tạo môi trường EV Charging.

        Tham số:
            map_name: Tên bản đồ cấu hình ('default_10x10' hoặc 'extended_15x15').
            obs_type: Kiểu quan sát ('tuple' cho Q-Learning/SARSA, 'box' cho DQN).
            max_steps: Giới hạn số bước tối đa mỗi episode (mặc định 100).
            render_mode: Chế độ render (None, 'human', 'rgb_array').
        """
        super().__init__()

        if map_name not in MAP_CONFIGS:
            raise ValueError(f"Bản đồ không hợp lệ: '{map_name}'. Chọn từ: {list(MAP_CONFIGS.keys())}")
        self.map_config = MAP_CONFIGS[map_name]
        self.map_name = map_name

        self.grid_width: int = self.map_config["width"]
        self.grid_height: int = self.map_config["height"]
        self.stations: list[dict[str, Any]] = self.map_config["stations"]
        self.num_stations: int = len(self.stations)
        self.station_positions: dict[tuple[int, int], dict[str, Any]] = {
            s["pos"]: s for s in self.stations
        }
        self.entrance: tuple[int, int] = self.map_config.get("entrance", (self.grid_width - 1, 0))
        self.entrance_col: int = self.map_config.get("entrance_col", self.grid_width - 1)

        self.obs_type = obs_type.lower()
        if self.obs_type not in ("tuple", "box"):
            raise ValueError("obs_type phải là 'tuple' hoặc 'box'.")

        self.max_steps = max_steps
        self.render_mode = render_mode

        # Không gian hành động: 0: Lên, 1: Xuống, 2: Trái, 3: Phải, 4: Sạc
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Không gian quan sát theo yêu cầu
        if self.obs_type == "box":
            # Vector chuẩn hóa [x/(N-1), y/(M-1), c_1, ..., c_k] thuộc [0.0, 1.0]
            low = np.zeros(2 + self.num_stations, dtype=np.float32)
            high = np.ones(2 + self.num_stations, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            # Tuple dạng (x, y, tuple(station_status)) cho Tabular RL
            self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_width),
                spaces.Discrete(self.grid_height),
                spaces.MultiBinary(self.num_stations),
            ))

        # Trạng thái nội bộ của môi trường
        self.agent_pos: tuple[int, int] = self.map_config["start_pos"]
        self.station_status: list[int] = [0] * self.num_stations
        self.extra_obstacles: set[tuple[int, int]] = set()
        self.grid: list[list[int]] = []
        self.trail: list[tuple[int, int]] = []
        self.step_count: int = 0
        self.collision_attempts: int = 0
        self.cumulative_reward: float = 0.0
        self.is_success: bool = False
        self.charged_type: Optional[str] = None
        self.current_scenario_id: str = "C06"
        self.status: str = "Khởi tạo"
        self.terminated: bool = False
        self.truncated: bool = False

        # Quản lý renderer (chỉ khởi tạo khi có yêu cầu render để tối ưu huấn luyện)
        self.renderer: Any = None

    def _rebuild_grid(self) -> list[list[int]]:
        """Xây dựng lại ma trận lưới 2D phản ánh vật cản tĩnh và trạng thái động của trạm."""
        grid = [[ROAD for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        # 1. Gán vật cản cố định từ cấu hình bản đồ
        for x, y in self.map_config["obstacles"]:
            grid[y][x] = OBSTACLE

        # 2. Gán các vật cản phụ theo kịch bản (nếu có)
        for x, y in self.extra_obstacles:
            grid[y][x] = OBSTACLE

        # 3. Gán trạng thái các trụ sạc
        for idx, station in enumerate(self.stations):
            x, y = station["pos"]
            is_busy = bool(self.station_status[idx])
            if is_busy:
                grid[y][x] = BUSY
            else:
                grid[y][x] = station["type"]

        return grid

    def _get_obs(self) -> Any:
        """Trả về quan sát hiện tại theo định dạng obs_type."""
        x, y = self.agent_pos
        if self.obs_type == "box":
            denom_x = max(1, self.grid_width - 1)
            denom_y = max(1, self.grid_height - 1)
            obs_array = np.array(
                [x / denom_x, y / denom_y, *[float(c) for c in self.station_status]],
                dtype=np.float32,
            )
            return obs_array
        # Mặc định obs_type == 'tuple'
        return (int(x), int(y), tuple(int(c) for c in self.station_status))

    def _get_info(self) -> dict[str, Any]:
        """Trả về từ điển info chứa đầy đủ 5 khóa theo đặc tả kỹ thuật."""
        return {
            "step_count": self.step_count,
            "collision_attempts": self.collision_attempts,
            "is_success": self.is_success,
            "charged_type": self.charged_type,
            "scenario_id": self.current_scenario_id,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Thiết lập lại trạng thái môi trường về đầu episode.

        Tham số:
            seed: Hạt giống ngẫu nhiên theo chuẩn Gymnasium.
            options: Tùy chọn thiết lập, hỗ trợ {'scenario': 'C01'..'C08'}.
        """
        super().reset(seed=seed)

        # Lấy mã kịch bản từ options (mặc định là 'C06' ngẫu nhiên)
        scenario_id = "C06"
        if options and "scenario" in options:
            scenario_id = str(options["scenario"]).upper()

        scenario_data: ScenarioData = setup_scenario(
            scenario_id=scenario_id,
            map_config=self.map_config,
            np_random=self.np_random,
        )

        self.current_scenario_id = scenario_data.scenario_id
        self.agent_pos = scenario_data.start_pos
        self.station_status = list(scenario_data.station_status)
        self.extra_obstacles = set(scenario_data.extra_obstacles)

        self.grid = self._rebuild_grid()
        self.trail = [self.agent_pos]
        self.step_count = 0
        self.collision_attempts = 0
        self.cumulative_reward = 0.0
        self.is_success = False
        self.charged_type = None
        self.status = f"Đang tìm đường ({scenario_data.scenario_id})"
        self.terminated = False
        self.truncated = False

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Thực thi một bước hành động trong môi trường.

        Tham số:
            action: Hành động (0: Up, 1: Down, 2: Left, 3: Right, 4: Charge).

        Trả về:
            (observation, reward, terminated, truncated, info)
        """
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, self.terminated, self.truncated, self._get_info()

        self.step_count += 1
        reward: float = 0.0

        if action in (0, 1, 2, 3):
            # Di chuyển xe điện
            dx, dy = ACTION_DELTAS[action]
            target_x = self.agent_pos[0] + dx
            target_y = self.agent_pos[1] + dy

            # Kiểm tra tính hợp lệ của tọa độ đích
            is_within_bounds = 0 <= target_x < self.grid_width and 0 <= target_y < self.grid_height
            is_blocked = not is_within_bounds or (self.grid[target_y][target_x] in (OBSTACLE, BUSY))

            if is_blocked:
                # Va chạm: phạt -100, tăng số lần va chạm, xe giữ nguyên vị trí
                reward = REWARD_COLLISION
                self.collision_attempts += 1
                self.status = "Va chạm - đường bị chặn"
            else:
                # Bước đi hợp lệ: phạt chi phí bước đi -5, cập nhật vị trí và vệt đường
                self.agent_pos = (target_x, target_y)
                self.trail.append(self.agent_pos)
                reward = REWARD_STEP
                self.status = f"Đang di chuyển ({target_y}, {target_x})"

            # Kiểm tra giới hạn số bước tối đa (Horizon Truncation)
            if self.step_count >= self.max_steps and not self.terminated:
                self.truncated = True
                self.status = "Hết số bước tối đa"

        elif action == ACTION_CHARGE:
            # Thực thi hành động sạc điện
            current_cell_code = self.grid[self.agent_pos[1]][self.agent_pos[0]]

            if current_cell_code == FAST:
                reward = REWARD_FAST
                self.is_success = True
                self.charged_type = "fast"
                self.status = "Sạc nhanh thành công (+120)"
            elif current_cell_code == NORMAL:
                reward = REWARD_NORMAL
                self.is_success = True
                self.charged_type = "normal"
                self.status = "Sạc thường thành công (+100)"
            elif current_cell_code == BUSY:
                reward = REWARD_BUSY_CHARGE
                self.is_success = False
                self.charged_type = "busy"
                self.status = "Trụ đang bận (-100)"
            else:
                reward = REWARD_INVALID_CHARGE
                self.is_success = False
                self.charged_type = None
                self.status = "Charge sai vị trí (-50)"

            # Lệnh Charge luôn kết thúc episode: terminated = True, truncated = False
            self.terminated = True
            self.truncated = False

        else:
            raise ValueError(f"Hành động không hợp lệ: {action}. Giá trị chấp nhận từ 0 đến 4.")

        self.cumulative_reward += reward

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, self.terminated, self.truncated, info

    def render(self) -> Any:
        """Hiển thị giao diện trực quan của môi trường."""
        if self.render_mode is None:
            return None

        if self.renderer is None:
            # Lazy import để không khởi tạo Pygame khi render_mode = None
            from gui.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer(self)

        return self.renderer.render(self)

    def close(self) -> None:
        """Đóng tài nguyên hiển thị Pygame nếu đã mở."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

