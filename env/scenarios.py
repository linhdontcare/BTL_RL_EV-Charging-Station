"""Bộ sinh dữ liệu động cho 8 kịch bản C01–C08 theo đặc tả bài toán RL.

Tách bạch hoàn toàn:
1. Dữ liệu tĩnh: Kích thước lưới, tường bao, vật cản cố định, tọa độ và loại trụ sạc L_stations.
2. Dữ liệu động theo episode:
   - Tọa độ xuất phát của xe điện: (x0, y0).
   - Vector trạng thái nhị phân của các trụ: station_status = [c_1, c_2, ..., c_k]
     với 0: Trống (Free), 1: Bận (Busy).
   - Các vật cản bổ sung (nếu có để định hình hành lang/thử thách).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import FAST, NORMAL


@dataclass
class ScenarioData:
    """Dữ liệu động thiết lập cho một episode theo kịch bản cụ thể."""

    scenario_id: str
    start_pos: tuple[int, int]
    station_status: list[int]  # 0: trống, 1: bận
    extra_obstacles: set[tuple[int, int]] = field(default_factory=set)
    description: str = ""


def get_valid_road_cells(
    map_config: dict[str, Any],
    extra_obstacles: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Trả về danh sách tất cả các tọa độ ô đường đi hợp lệ (không phải vật cản/trụ sạc)."""
    width: int = map_config["width"]
    height: int = map_config["height"]
    obstacles: set[tuple[int, int]] = map_config["obstacles"]
    station_positions: set[tuple[int, int]] = {s["pos"] for s in map_config["stations"]}
    extra = extra_obstacles or set()

    valid_cells: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            pos = (x, y)
            if pos not in obstacles and pos not in station_positions and pos not in extra:
                valid_cells.append(pos)
    return valid_cells


def setup_scenario(
    scenario_id: str,
    map_config: dict[str, Any],
    np_random: np.random.Generator,
) -> ScenarioData:
    """Tạo cấu hình động cho episode dựa trên mã kịch bản C01 đến C08.
    
    Tham số:
        scenario_id: Mã kịch bản ("C01" đến "C08").
        map_config: Cấu hình bản đồ tĩnh (default_10x10 hoặc extended_15x15).
        np_random: Generator số ngẫu nhiên chuẩn của Gymnasium Env.
        
    Trả về:
        ScenarioData chứa vị trí xuất phát, vector trạng thái trụ và vật cản phụ.
    """
    map_name: str = map_config.get("name", "default_10x10")
    stations: list[dict[str, Any]] = map_config["stations"]
    num_stations: int = len(stations)
    default_start: tuple[int, int] = map_config["start_pos"]
    sid = scenario_id.upper()

    # -----------------------------------------------------------------------
    # KỊCH BẢN C01: Xe xuất phát gần một trụ sạc đang trống
    # -----------------------------------------------------------------------
    if sid == "C01":
        # Chọn trụ đầu tiên (NORMAL) và đặt xe ngay sát phía trên (1 bước)
        target_station = stations[0]
        tx, ty = target_station["pos"]
        start_pos = (tx, ty - 1)  # Ô đường đi hợp lệ ngay trước trụ
        # Trụ mục tiêu trống (0), các trụ còn lại bận (1)
        status = [1] * num_stations
        status[0] = 0
        return ScenarioData(
            scenario_id="C01",
            start_pos=start_pos,
            station_status=status,
            description="Xe xuất phát ngay sát trụ sạc thường đang trống (khoảng cách 1 bước).",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C02: Xe xuất phát ở góc xa khu vực trụ sạc
    # -----------------------------------------------------------------------
    if sid == "C02":
        # Xe xuất phát tại vị trí xa nhất (cổng vào góc trên hoặc góc đối diện)
        start_pos = default_start
        # Một số trụ sạc ngẫu nhiên đang trống
        status = [0 if i % 2 == 0 else 1 for i in range(num_stations)]
        return ScenarioData(
            scenario_id="C02",
            start_pos=start_pos,
            station_status=status,
            description="Xe xuất phát ở góc xa khu vực trạm sạc, kiểm tra khả năng lập kế hoạch đường dài.",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C03: Bản đồ có nhiều vật cản / hành lang hẹp giữa xe và trụ sạc
    # -----------------------------------------------------------------------
    if sid == "C03":
        extra_obstacles: set[tuple[int, int]] = set()
        if map_name == "default_10x10":
            # Tạo hành lang hẹp tại hàng 4 (chỉ chừa khe hở tại x = 5)
            extra_obstacles.update({(6, 4), (7, 4)})
            start_pos = (8, 1)
        else:
            # Bản đồ 15x15: tạo vách ngăn phụ tạo hành lang zic-zac
            extra_obstacles.update({(10, 4), (11, 4), (12, 4)})
            start_pos = (13, 1)

        # Trụ sạc mục tiêu đang trống
        status = [0 if i == 0 else 1 for i in range(num_stations)]
        return ScenarioData(
            scenario_id="C03",
            start_pos=start_pos,
            station_status=status,
            extra_obstacles=extra_obstacles,
            description="Xuất hiện hành lang hẹp/nhiều vật cản buộc xe điều hướng chính xác tránh va chạm.",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C04: Trụ sạc gần nhất bị vật cản chắn, buộc đi vòng hoặc chọn trụ khác
    # -----------------------------------------------------------------------
    if sid == "C04":
        extra_obstacles = set()
        if map_name == "default_10x10":
            # Đặt xe tại (1, 4). Trụ gần nhất là (2, 5).
            # Chặn trực tiếp tại ô (2, 4) khiến xe không thể tiếp cận trực diện (2, 5)
            extra_obstacles.add((2, 4))
            start_pos = (1, 4)
            # Cả trụ 0 (2, 5) và trụ 1 (4, 5) đều trống
            status = [0, 0, 1, 1]
        else:
            # 15x15: đặt xe tại (2, 4), chắn tại (3, 4)
            extra_obstacles.add((3, 4))
            start_pos = (2, 4)
            status = [0] * 2 + [1] * (num_stations - 2)

        return ScenarioData(
            scenario_id="C04",
            start_pos=start_pos,
            station_status=status,
            extra_obstacles=extra_obstacles,
            description="Trụ gần nhất bị vật cản chắn mặt trước, kiểm tra khả năng đi đường vòng hoặc đổi đích đến.",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C05: Trụ sạc gần nhất đang bận (mã 4), buộc tìm trụ trống khác
    # -----------------------------------------------------------------------
    if sid == "C05":
        if map_name == "default_10x10":
            # Xe ở (5, 4). Trụ gần nhất là trụ 1 (4, 5) cách 2 bước.
            start_pos = (5, 4)
            # Trụ 1 bị bận (c_1 = 1), trụ 0 và trụ 3 trống
            status = [0, 1, 1, 0]
        else:
            # 15x15: xe ở (6, 4). Trụ gần nhất là (5, 5) [FAST] bị bận.
            start_pos = (6, 4)
            status = [0, 1, 1, 0] + [1] * (num_stations - 4)

        return ScenarioData(
            scenario_id="C05",
            start_pos=start_pos,
            station_status=status,
            description="Trụ gần nhất bị bận (c_i = 1, mã ô 4), ép tác tử nhận biết trạng thái để không đâm vào trụ bận.",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C06: Sinh ngẫu nhiên vị trí xuất phát và trạng thái trụ sạc
    # -----------------------------------------------------------------------
    if sid == "C06":
        valid_cells = get_valid_road_cells(map_config)
        random_idx = int(np_random.integers(0, len(valid_cells)))
        start_pos = valid_cells[random_idx]

        # Sinh trạng thái ngẫu nhiên nhị phân cho k trụ sạc
        raw_status = [int(np_random.integers(0, 2)) for _ in range(num_stations)]
        # Đảm bảo có ít nhất một trụ sạc đang trống để bài toán có nghiệm
        if all(c == 1 for c in raw_status):
            free_idx = int(np_random.integers(0, num_stations))
            raw_status[free_idx] = 0

        return ScenarioData(
            scenario_id="C06",
            start_pos=start_pos,
            station_status=raw_status,
            description="Ngẫu nhiên vị trí xuất phát và cấu hình bận/trống của các trụ sạc.",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C07: Sạc thường (+100) và sạc nhanh (+120) cùng trống,
    #              trụ nhanh xa hơn trụ thường từ 1–3 bước đi.
    # -----------------------------------------------------------------------
    if sid == "C07":
        if map_name == "default_10x10":
            # Trụ 0: NORMAL tại (2, 5).
            # Trụ 1: FAST tại (4, 5).
            # Đặt xe tại (1, 5):
            # Khoảng cách tới (2, 5) là 1 bước.
            # Khoảng cách tới (4, 5) là 3 bước (xa hơn đúng 2 bước!).
            start_pos = (1, 5)
            # Cả hai trụ đều trống, các trụ khác bận
            status = [0, 0, 1, 1]
        else:
            # 15x15: Trụ 0: (3, 5) NORMAL, Trụ 1: (5, 5) FAST.
            # Đặt xe tại (2, 5): cách Trụ 0 đúng 1 bước, cách Trụ 1 đúng 3 bước.
            start_pos = (2, 5)
            status = [0, 0] + [1] * (num_stations - 2)

        return ScenarioData(
            scenario_id="C07",
            start_pos=start_pos,
            station_status=status,
            description="Thử thách đánh đổi: Trụ sạc nhanh (+120) xa hơn trụ thường (+100) 2 bước đi.",
        )

    # -----------------------------------------------------------------------
    # KỊCH BẢN C08: Tất cả trụ sạc đều bận (station_status = [1, 1, ..., 1])
    # -----------------------------------------------------------------------
    if sid == "C08":
        start_pos = default_start
        status = [1] * num_stations
        return ScenarioData(
            scenario_id="C08",
            start_pos=start_pos,
            station_status=status,
            description="Tất cả trụ sạc đều bận, kiểm tra điều kiện biên an toàn đến khi truncated = True.",
        )

    # Mặc định rơi về C06 nếu mã không khớp
    return setup_scenario("C06", map_config, np_random)

