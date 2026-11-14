"""Kiểm thử tự động toàn diện cho hệ thống môi trường sau tái cấu trúc."""

from __future__ import annotations

import os
import sys

# Thêm đường dẫn thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np

import config
from env.ev_charging_env import EVChargingEnv
from env.scenarios import setup_scenario


def test_config():
    """Kiểm tra tệp config.py."""
    print("-> Kiểm tra config.py...")
    assert config.ROAD == 0
    assert config.OBSTACLE == 1
    assert config.NORMAL == 2
    assert config.FAST == 3
    assert config.BUSY == 4

    assert config.REWARD_FAST == 120.0
    assert config.REWARD_NORMAL == 100.0
    assert config.REWARD_BUSY_CHARGE == -100.0
    assert config.REWARD_INVALID_CHARGE == -50.0
    assert config.REWARD_COLLISION == -100.0
    assert config.REWARD_STEP == -5.0

    assert config.MAX_STEPS == 100
    assert config.GAMMA == 0.95
    assert config.SEEDS == [42, 100, 2026, 7, 99]

    # Kiểm tra bản đồ default_10x10
    cfg10 = config.MAP_CONFIGS["default_10x10"]
    assert cfg10["width"] == 10 and cfg10["height"] == 10
    assert cfg10["num_stations"] == 4
    # Đếm số trụ NORMAL và FAST
    normal_count = sum(1 for s in cfg10["stations"] if s["type"] == config.NORMAL)
    fast_count = sum(1 for s in cfg10["stations"] if s["type"] == config.FAST)
    assert normal_count == 2 and fast_count == 2, f"Expected 2 normal, 2 fast, got {normal_count}, {fast_count}"

    # Kiểm tra bản đồ extended_15x15
    cfg15 = config.MAP_CONFIGS["extended_15x15"]
    assert cfg15["width"] == 15 and cfg15["height"] == 15
    assert cfg15["num_stations"] == 8
    print("   [OK] config.py đầy đủ và chính xác 100%.")


def test_scenarios():
    """Kiểm tra bộ sinh kịch bản C01 - C08."""
    print("-> Kiểm tra scenarios.py (C01 - C08)...")
    rng = np.random.default_rng(42)
    cfg10 = config.MAP_CONFIGS["default_10x10"]

    for sc_id in ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"]:
        data = setup_scenario(sc_id, cfg10, rng)
        assert data.scenario_id == sc_id
        assert len(data.station_status) == 4
        assert isinstance(data.start_pos, tuple)
        assert len(data.start_pos) == 2

        # Kiểm tra cụ thể
        if sc_id == "C01":
            assert data.station_status[0] == 0  # Trụ mục tiêu trống
        elif sc_id == "C05":
            assert data.station_status[1] == 1  # Trụ gần nhất bận
        elif sc_id == "C07":
            assert data.station_status[0] == 0 and data.station_status[1] == 0  # Cả 2 đều trống
        elif sc_id == "C08":
            assert all(c == 1 for c in data.station_status)  # Tất cả đều bận
    print("   [OK] Bộ sinh 8 kịch bản C01 - C08 hoạt động chính xác.")


def test_env_gymnasium_compliance():
    """Kiểm tra tính tuân thủ Gymnasium của EVChargingEnv."""
    print("-> Kiểm tra EVChargingEnv tuân thủ chuẩn Gymnasium...")

    # 1. Kiểm tra không render Pygame khi render_mode = None
    env = EVChargingEnv(map_name="default_10x10", obs_type="tuple", render_mode=None)
    assert env.renderer is None

    # 2. Kiểm tra reset() và cấu trúc info
    obs, info = env.reset(seed=42, options={"scenario": "C01"})
    assert isinstance(obs, tuple)
    assert len(obs) == 3
    assert isinstance(obs[0], int) and isinstance(obs[1], int) and isinstance(obs[2], tuple)

    required_keys = {"step_count", "collision_attempts", "is_success", "charged_type", "scenario_id"}
    assert set(info.keys()) == required_keys, f"Thiếu khóa trong info: {set(info.keys()) ^ required_keys}"
    assert info["step_count"] == 0
    assert info["collision_attempts"] == 0
    assert info["is_success"] is False
    assert info["charged_type"] is None
    assert info["scenario_id"] == "C01"

    # 3. Kiểm tra định dạng Box cho DQN
    env_box = EVChargingEnv(map_name="default_10x10", obs_type="box", render_mode=None)
    obs_b, info_b = env_box.reset(seed=100)
    assert isinstance(obs_b, np.ndarray)
    assert obs_b.dtype == np.float32
    assert obs_b.shape == (2 + 4,)
    assert np.all(obs_b >= 0.0) and np.all(obs_b <= 1.0)

    # 4. Kiểm tra hành động di chuyển hợp lệ (-5)
    env.reset(seed=42, options={"scenario": "C01"})
    # Tại C01, start_pos = (2, 4). Di chuyển xuống (Down, action=1) tới (2, 5) là trụ sạc NORMAL
    obs, reward, terminated, truncated, info = env.step(config.ACTION_DOWN)
    assert reward == config.REWARD_STEP  # -5.0
    assert env.agent_pos == (2, 5)
    assert terminated is False
    assert truncated is False
    assert info["step_count"] == 1
    assert info["collision_attempts"] == 0

    # 5. Kiểm tra hành động sạc thành công tại trụ NORMAL (+100)
    obs, reward, terminated, truncated, info = env.step(config.ACTION_CHARGE)
    assert reward == config.REWARD_NORMAL  # +100.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert info["charged_type"] == "normal"

    # 6. Kiểm tra va chạm biên / vật cản (-100, giữ nguyên vị trí, collision_attempts += 1)
    env.reset(seed=42, options={"scenario": "C02"})  # start_pos = (8, 1)
    # Di chuyển lên (Up = 0) đâm vào tường phía trên y=0 (ngoại trừ x=8)
    # Tại (8, 1), ô (8, 0) là entrance (ROAD)! Di chuyển sang phải (Right = 3) tới (9, 1) là tường!
    obs, reward, terminated, truncated, info = env.step(config.ACTION_RIGHT)
    assert reward == config.REWARD_COLLISION  # -100.0
    assert env.agent_pos == (8, 1)  # Giữ nguyên vị trí
    assert info["collision_attempts"] == 1
    assert terminated is False

    # 7. Kiểm tra Charge sai vị trí (-50)
    env.reset(seed=42, options={"scenario": "C02"})  # start_pos = (8, 1) là ROAD
    obs, reward, terminated, truncated, info = env.step(config.ACTION_CHARGE)
    assert reward == config.REWARD_INVALID_CHARGE  # -50.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is False
    assert info["charged_type"] is None

    # 8. Kiểm tra hết số bước tối đa (truncated = True)
    env.reset(seed=42, options={"scenario": "C08"})  # Tất cả trụ bận
    for s in range(config.MAX_STEPS - 1):
        # Đâm va chạm giữ nguyên vị trí
        obs, reward, terminated, truncated, info = env.step(config.ACTION_RIGHT)
        assert not terminated
        assert not truncated

    # Bước thứ 100
    obs, reward, terminated, truncated, info = env.step(config.ACTION_RIGHT)
    assert truncated is True
    assert terminated is False
    assert info["step_count"] == 100

    # 9. Kiểm tra lệnh Charge tại bước 100 vẫn ra terminated = True, truncated = False
    env.reset(seed=42, options={"scenario": "C08"})
    for s in range(config.MAX_STEPS - 1):
        env.step(config.ACTION_RIGHT)
    # Bước thứ 100 thực hiện Charge
    obs, reward, terminated, truncated, info = env.step(config.ACTION_CHARGE)
    assert terminated is True
    assert truncated is False  # Phải là False theo đúng ràng buộc bất di bất dịch!
    print("   [OK] EVChargingEnv thỏa mãn 100% các tiêu chí Gymnasium và luật chơi.")


def main():
    print("=" * 60)
    print("CHẠY BỘ KIỂM THỬ TỰ ĐỘNG CHO MÔ HÌNH SAU REFACTOR")
    print("=" * 60)
    test_config()
    test_scenarios()
    test_env_gymnasium_compliance()
    print("=" * 60)
    print("TẤT CẢ CÁC BÀI TEST ĐÃ VƯỢT QUA XUẤT SẮC (PASSED)!")
    print("=" * 60)


if __name__ == "__main__":
    main()

