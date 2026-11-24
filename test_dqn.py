import torch
import numpy as np
# Import class DQNAgent của bạn (điều chỉnh đường dẫn import nếu cần)
from agents.dqn_pytorch import DQNAgent 

def test_dqn_agent():
    print("Khởi tạo DQNAgent...")
    # Giả định input_dim = 6 (x, y, c1, c2, c3, c4), action_dim = 5
    agent = DQNAgent(input_dim=6, action_dim=5)
    
    # 1. Test Select Action
    # State giả lập: xe ở giữa bản đồ (0.5, 0.5), 1 trụ bận (1.0), 3 trụ trống (0.0)
    dummy_state = np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0])
    action = agent.select_action(dummy_state, epsilon=0.1)
    print(f"Hành động Agent chọn (kiểu int): {action}, Thuộc tập {{0..4}}: {action in range(5)}")
    
    # 2. Test Replay Buffer & Update Network
    print("Đẩy 70 mẫu dữ liệu giả vào Replay Buffer để test Batch Update...")
    for _ in range(70):
        next_state = np.random.rand(6)
        reward = -5
        done = False
        # Push 5 thành phần: state, action, reward, next_state, done
        agent.replay_buffer.push(dummy_state, action, reward, next_state, done)
    
    print("Test quá trình huấn luyện (Cập nhật trọng số)...")
    try:
        # Gọi hàm update của agent (tên hàm có thể khác tùy Anti Gravity sinh ra)
        agent.update_network() 
        print("Cập nhật thành công! Không có lỗi Tensor Shape.")
    except Exception as e:
        print(f"Phát hiện lỗi trong quá trình update: {e}")

if __name__ == "__main__":
    test_dqn_agent()