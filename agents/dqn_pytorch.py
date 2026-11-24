"""Deep Q-Network (DQN) Agent implemented in PyTorch for EV Charging Station navigation.

Architectural and Mathematical Specifications:
1. State & Action Space:
   - Action space: Discrete(5) -> {0: Up, 1: Down, 2: Left, 3: Right, 4: Charge}.
   - State vector phi(s): Normalized vector of size (2 + k), default k = 4.
   - State representation: [x / (N - 1), y / (M - 1), c_1, c_2, ..., c_k] with c_i in {0, 1}.
2. Q-Network Architecture:
   - Input layer: size (2 + k)
   - Hidden Layer 1: Linear(64) -> ReLU
   - Hidden Layer 2: Linear(64) -> ReLU
   - Output layer: Linear(5)
3. Hyperparameters:
   - Replay Buffer Capacity: 10,000 transitions.
   - Batch Size: 64.
   - Optimizer: Adam with Learning Rate = 0.001.
   - Loss Function: MSELoss.
   - Gamma (discount factor): 0.95.
   - Target Network Update Frequency (C): Every 100 steps (hard update).
   - Epsilon: Starts at 1.0, minimum 0.05, decay rate 0.995 per episode.
4. Compliance:
   - Strictly inherits from BaseAgent.
   - Implements ReplayBuffer, QNetwork, and DQNAgent classes.
"""

from __future__ import annotations

from collections import deque
import os
import random
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from config import NUM_ACTIONS


def _format_state(state: Any) -> np.ndarray:
    """Convert any supported state representation to a flat 1D float32 numpy array.

    Supports:
    - Normalized 1D numpy array from Gymnasium obs_type='box'.
    - PyTorch Tensors.
    - Flat lists/tuples.
    - Gymnasium obs_type='tuple' structures: (x, y, (c_1, ..., c_k)).
    """
    if isinstance(state, np.ndarray):
        return state.astype(np.float32).ravel()
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().numpy().astype(np.float32).ravel()
    if isinstance(state, (tuple, list)):
        if len(state) == 3 and isinstance(state[2], (tuple, list, np.ndarray)):
            flat_items = [float(state[0]), float(state[1]), *[float(c) for c in state[2]]]
            return np.array(flat_items, dtype=np.float32)
        return np.array(state, dtype=np.float32).ravel()
    return np.asarray(state, dtype=np.float32).ravel()


class ReplayBuffer:
    """Experience Replay Buffer using collections.deque to store transitions (s, a, r, s', done)."""

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None) -> None:
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store (default: 10,000).
            seed: Random seed for reproducible batch sampling.
        """
        self.capacity: int = int(capacity)
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)
        self.rng: random.Random = random.Random(seed)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Store a new experience transition into the buffer.

        Args:
            state: Current state observation.
            action: Action executed.
            reward: Reward received.
            next_state: Next state observation.
            done: Terminal flag (True if episode naturally ended).
        """
        s_arr = _format_state(state)
        s_next_arr = _format_state(next_state)
        self.buffer.append((s_arr, int(action), float(reward), s_next_arr, bool(done)))

    # Alias push as add for interface flexibility
    add = push

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of transitions uniformly at random.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones).
            - states: shape (B, state_dim), dtype float32
            - actions: shape (B, 1), dtype int64
            - rewards: shape (B, 1), dtype float32
            - next_states: shape (B, state_dim), dtype float32
            - dones: shape (B, 1), dtype float32
        """
        batch = self.rng.sample(self.buffer, batch_size)
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([t[2] for t in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Deep Q-Network approximating the action-value function Q(s, a).

    Architecture:
        - Input Layer: Linear(state_dim, 64) -> ReLU
        - Hidden Layer 1: Linear(64, 64) -> ReLU
        - Output Layer: Linear(64, action_dim)
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 5) -> None:
        """Initialize the Q-Network layers.

        Args:
            state_dim: Dimension of the state vector phi(s) = (2 + k), default 6.
            action_dim: Number of discrete actions, default 5.
        """
        super().__init__()
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Perform forward pass to compute Q(s, a) for all actions.

        Args:
            state: Tensor of state vectors, shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Tensor of Q-values for each action, shape (batch_size, action_dim).
        """
        return self.net(state)


class DQNAgent(BaseAgent):
    """Deep Q-Network (DQN) Agent implemented in PyTorch for EV Charging Station.

    Features:
    - Policy Network and Target Network with hard updates every C steps.
    - Experience Replay Buffer for off-policy transition storage and uniform sampling.
    - Epsilon-greedy action selection with random tie-breaking.
    - Rigorous device management (CPU/CUDA GPU) and tensor shape alignment.
    - Full compliance with BaseAgent abstract methods.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = NUM_ACTIONS,
        num_stations: int = 4,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DQNAgent with hyperparameters and network components.

        Args:
            state_dim: Size of state vector (2 + k). If None, calculated as 2 + num_stations.
            action_dim: Number of discrete actions (default 5).
            num_stations: Number of charging stations k (default 4).
            lr: Learning rate for Adam optimizer (default 0.001).
            gamma: Discount factor (default 0.95).
            epsilon: Initial exploration probability (default 1.0).
            epsilon_min: Minimum exploration probability (default 0.05).
            epsilon_decay: Epsilon decay rate per episode (default 0.995).
            buffer_capacity: Capacity of the ReplayBuffer (default 10,000).
            batch_size: Minibatch size for training (default 64).
            target_update_freq: Hard target update frequency C in steps (default 100).
            device: Computing device ('cpu', 'cuda', or torch.device). Auto-detected if None.
            seed: Random seed for reproducibility.
            **kwargs: Additional optional parameters (e.g. action_space alias).
        """
        super().__init__()

        # Handle aliases and dimensions
        if "action_space" in kwargs:
            action_dim = int(kwargs["action_space"])
        if state_dim is None:
            self.state_dim: int = 2 + int(num_stations)
        else:
            self.state_dim = int(state_dim)

        self.action_dim: int = int(action_dim)
        self.num_stations: int = int(num_stations)
        self.lr: float = float(lr)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(epsilon)
        self.epsilon_min: float = float(epsilon_min)
        self.epsilon_decay: float = float(epsilon_decay)
        self.buffer_capacity: int = int(buffer_capacity)
        self.batch_size: int = int(batch_size)
        self.target_update_freq: int = int(target_update_freq)

        # Device mapping
        if device is None:
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Random number generator and seeds
        self.seed: Optional[int] = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        # 1. Networks: Policy Net (trained) & Target Net (evaluation only)
        self.policy_net: QNetwork = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net: QNetwork = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.update_target_network()
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # 2. Optimizer & Loss Function
        self.optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion: nn.MSELoss = nn.MSELoss()

        # 3. Experience Replay Buffer
        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity=self.buffer_capacity, seed=seed)

        # Total training steps counter (for Target Network periodic update)
        self.total_steps: int = 0

    def select_action(
        self,
        state: Any,
        eval_mode: bool = False,
        epsilon: Optional[float] = None,
    ) -> int:
        """Select an action using the epsilon-greedy policy with random tie-breaking.

        Args:
            state: Current state observation (assumed normalized vector or convertible).
            eval_mode:
                - If False: Uses exploration governed by epsilon.
                - If True: Pure exploitation (greedy action selection).
            epsilon: Optional explicit exploration rate. If provided or if eval_mode is passed
                     as a float, overrides self.epsilon for this call.

        Returns:
            Selected integer action in range [0, action_dim - 1].
        """
        # Support callers passing epsilon as the 2nd positional argument
        if isinstance(eval_mode, (int, float)) and not isinstance(eval_mode, bool):
            current_eps = float(eval_mode)
            is_eval = False
        elif eval_mode:
            current_eps = 0.0
            is_eval = True
        elif epsilon is not None:
            current_eps = float(epsilon)
            is_eval = False
        else:
            current_eps = self.epsilon
            is_eval = False

        # Exploration: random action
        if not is_eval and self.rng.random() < current_eps:
            return int(self.rng.integers(0, self.action_dim))

        # Exploitation: forward through policy network with random tie-breaking
        state_arr = _format_state(state)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()

        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q, atol=1e-7))
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def update(
        self,
        s: Any,
        a: int,
        r: float,
        s_next: Any,
        a_next: Any = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> Optional[float]:
        """Update agent knowledge after an environment transition.

        Stores transition into the ReplayBuffer, triggers a gradient step if buffer
        capacity allows, and triggers hard target network updates every C steps.

        Args:
            s: Current state observation.
            a: Action executed.
            r: Reward received.
            s_next: Next state observation.
            a_next: Next action (optional / unused in off-policy DQN, supported for BaseAgent compatibility).
            terminated: Natural terminal condition (true goal reached or crash).
            truncated: Horizon limit reached (timeout).

        Returns:
            Computed loss value (float) if network was updated, or None.
        """
        # Handle calling conventions gracefully
        if isinstance(a_next, bool):
            actual_truncated = bool(terminated)
            actual_terminated = a_next
        else:
            actual_terminated = bool(terminated)
            actual_truncated = bool(truncated)

        # In standard MDP / Gymnasium RL, only natural termination zeroes out target bootstrapping
        done = actual_terminated
        self.replay_buffer.push(s, a, r, s_next, done)
        self.total_steps += 1

        # Train policy network if buffer contains enough samples
        loss = None
        if len(self.replay_buffer) >= self.batch_size:
            loss = self.update_network()

        # Hard update of target network every C steps
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss

    def update_network(self) -> Optional[float]:
        """Perform a single minibatch optimization step on the Policy Network.

        Mathematical formulation:
            Q(s, a) = policy_net(s)[a]
            y = r + (1 - done) * gamma * max_a' target_net(s')[a']
            Loss = MSE(Q(s, a), y)

        Returns:
            Loss value (float) if update was executed, or None if buffer size < batch_size.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 1. Sample minibatch from ReplayBuffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 2. Map arrays to PyTorch Tensors on self.device with strict shape alignment
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)           # (B, state_dim)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)            # (B, 1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)        # (B, 1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device) # (B, state_dim)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)             # (B, 1)

        # 3. Compute predicted Q(s, a) for the actions taken
        # policy_net(states_t) shape: (B, action_dim)
        # gather(1, actions_t) shape: (B, 1)
        state_action_values = self.policy_net(states_t).gather(dim=1, index=actions_t)

        # 4. Compute target Q-values with target network (no gradient flow, detached)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)  # shape: (B, action_dim)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]  # shape: (B, 1)
            target_q_values = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q_values
            target_q_values = target_q_values.detach()  # shape: (B, 1)

        # 5. Compute MSE loss between predicted Q-values and target Q-values
        loss = self.criterion(state_action_values, target_q_values)

        # 6. Backpropagation and gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Perform a hard update: copy all weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate epsilon by factor epsilon_decay down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save model weights, optimizer state, and training metadata to a file.

        Args:
            path: Target file path (e.g. 'models/dqn_model.pth').
        """
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "gamma": self.gamma,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_stations": self.num_stations,
        }
        torch.save(checkpoint, abs_path)

    def load(self, path: str) -> None:
        """Load model weights, optimizer state, and training metadata from a checkpoint file.

        Args:
            path: Path to the saved checkpoint file.
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {abs_path}")

        checkpoint = torch.load(abs_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "policy_net_state_dict" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            if "target_net_state_dict" in checkpoint:
                self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "epsilon" in checkpoint:
                self.epsilon = float(checkpoint["epsilon"])
            if "total_steps" in checkpoint:
                self.total_steps = int(checkpoint["total_steps"])
        elif isinstance(checkpoint, dict) and "policy_net" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            if "target_net" in checkpoint:
                self.target_net.load_state_dict(checkpoint["target_net"])
            else:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "epsilon" in checkpoint:
                self.epsilon = float(checkpoint["epsilon"])
            if "total_steps" in checkpoint:
                self.total_steps = int(checkpoint["total_steps"])
        elif isinstance(checkpoint, dict):
            # Fallback if raw state_dict of policy_net was saved
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            raise ValueError(f"Invalid checkpoint format loaded from: {abs_path}")
