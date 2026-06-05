"""
replay_buffer.py
----------------
Experience replay buffer for DDQN.

Stores (state, action, reward, next_state, done) transitions.
Samples random minibatches for training to break correlation
between consecutive experiences.
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size circular buffer storing DDQN transitions.

    Args:
        capacity  : max number of transitions to store (older ones are dropped)
        state_size: dimension of the state vector
    """

    def __init__(self, capacity: int = 50_000, state_size: int = 32):
        self.capacity   = capacity
        self.state_size = state_size
        self.buffer     = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add one transition to the buffer."""
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> dict:
        """
        Prioritised sampling — recent transitions sampled more often.
        Helps agent learn from successful episodes faster.
        """
        assert len(self) >= batch_size

        n = len(self.buffer)
        
        # Weight recent experiences higher (last 20% get 3x more chance)
        weights = np.ones(n)
        recent_start = int(n * 0.8)
        weights[recent_start:] = 3.0
        weights = weights / weights.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=weights)
        batch   = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states":      np.stack(states),
            "actions":     np.array(actions,     dtype=np.int64),
            "rewards":     np.array(rewards,     dtype=np.float32),
            "next_states": np.stack(next_states),
            "dones":       np.array(dones,       dtype=np.float32),
        }
    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """True when buffer has enough samples to start training."""
        return len(self) >= batch_size