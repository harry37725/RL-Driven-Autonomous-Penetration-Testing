"""
ddqn_agent.py
-------------
Double DQN agent with optional Dueling architecture.

Architecture:
    Input (state_size)
        → Dense(128, ReLU)
        → Dense(128, ReLU)
        → Dense(64,  ReLU)
        → [Value stream V(s)] + [Advantage stream A(s,a)]
        → Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

DDQN fix:
    Online net  → selects best action   a* = argmax Q_online(s', a)
    Target net  → evaluates that action Q_target(s', a*)
    This decoupling prevents overestimation of risky actions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent.replay_buffer import ReplayBuffer


# NEURAL NETWORK
class DuelingDQN(nn.Module):
    """
    Dueling DDQN network.

    Splits into two streams after shared layers:
        Value     V(s)     → scalar, how good is this state?
        Advantage A(s,a)   → vector, how much better is action a vs average?

    Combined: Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )

        # Value stream: state → scalar
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Advantage stream: state → action_size
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.shared(x)
        value     = self.value_stream(features)         # (batch, 1)
        advantage = self.advantage_stream(features)     # (batch, action_size)

        # Combine: subtract mean advantage for stability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# DDQN AGENT
class DDQNAgent:
    def __init__(
        self,
        state_size:         int   = 32,
        action_size:        int   = 20,
        lr:                 float = 1e-3,
        gamma:              float = 0.99,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay:      float = 0.995,
        batch_size:         int   = 64,
        buffer_capacity:    int   = 50_000,
        target_update_freq: int   = 1_000,
        device:             str   = "cpu",
    ):
        self.state_size         = state_size
        self.action_size        = action_size
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device             = torch.device(device)

        # Online network — trained every step
        self.online_net = DuelingDQN(state_size, action_size).to(self.device)

        # Target network — updated every target_update_freq steps
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity, state_size)

        self.steps_done = 0
        self.losses     = []

    # ACTION SELECTION
    def select_action(self, state: np.ndarray, valid_actions: list = None) -> int:
        if np.random.random() < self.epsilon:
            # Explore
            if valid_actions:
                return np.random.choice(valid_actions)
            return np.random.randint(self.action_size)

        # Exploit
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor).squeeze(0).cpu().numpy()

        if valid_actions:
            masked = np.full(self.action_size, -np.inf)
            masked[valid_actions] = q_values[valid_actions]
            return int(np.argmax(masked))

        return int(np.argmax(q_values))

    # STORE TRANSITION
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # TRAINING STEP
    def train_step(self) -> float:
        if not self.buffer.is_ready(self.batch_size):
            return 0.0

        batch = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(batch["states"]).to(self.device)
        actions     = torch.LongTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones       = torch.FloatTensor(batch["dones"]).to(self.device)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        if self.steps_done % self.target_update_freq == 0:
            self.update_target()

        return loss_val

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_tensor).squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "steps_done":  self.steps_done,
        }, path)
        print(f"Model saved → {path}")

    # def load(self, path: str):
    #     checkpoint = torch.load(path, map_location=self.device)
    #     self.online_net.load_state_dict(checkpoint["online_net"])
    #     self.target_net.load_state_dict(checkpoint["target_net"])
    #     self.optimizer.load_state_dict(checkpoint["optimizer"])
    #     self.epsilon    = checkpoint["epsilon"]
    #     self.steps_done = checkpoint["steps_done"]
    #     print(f"Model loaded ← {path}")

    def load(self, path: str, partial_transfer: bool = False):
        checkpoint = torch.load(path, map_location=self.device)
        
        saved_state_size  = checkpoint["online_net"]["shared.0.weight"].shape[1]
        saved_action_size = checkpoint["online_net"]["advantage_stream.2.weight"].shape[0]
        
        current_state_size  = self.state_size
        current_action_size = self.action_size
        
        sizes_match = (saved_state_size  == current_state_size and
                    saved_action_size == current_action_size)
        
        if sizes_match:
            # Clean load — no size mismatch
            self.online_net.load_state_dict(checkpoint["online_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            print(f"  [load] Full load from {path}")
        else:
            print(f"  [load] Size mismatch — attempting partial weight transfer")
            print(f"  Checkpoint : state={saved_state_size}, actions={saved_action_size}")
            print(f"  Current env: state={current_state_size}, actions={current_action_size}")
            
            self._partial_load(checkpoint["online_net"], self.online_net, "online")
            self._partial_load(checkpoint["target_net"],  self.target_net,  "target")
        
        self.target_net.eval()
        print(f"  [load] Done.")


    def _partial_load(self, saved_state_dict: dict, model, label: str):
        """
        Transfer weights layer by layer, skipping layers whose shape changed.
        
        Layers that transfer cleanly:
        - shared.2  (second Linear — hidden→hidden, shape unchanged)
        - shared.3  (BatchNorm or activation params if any)
        - value_stream.0, value_stream.2  (fully hidden, shape unchanged)
        - advantage_stream.0              (fully hidden, shape unchanged)
        
        Layers that are SKIPPED (shape changed):
        - shared.0.weight/bias            (input layer: 48→57 or 30→36)
        - advantage_stream.2.weight/bias  (output: 30→36 actions)
        """
        current_state = model.state_dict()
        
        transferred = []
        skipped     = []
        
        for name, saved_param in saved_state_dict.items():
            if name not in current_state:
                skipped.append((name, "not in current model"))
                continue
            
            current_param = current_state[name]
            
            if saved_param.shape == current_param.shape:
                current_state[name] = saved_param.clone()
                transferred.append(name)
            else:
                # Shape mismatch — try partial copy for the input layer
                if "shared.0.weight" in name:
                    current_state[name] = self._transfer_input_layer(
                        saved_param, current_param
                    )
                    transferred.append(f"{name} (partial — old cols copied, new cols random)")
                else:
                    skipped.append((name, f"{saved_param.shape} → {current_param.shape}"))
        
        model.load_state_dict(current_state)
        
        print(f"\n  [{label}] Transferred ({len(transferred)}):")
        for t in transferred:
            print(f"    ✅  {t}")
        print(f"  [{label}] Skipped ({len(skipped)}):")
        for name, reason in skipped:
            print(f"    ⚠️   {name}  ({reason})")


    def _transfer_input_layer(self, saved_weight, current_weight):
        """
        For the input projection (shared.0.weight):
        - Copy the first N columns (old state features) from checkpoint
        - Leave the new columns (extra state features) as random init
        
        Shape: [hidden_dim, state_size]
        Old  : [128, 48]
        New  : [128, 57]  ← copies cols 0..47, keeps cols 48..56 random
        """
        import torch
        new_w   = current_weight.clone()          # keeps random init for new cols
        old_dim = saved_weight.shape[1]           # 48
        new_dim = current_weight.shape[1]         # 57
        
        cols_to_copy = min(old_dim, new_dim)      # 48
        new_w[:, :cols_to_copy] = saved_weight[:, :cols_to_copy]
        
        print(f"    [input layer] Copied {cols_to_copy}/{new_dim} input features "
            f"({new_dim - cols_to_copy} new features left as random init)")
        return new_w


# QUICK TEST
if __name__ == "__main__":
    agent = DDQNAgent(state_size=32, action_size=20)
    print(f"Online net params : {sum(p.numel() for p in agent.online_net.parameters()):,}")
    print(f"Target net params : {sum(p.numel() for p in agent.target_net.parameters()):,}")

    state      = np.random.rand(32).astype(np.float32)
    action     = agent.select_action(state)
    next_state = np.random.rand(32).astype(np.float32)
    agent.store(state, action, 1.0, next_state, False)

    print(f"Buffer size : {len(agent.buffer)}")
    print(f"Epsilon     : {agent.epsilon}")
    print(f"Action selected: {action}")
    print("DDQN agent OK")
