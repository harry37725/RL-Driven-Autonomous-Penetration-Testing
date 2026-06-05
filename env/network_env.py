"""
network_env.py
--------------
Penetration Testing Environment for DDQN.

The agent starts at node 0 (Internet Entry) and must reach the
target node (Internal DB) by probing and exploiting machines in a
simulated corporate network, while avoiding detection by a defender
that patches nodes and raises alerts.

MDP:
  State  : 32-float vector  (compromised, patched, position, detection, timestep)
  Actions: 20               (probe node 0-9, exploit node 0-9)
  Reward : see _compute_reward()
  Done   : target reached | caught | timeout
"""

import numpy as np
import networkx as nx


# ─────────────────────────────────────────────
#  NETWORK TOPOLOGY
# ─────────────────────────────────────────────
#
#   [0: Entry] ──► [1: WebServer] ──► [2: AppServer] ──► [8: InternalDB ★]
#                        │                   │
#                        ▼                   ▼
#                  [3: MailServer]     [4: DevBox] ──► [7: AdminPanel] ──► [8]
#                        │                   │
#                        ▼                   ▼
#                  [6: DBBackup]       [5: FileServer]
#                        │
#                        └──────────────────► [8]
#
# Two main paths to target:
#   Fast path  : 0 → 1 → 2 → 8          (high noise, hard exploit at end)
#   Stealth path: 0 → 1 → 2 → 4 → 7 → 8 (lower noise, more hops)
# ─────────────────────────────────────────────

NODES = {
    0: {"name": "Entry Point",    "vulnerability": 0.95, "value": 0,   "color": "#4FC3F7"},
    1: {"name": "Web Server",     "vulnerability": 0.75, "value": 10,  "color": "#81C784"},
    2: {"name": "App Server",     "vulnerability": 0.55, "value": 15,  "color": "#81C784"},
    3: {"name": "Mail Server",    "vulnerability": 0.60, "value": 10,  "color": "#FFB74D"},
    4: {"name": "Dev Box",        "vulnerability": 0.80, "value": 12,  "color": "#FFB74D"},
    5: {"name": "File Server",    "vulnerability": 0.50, "value": 10,  "color": "#FFB74D"},
    6: {"name": "DB Backup",      "vulnerability": 0.45, "value": 20,  "color": "#CE93D8"},
    7: {"name": "Admin Panel",    "vulnerability": 0.35, "value": 25,  "color": "#EF9A9A"},
    8: {"name": "Internal DB",    "vulnerability": 0.25, "value": 100, "color": "#F44336"},  # TARGET
    9: {"name": "Dev Server",     "vulnerability": 0.65, "value": 15,  "color": "#FFB74D"},
}

EDGES = [
    # (from, to, difficulty, noise)
    (0, 1, 0.15, 0.10),   # Entry → Web Server        easy, quiet
    (0, 2, 0.40, 0.20),   # Entry → App Server 
    (0, 3, 0.35, 0.15),   # Entry → Mail Server  
    (1, 2, 0.35, 0.25),   # Web → App Server
    (1, 3, 0.30, 0.20),   # Web → Mail Server
    (2, 4, 0.40, 0.30),   # App → Dev Box
    (2, 8, 0.60, 0.45),   # App → Internal DB         hard, loud (direct path)
    (3, 6, 0.45, 0.25),   # Mail → DB Backup
    (4, 5, 0.35, 0.20),   # Dev Box → File Server
    (4, 7, 0.55, 0.35),   # Dev Box → Admin Panel
    (4, 9, 0.30, 0.15),   # Dev Box → Dev Server
    (5, 7, 0.50, 0.30),   # File Server → Admin Panel
    (6, 8, 0.60, 0.40),   # DB Backup → Internal DB
    (7, 8, 0.25, 0.20),   # Admin Panel → Internal DB  quieter path
    (9, 4, 0.25, 0.15),   # Dev Server → Dev Box
]

NUM_NODES      = len(NODES)       # 10
TARGET_NODE    = 8
ENTRY_NODE     = 0
MAX_STEPS      = 100

# Action space: 0-9 = probe node, 10-19 = exploit node
NUM_ACTIONS    = NUM_NODES * 2    # 20
STATE_SIZE     = NUM_NODES * 3 + 2  # compromised(10) + patched(10) + position(10) + detection(1) + timestep(1) = 32

# Reward constants
R_TARGET          = +500
R_NEW_NODE        = +50
R_FAILED_EXPLOIT  = -4
R_HIGH_ALERT      = -7
R_CAUGHT          = -200
R_STEP            = -4
DETECTION_DANGER  = 0.8
DETECTION_CAUGHT  = 1.0

# Detection dynamics
PROBE_NOISE       = 0.06
EXPLOIT_NOISE_MIN = 0.05
EXPLOIT_NOISE_MAX = 0.12
DETECTION_DECAY   = 0.05  # per step if agent stays quiet-ish

# Defender
PATCH_PROBABILITY = 0.15   # prob defender patches an exploited node each step


class NetworkEnv:
    """
    Penetration testing environment modelled as a directed weighted graph.

    Usage:
        env = NetworkEnv()
        state = env.reset()
        state, reward, done, info = env.step(action)
    """

    def __init__(self, defender_level: int = 1, seed: int = None):
        """
        Args:
            defender_level: 1 = reactive (default), 2 = adaptive
            seed          : random seed for reproducibility
        """
        self.defender_level = defender_level
        self.rng = np.random.default_rng(seed)

        # Build the graph once — it never changes structurally
        self.G = self._build_graph()

        # These are reset each episode
        self.compromised   = np.zeros(NUM_NODES, dtype=np.float32)
        self.patched       = np.zeros(NUM_NODES, dtype=np.float32)
        self.position      = ENTRY_NODE
        self.detection     = 0.0
        self.timestep      = 0
        self.done          = False
        self.probe_results = {}   # node_id → vulnerability score (revealed by probing)

        # Episode tracking
        self.episode_reward     = 0.0
        self.nodes_compromised  = 0
        self.caught             = False
        self.success            = False

        # Defender state (level 2)
        self.recently_exploited = []   # tracks last 3 exploited nodes for adaptive defender

        # Stats (updated each episode, useful for plotting)
        self.stats = {
            "reward":    0.0,
            "steps":     0,
            "success":   False,
            "caught":    False,
            "timeout":   False,
            "path":      [],
        }

    # ─────────────────────────────────────────────
    #  GRAPH BUILDER
    # ─────────────────────────────────────────────

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()

        for node_id, attrs in NODES.items():
            G.add_node(node_id, **attrs)

        for (u, v, difficulty, noise) in EDGES:
            G.add_edge(u, v, difficulty=difficulty, noise=noise)

        return G

    # ─────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment to start of a new episode. Returns initial state."""
        self.compromised            = np.zeros(NUM_NODES, dtype=np.float32)
        self.patched                = np.zeros(NUM_NODES, dtype=np.float32)
        self.position               = ENTRY_NODE
        self.detection              = 0.0
        self.timestep               = 0
        self.done                   = False
        self.probe_results          = {}
        self.episode_reward         = 0.0
        self.nodes_compromised      = 0
        self.caught                 = False
        self.success                = False
        self.recently_exploited     = []
        self.high_alert_steps       = 0

        # Entry node is already "compromised" — that's our starting foothold
        self.compromised[ENTRY_NODE] = 1.0

        self.stats = {
            "reward":  0.0,
            "steps":   0,
            "success": False,
            "caught":  False,
            "timeout": False,
            "path":    [ENTRY_NODE],
        }

        return self._get_state()

    # ─────────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────────

    def step(self, action: int):
        """
        Execute one action.

        Actions:
            0  – 9  : probe node (action)
            10 – 19 : exploit node (action - 10)

        Returns:
            next_state (np.ndarray), reward (float), done (bool), info (dict)
        """
        assert not self.done, "Episode is done. Call reset()."
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"

        self.timestep += 1
        reward = R_STEP   # baseline cost per step

        if action < NUM_NODES:
            # ── PROBE ──────────────────────────────
            reward += self._probe(action)
        else:
            # ── EXPLOIT ────────────────────────────
            target_node = action - NUM_NODES
            reward += self._exploit(target_node)

        # Detection decay each step (going quiet helps)
        self.detection = max(0.0, self.detection - DETECTION_DECAY)

        # Defender response
        self._defender_step()

        # Check termination
        done, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.done = done

        self.episode_reward += reward

        # Update stats
        self.stats["reward"] = self.episode_reward
        self.stats["steps"]  = self.timestep

        info = {
            "position":          self.position,
            "detection":         self.detection,
            "compromised_nodes": int(self.compromised.sum()),
            "patched_nodes":     int(self.patched.sum()),
            "success":           self.success,
            "caught":            self.caught,
        }

        return self._get_state(), reward, self.done, info

    # ─────────────────────────────────────────────
    #  ACTIONS
    # ─────────────────────────────────────────────

    def _probe(self, node: int) -> float:
        """
        Probe a node: low noise, reveals vulnerability score.
        Does not compromise the node.
        """
        self.detection = min(1.0, self.detection + PROBE_NOISE)
        self.probe_results[node] = self.G.nodes[node]["vulnerability"]
        return 0.0   # no reward for probing, just information gain

    def _exploit(self, node: int) -> float:
        """
        Attempt to compromise a node.
        Only succeeds if:
          - agent is adjacent to the node (has a path from current position)
          - node is not already compromised
          - node is not patched
        """
        reward = 0.0

        # Can't exploit current node or already compromised
        if node == self.position:
            return reward

        if self.compromised[node] == 1.0:
            return reward

        # Must be reachable from current position
        if not self.G.has_edge(self.position, node):
            # Invalid move — penalise slightly to discourage random flailing
            reward -= 2.0
            self.detection = min(1.0, self.detection + PROBE_NOISE)
            return reward

        # Patched node — can't exploit
        if self.patched[node] == 1.0:
            reward -= 3.0
            self.detection = min(1.0, self.detection + PROBE_NOISE)
            return reward

        # Get edge properties
        edge   = self.G[self.position][node]
        noise  = edge["noise"]
        difficulty = edge["difficulty"]

        # Raise detection
        noise_amount = EXPLOIT_NOISE_MIN + noise * (EXPLOIT_NOISE_MAX - EXPLOIT_NOISE_MIN)
        self.detection = min(1.0, self.detection + noise_amount)

        # Success probability = node vulnerability - edge difficulty
        vuln     = self.G.nodes[node]["vulnerability"]
        success_prob = max(0.05, vuln - difficulty * 0.5)
        success  = self.rng.random() < success_prob

        if success:
            self.compromised[node] = 1.0
            self.position = node
            self.nodes_compromised += 1
            self.recently_exploited.append(node)
            self.stats["path"].append(node)
            reward += R_NEW_NODE
            reward += self.G.nodes[node]["value"] * 0.1  # small bonus for high-value nodes

            try:
                dist = nx.shortest_path_length(self.G, node, TARGET_NODE)
                if dist <= 2:   # only reward being close
                    reward += max(0, (3 - dist) * 20)  # 2 hops=+20, 1 hop=+40
            except nx.NetworkXNoPath:
                pass

        else:
            # Failed exploit — made noise for nothing
            reward += R_FAILED_EXPLOIT

        # High alert penalty
        if self.detection >= DETECTION_DANGER:
            reward += R_HIGH_ALERT

        return reward

    # ─────────────────────────────────────────────
    #  DEFENDER
    # ─────────────────────────────────────────────

    def _defender_step(self):
        """Defender responds to attacker activity."""

        if self.defender_level == 1:
            PROTECTED_NODES = [0, 1]
            exploited = [n for n in range(NUM_NODES)
                        if self.compromised[n] == 1.0
                        and self.patched[n] == 0.0
                        and n != self.position
                        and n != ENTRY_NODE
                        and n not in PROTECTED_NODES]
            
            if self.timestep < 30:
                patch_prob = PATCH_PROBABILITY * 0.3
            else:
                patch_prob = PATCH_PROBABILITY

            if exploited and self.rng.random() < patch_prob:
                node_to_patch = self.rng.choice(exploited)
                self.patched[node_to_patch] = 1.0

        elif self.defender_level == 2:
            if self.recently_exploited:
                last = self.recently_exploited[-1]
                neighbours = list(self.G.successors(last))
                candidates = [n for n in neighbours
                            if self.patched[n] == 0.0
                            and n != self.position]
                if candidates and self.rng.random() < PATCH_PROBABILITY * 1.5:
                    node_to_patch = self.rng.choice(candidates)
                    self.patched[node_to_patch] = 1.0

            self.recently_exploited = self.recently_exploited[-3:]

        # ── High alert timer — caught if detection > 0.9 for 10+ consecutive steps ──
        if self.detection > 0.93:
            self.high_alert_steps += 1
        else:
            self.high_alert_steps = 0
    # ─────────────────────────────────────────────
    #  TERMINATION
    # ─────────────────────────────────────────────

    def _check_termination(self):
        """Returns (done, terminal_reward)."""

        # Caught via sustained high alert
        if self.high_alert_steps >= 25:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT

        # Caught via detection hitting 1.0
        if self.detection >= DETECTION_CAUGHT:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT

        # Target reached
        if self.position == TARGET_NODE:
            self.success = True
            self.stats["success"] = True
            return True, R_TARGET

        # Timeout
        if self.timestep >= MAX_STEPS:
            self.stats["timeout"] = True
            return True, 0.0

        return False, 0.0

    # ─────────────────────────────────────────────
    #  STATE
    # ─────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Build the 32-float state vector:
          [compromised x10] [patched x10] [position_onehot x10] [detection] [timestep_norm]
        """
        position_onehot = np.zeros(NUM_NODES, dtype=np.float32)
        position_onehot[self.position] = 1.0

        timestep_norm = np.float32(self.timestep / MAX_STEPS)
        detection     = np.float32(self.detection)

        state = np.concatenate([
            self.compromised,
            self.patched,
            position_onehot,
            [detection],
            [timestep_norm],
        ])

        return state.astype(np.float32)

    # ─────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────

    def get_valid_actions(self) -> list:
        """Returns list of actions that make sense from current position."""
        valid = []
        neighbours = list(self.G.successors(self.position))

        for node in range(NUM_NODES):
            # Probing: any node is technically probeable (recon)
            valid.append(node)          # probe action

        for node in neighbours:
            if self.compromised[node] == 0.0 and self.patched[node] == 0.0:
                valid.append(node + NUM_NODES)   # exploit action

        return list(set(valid))

    def get_node_name(self, node_id: int) -> str:
        return self.G.nodes[node_id]["name"]

    def get_graph(self) -> nx.DiGraph:
        """Returns the underlying NetworkX graph (for visualisation)."""
        return self.G

    def get_render_state(self) -> dict:
        """Returns everything the renderer needs to draw the current frame."""
        return {
            "graph":       self.G,
            "position":    self.position,
            "compromised": self.compromised.copy(),
            "patched":     self.patched.copy(),
            "detection":   self.detection,
            "timestep":    self.timestep,
            "target":      TARGET_NODE,
            "entry":       ENTRY_NODE,
            "success":     self.success,
            "caught":      self.caught,
            "path":        self.stats["path"].copy(),
        }

    @property
    def state_size(self):
        return STATE_SIZE

    @property
    def action_size(self):
        return NUM_ACTIONS

    @property
    def num_nodes(self):
        return NUM_NODES


# ─────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    env = NetworkEnv(defender_level=1, seed=42)
    state = env.reset()

    print(f"State size  : {env.state_size}")
    print(f"Action size : {env.action_size}")
    print(f"State       : {state}")
    print(f"State shape : {state.shape}")
    print()

    # Random agent
    total_reward = 0
    for step in range(50):
        action = np.random.randint(0, NUM_ACTIONS)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        action_type = "PROBE  " if action < NUM_NODES else "EXPLOIT"
        node        = action if action < NUM_NODES else action - NUM_NODES
        print(f"Step {step+1:3d} | {action_type} node {node} [{env.get_node_name(node):15s}] "
              f"| Reward: {reward:+7.1f} | Detection: {info['detection']:.2f} "
              f"| Position: {env.get_node_name(info['position'])}")

        if done:
            print()
            if info["success"]:
                print(f"✅  TARGET REACHED in {step+1} steps! Total reward: {total_reward:.1f}")
            elif info["caught"]:
                print(f"🚨  CAUGHT! Total reward: {total_reward:.1f}")
            else:
                print(f"⏱️   TIMEOUT. Total reward: {total_reward:.1f}")
            break