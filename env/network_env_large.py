"""
network_env_large.py
--------------------
15-node Corporate Network Environment for DDQN — Stage 3.

Extends the 10-node environment with:
  - 5 new node types: DMZ, VPN Gateway, Cloud Instance, Honeypot, Secondary DB
  - Progressive difficulty: environment gets harder as agent succeeds
  - Honeypot mechanic: instant high alert if exploited
  - Credential requirement: VPN Gateway needs DMZ first
  - Larger state vector: 47 floats

Progressive Difficulty Levels:
  Level 1 (0–50  cumulative successes): Base difficulty, reactive defender
  Level 2 (50–150 successes):           Faster patching, adaptive defender
  Level 3 (150–300 successes):          Honeypot traps added, patrol mechanic
  Level 4 (300+  successes):            Full adaptive defender + decoy nodes

Network Topology:
  [0: Entry] ──► [1: WebServer] ──► [2: AppServer] ──► [8: InternalDB ★]
                      │                   │
                      ▼                   ▼
               [3: MailServer]      [4: DevBox] ──► [7: AdminPanel] ──► [8]
                      │                   │
                      ▼                   ▼
               [6: DBBackup]       [5: FileServer]
                      │                   │
                      ▼                   ▼
               [8: InternalDB]     [9: DevServer]
                                          │
  NEW ──────────────────────────────────  ▼
  [0] ──► [10: DMZ] ──► [11: VPNGateway*] ──► [12: CloudInstance] ──► [8]
                │                                      │
                ▼                                      ▼
         [13: Honeypot⚠]                      [14: SecondaryDB] ──► [8]

  * VPN Gateway requires DMZ credential
  ⚠ Honeypot: exploiting triggers instant high alert (detection → 0.85)

MDP:
  State  : 47-float vector
  Actions: 30 (probe 0-14, exploit 0-14)
  Reward : same structure as 10-node + progressive penalties
  Done   : target reached | caught | timeout
"""

import numpy as np
import networkx as nx


# ─────────────────────────────────────────────
#  NETWORK DEFINITION
# ─────────────────────────────────────────────

NODES_LARGE = {
    # ── Original 10 nodes ────────────────────
    0:  {"name": "Entry Point",     "vulnerability": 0.95, "value": 0,   "type": "entry",    "is_honeypot": False},
    1:  {"name": "Web Server",      "vulnerability": 0.75, "value": 10,  "type": "server",   "is_honeypot": False},
    2:  {"name": "App Server",      "vulnerability": 0.55, "value": 15,  "type": "server",   "is_honeypot": False},
    3:  {"name": "Mail Server",     "vulnerability": 0.60, "value": 10,  "type": "server",   "is_honeypot": False},
    4:  {"name": "Dev Box",         "vulnerability": 0.80, "value": 12,  "type": "workstation","is_honeypot": False},
    5:  {"name": "File Server",     "vulnerability": 0.50, "value": 10,  "type": "server",   "is_honeypot": False},
    6:  {"name": "DB Backup",       "vulnerability": 0.45, "value": 20,  "type": "database", "is_honeypot": False},
    7:  {"name": "Admin Panel",     "vulnerability": 0.35, "value": 25,  "type": "admin",    "is_honeypot": False},
    8:  {"name": "Internal DB",     "vulnerability": 0.25, "value": 100, "type": "target",   "is_honeypot": False},
    9:  {"name": "Dev Server",      "vulnerability": 0.65, "value": 15,  "type": "server",   "is_honeypot": False},
    # ── New 5 nodes ───────────────────────────
    10: {"name": "DMZ Server",      "vulnerability": 0.70, "value": 20,  "type": "dmz",      "is_honeypot": False},
    11: {"name": "VPN Gateway",     "vulnerability": 0.45, "value": 30,  "type": "gateway",  "is_honeypot": False},
    12: {"name": "Cloud Instance",  "vulnerability": 0.85, "value": 15,  "type": "cloud",    "is_honeypot": False},
    13: {"name": "Honeypot",        "vulnerability": 0.90, "value": 0,   "type": "honeypot", "is_honeypot": True},
    14: {"name": "Secondary DB",    "vulnerability": 0.40, "value": 35,  "type": "database", "is_honeypot": False},
}

# (from, to, base_difficulty, base_noise)
EDGES_LARGE = [
    # ── Original edges ────────────────────────
    (0,  1,  0.15, 0.10),
    (0,  2,  0.40, 0.20),
    (0,  3,  0.35, 0.15),
    (1,  2,  0.35, 0.25),
    (1,  3,  0.30, 0.20),
    (2,  4,  0.40, 0.30),
    (2,  8,  0.60, 0.45),
    (3,  6,  0.45, 0.25),
    (4,  5,  0.35, 0.20),
    (4,  7,  0.55, 0.35),
    (4,  9,  0.30, 0.15),
    (5,  7,  0.50, 0.30),
    (6, 14,  0.35, 0.20),   # DB Backup → Secondary DB  ← ADD THIS
    (6,  8,  0.60, 0.40),
    (7,  8,  0.25, 0.20),
    (9,  4,  0.25, 0.15),
    # ── New edges ─────────────────────────────
    (0,  10, 0.25, 0.15),   # Entry → DMZ          easy first hop
    (10, 11, 0.50, 0.30),   # DMZ → VPN Gateway    requires credential
    (10, 13, 0.10, 0.05),   # DMZ → Honeypot       looks easy — trap!
    (11, 12, 0.35, 0.25),   # VPN → Cloud          moderate
    (12, 8,  0.55, 0.40),   # Cloud → Internal DB  alternative path
    (12, 14, 0.30, 0.20),   # Cloud → Secondary DB
    (14, 8,  0.40, 0.30),   # Secondary DB → Internal DB
    (1,  10, 0.30, 0.20),   # Web → DMZ            cross-link
    (10, 2,  0.45, 0.30),   # DMZ → App Server
]

NUM_NODES_L   = len(NODES_LARGE)   # 15
TARGET_NODE_L = 8
ENTRY_NODE_L  = 0
HONEYPOT_NODE = 13
VPN_NODE      = 11
DMZ_NODE      = 10

NUM_ACTIONS_L = NUM_NODES_L * 2    # 30
# State: compromised(15) + patched(15) + position(15) + detection(1) + timestep(1) + difficulty_level(1) = 48
STATE_SIZE_L  = NUM_NODES_L * 3 + 3

MAX_STEPS_L   = 150   # more steps for larger network

# ── Reward constants (same as 10-node) ────────
R_TARGET_L         = +800
R_NEW_NODE_L       = +200
R_FAILED_EXPLOIT_L = -4
R_HIGH_ALERT_L     = -5
R_CAUGHT_L         = -200
R_STEP_L           = -2
R_HONEYPOT_L       = -100  # extra penalty for triggering honeypot

DETECTION_DANGER_L  = 0.8
DETECTION_CAUGHT_L  = 1.0

PROBE_NOISE_L       = 0.08
EXPLOIT_NOISE_MIN_L = 0.05
EXPLOIT_NOISE_MAX_L = 0.12
DETECTION_DECAY_L   = 0.06

# ── Progressive difficulty thresholds ─────────
DIFF_THRESHOLDS = {1: 0, 2: 50}

# Patch probability per difficulty level
PATCH_PROB_BY_LEVEL = {1: 0.10, 2: 0.15}

# Edge difficulty multiplier per level
EDGE_DIFF_MULTIPLIER = {1: 1.0, 2: 1.10}


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────

class NetworkEnvLarge:
    """
    15-node progressive difficulty penetration testing environment.

    Progressive difficulty levels (based on cumulative successes):
      Level 1 — Base: reactive defender, standard difficulty
      Level 2 — Harder: adaptive defender, faster patching
      Level 3 — Hard: honeypot patrols, edge difficulty increased
      Level 4 — Expert: full adaptive defender, maximum edge difficulty

    Usage:
        env = NetworkEnvLarge()
        state = env.reset()
        state, reward, done, info = env.step(action)
    """

    def __init__(self, seed: int = None,start_level: int=1):
        self.rng = np.random.default_rng(seed)
        self.G   = self._build_graph()

        # Progressive difficulty tracking
        self.cumulative_successes = 0
        self.difficulty_level     = start_level
        self.cumulative_successes = DIFF_THRESHOLDS.get(start_level, 0)
        self._update_edge_difficulties()

        # Episode state
        self.compromised       = np.zeros(NUM_NODES_L, dtype=np.float32)
        self.patched           = np.zeros(NUM_NODES_L, dtype=np.float32)
        self.position          = ENTRY_NODE_L
        self.detection         = 0.0
        self.timestep          = 0
        self.done              = False
        self.probe_results     = {}
        self.episode_reward    = 0.0
        self.nodes_compromised = 0
        self.caught            = False
        self.success           = False
        self.high_alert_steps  = 0
        self.recently_exploited= []
        self.honeypot_triggered= False

        # Credential tracking — VPN requires DMZ
        self.has_dmz_credential = False

        self.stats = {
            "reward": 0.0, "steps": 0, "success": False,
            "caught": False, "timeout": False, "path": [],
            "difficulty_level": 1, "honeypot_triggered": False,
        }

    # ─────────────────────────────────────────
    #  GRAPH BUILDER
    # ─────────────────────────────────────────

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node_id, attrs in NODES_LARGE.items():
            G.add_node(node_id, **attrs)
        for (u, v, difficulty, noise) in EDGES_LARGE:
            G.add_edge(u, v, base_difficulty=difficulty,
                       difficulty=difficulty, noise=noise)
        return G

    def _update_edge_difficulties(self):
        """Scale edge difficulties based on current difficulty level."""
        mult = EDGE_DIFF_MULTIPLIER[self.difficulty_level]
        for u, v, data in self.G.edges(data=True):
            self.G[u][v]["difficulty"] = min(0.95, data["base_difficulty"] * mult)

    def _update_difficulty_level(self):
        """Check if cumulative successes crossed a threshold."""
        if self.difficulty_level >= 2:   # ← ADD THIS LINE
            return
        new_level = 1
        for level, threshold in sorted(DIFF_THRESHOLDS.items(), reverse=True):
            if self.cumulative_successes >= threshold:
                new_level = level
                break
        if new_level != self.difficulty_level:
            self.difficulty_level = new_level
            self._update_edge_difficulties()
            print(f"\n  ⚡ DIFFICULTY LEVEL UP → Level {self.difficulty_level}"
                  f"  (after {self.cumulative_successes} successes)")

    # ─────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.compromised        = np.zeros(NUM_NODES_L, dtype=np.float32)
        self.patched            = np.zeros(NUM_NODES_L, dtype=np.float32)
        self.position           = ENTRY_NODE_L
        self.detection          = 0.0
        self.timestep           = 0
        self.done               = False
        self.probe_results      = {}
        self.episode_reward     = 0.0
        self.nodes_compromised  = 0
        self.caught             = False
        self.success            = False
        self.high_alert_steps   = 0
        self.recently_exploited = []
        self.honeypot_triggered = False
        self.has_dmz_credential = False

        self.compromised[ENTRY_NODE_L] = 1.0

        self.stats = {
            "reward": 0.0, "steps": 0,
            "success": False, "caught": False, "timeout": False,
            "path": [ENTRY_NODE_L], "difficulty_level": self.difficulty_level,
            "honeypot_triggered": False,
        }

        return self._get_state()

    # ─────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────

    def step(self, action: int):
        assert not self.done, "Episode done. Call reset()."
        assert 0 <= action < NUM_ACTIONS_L

        self.timestep += 1
        reward = R_STEP_L

        if action < NUM_NODES_L:
            reward += self._probe(action)
        else:
            reward += self._exploit(action - NUM_NODES_L)

        self.detection = max(0.0, self.detection - DETECTION_DECAY_L)
        self._defender_step()

        done, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.done = done
        self.episode_reward += reward

        self.stats["reward"] = self.episode_reward
        self.stats["steps"]  = self.timestep

        info = {
            "position":           self.position,
            "detection":          self.detection,
            "compromised_nodes":  int(self.compromised.sum()),
            "patched_nodes":      int(self.patched.sum()),
            "success":            self.success,
            "caught":             self.caught,
            "difficulty_level":   self.difficulty_level,
            "honeypot_triggered": self.honeypot_triggered,
            "has_dmz_credential": self.has_dmz_credential,
        }

        return self._get_state(), reward, self.done, info

    # ─────────────────────────────────────────
    #  ACTIONS
    # ─────────────────────────────────────────

    def _probe(self, node: int) -> float:
        self.detection = min(1.0, self.detection + PROBE_NOISE_L)
        self.probe_results[node] = self.G.nodes[node]["vulnerability"]
        return 0.0

    def _exploit(self, node: int) -> float:
        reward = 0.0

        if node == self.position:
            return reward
        if self.compromised[node] == 1.0:
            return reward
        if not self.G.has_edge(self.position, node):
            reward -= 2.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_L)
            return reward
        if self.patched[node] == 1.0:
            reward -= 3.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_L)
            return reward

        # ── Credential check for VPN Gateway ──────────
        if node == VPN_NODE and not self.has_dmz_credential:
            reward -= 5.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_L * 2)
            return reward

        # ── Honeypot mechanic ──────────────────────────
        if NODES_LARGE[node]["is_honeypot"]:
            self.honeypot_triggered = True
            self.stats["honeypot_triggered"] = True
            self.detection = min(1.0, self.detection + 0.85)
            reward += R_HONEYPOT_L
            return reward

        edge       = self.G[self.position][node]
        noise      = edge["noise"]
        difficulty = edge["difficulty"]   # scaled by difficulty level

        noise_amount = EXPLOIT_NOISE_MIN_L + noise * (EXPLOIT_NOISE_MAX_L - EXPLOIT_NOISE_MIN_L)
        self.detection = min(1.0, self.detection + noise_amount)

        vuln         = self.G.nodes[node]["vulnerability"]
        success_prob = max(0.05, vuln - difficulty * 0.5)
        success      = self.rng.random() < success_prob

        if success:
            self.compromised[node] = 1.0
            self.position = node
            self.nodes_compromised += 1
            self.recently_exploited.append(node)
            self.stats["path"].append(node)
            reward += R_NEW_NODE_L
            reward += self.G.nodes[node]["value"] * 0.1

            # Grant DMZ credential
            if node == DMZ_NODE:
                self.has_dmz_credential = True

            # Proximity reward
            try:
                dist = nx.shortest_path_length(self.G, node, TARGET_NODE_L)
                if dist <= 2:
                    reward += max(0, (3 - dist) * 20)
            except nx.NetworkXNoPath:
                pass

        else:
            reward += R_FAILED_EXPLOIT_L

        if self.detection >= DETECTION_DANGER_L:
            reward += R_HIGH_ALERT_L

        return reward

    # ─────────────────────────────────────────
    #  DEFENDER
    # ─────────────────────────────────────────

    def _defender_step(self):
        patch_prob = PATCH_PROB_BY_LEVEL[self.difficulty_level]

        # Reduce patching in first 30 steps
        if self.timestep < 30:
            patch_prob *= 0.3

        protected = [ENTRY_NODE_L, 1,10]

        if self.difficulty_level <= 2:
            # Reactive: patch random exploited node
            exploited = [n for n in range(NUM_NODES_L)
                         if self.compromised[n] == 1.0
                         and self.patched[n] == 0.0
                         and n != self.position
                         and n not in protected]
            if exploited and self.rng.random() < patch_prob:
                self.patched[self.rng.choice(exploited)] = 1.0

        else:
            # Adaptive: patch neighbours of recently exploited nodes
            if self.recently_exploited:
                last       = self.recently_exploited[-1]
                neighbours = list(self.G.successors(last))
                candidates = [n for n in neighbours
                              if self.patched[n] == 0.0
                              and n != self.position
                              and n not in protected]
                if candidates and self.rng.random() < patch_prob:
                    self.patched[self.rng.choice(candidates)] = 1.0

            self.recently_exploited = self.recently_exploited[-3:]

            # Level 3+: random patrol — patches high-value nodes occasionally
            if self.difficulty_level >= 4:
                high_value = [7, 11, 14]
                patrol_candidates = [n for n in high_value
                                     if self.patched[n] == 0.0
                                     and n != self.position]
                if patrol_candidates and self.rng.random() < 0.05:
                    self.patched[self.rng.choice(patrol_candidates)] = 1.0

        # High alert timer
        if self.detection > 0.93:
            self.high_alert_steps += 1
        else:
            self.high_alert_steps = 0

    # ─────────────────────────────────────────
    #  TERMINATION
    # ─────────────────────────────────────────

    def _check_termination(self):
        if self.high_alert_steps >= 40:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT_L

        if self.detection >= DETECTION_CAUGHT_L:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT_L

        if self.position == TARGET_NODE_L:
            self.success = True
            self.stats["success"] = True
            self.cumulative_successes += 1
            self._update_difficulty_level()
            return True, R_TARGET_L

        if self.timestep >= MAX_STEPS_L:
            self.stats["timeout"] = True
            return True, 0.0

        return False, 0.0

    # ─────────────────────────────────────────
    #  STATE
    # ─────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        position_onehot = np.zeros(NUM_NODES_L, dtype=np.float32)
        position_onehot[self.position] = 1.0

        state = np.concatenate([
            self.compromised,
            self.patched,
            position_onehot,
            [np.float32(self.detection)],
            [np.float32(self.timestep / MAX_STEPS_L)],
            [np.float32((self.difficulty_level - 1) / 3)],  # normalised 0–1
        ])
        return state.astype(np.float32)

    # ─────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────

    def get_valid_actions(self) -> list:
        valid      = list(range(NUM_NODES_L))   # all probes
        neighbours = list(self.G.successors(self.position))
        for node in neighbours:
            if self.compromised[node] == 0.0 and self.patched[node] == 0.0:
                # Skip VPN if no credential
                if node == VPN_NODE and not self.has_dmz_credential:
                    continue
                valid.append(node + NUM_NODES_L)
        return list(set(valid))

    def get_node_name(self, node_id: int) -> str:
        return self.G.nodes[node_id]["name"]

    def get_graph(self) -> nx.DiGraph:
        return self.G

    def get_render_state(self) -> dict:
        return {
            "graph":            self.G,
            "position":         self.position,
            "compromised":      self.compromised.copy(),
            "patched":          self.patched.copy(),
            "detection":        self.detection,
            "timestep":         self.timestep,
            "target":           TARGET_NODE_L,
            "entry":            ENTRY_NODE_L,
            "success":          self.success,
            "caught":           self.caught,
            "path":             self.stats["path"].copy(),
            "difficulty_level": self.difficulty_level,
            "honeypot_node":    HONEYPOT_NODE,
            "vpn_node":         VPN_NODE,
            "has_dmz_cred":     self.has_dmz_credential,
            "cumulative_successes": self.cumulative_successes,
        }

    def get_difficulty_info(self) -> dict:
        return {
            "level":            self.difficulty_level,
            "cumulative_wins":  self.cumulative_successes,
            "next_threshold":   DIFF_THRESHOLDS.get(self.difficulty_level + 1, "MAX"),
            "patch_prob":       PATCH_PROB_BY_LEVEL[self.difficulty_level],
            "edge_multiplier":  EDGE_DIFF_MULTIPLIER[self.difficulty_level],
        }

    @property
    def state_size(self):  return STATE_SIZE_L
    @property
    def action_size(self): return NUM_ACTIONS_L
    @property
    def num_nodes(self):   return NUM_NODES_L


# ─────────────────────────────────────────────
#  QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    env   = NetworkEnvLarge(seed=42)
    state = env.reset()

    print(f"State size    : {env.state_size}")
    print(f"Action size   : {env.action_size}")
    print(f"Num nodes     : {env.num_nodes}")
    print(f"State shape   : {state.shape}")
    print(f"Difficulty    : Level {env.difficulty_level}")
    print()

    total_reward = 0
    for step in range(100):
        action = np.random.choice(env.get_valid_actions())
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        atype = "PROBE  " if action < NUM_NODES_L else "EXPLOIT"
        node  = action if action < NUM_NODES_L else action - NUM_NODES_L
        print(f"Step {step+1:3d} | {atype} [{env.get_node_name(node):15s}] "
              f"| Reward: {reward:+7.1f} | Det: {info['detection']:.2f} "
              f"| Pos: {env.get_node_name(info['position'])} "
              f"| Lvl: {info['difficulty_level']}"
              f"{'  ⚠ HONEYPOT' if info['honeypot_triggered'] else ''}")

        if done:
            print()
            if info["success"]:
                print(f"✅  TARGET REACHED! Reward: {total_reward:.1f}  "
                      f"Difficulty: Level {env.difficulty_level}")
            elif info["caught"]:
                print(f"🚨  CAUGHT! Reward: {total_reward:.1f}")
            else:
                print(f"⏱   TIMEOUT. Reward: {total_reward:.1f}")
            break