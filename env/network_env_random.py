"""
random_network_env.py
---------------------
Random Network Generator for DDQN Generalisation Testing — Stage 5.

Generates unseen corporate network topologies at runtime with randomised:
  - Node vulnerability scores
  - Edge difficulties and noise levels
  - Network topology (within realistic constraints)
  - Number of nodes (10–18)

Guarantees:
  - Always at least one valid path from entry to target
  - Target node always exists and is reachable
  - No isolated nodes
  - Honeypot node included (random position)
  - Realistic topology (DMZ zone, internal zone, target zone)

Usage:
    # Training on random networks (generalisation training)
    env = RandomNetworkEnv(seed=None)   # new random network each reset
    state = env.reset()

    # Testing on fixed unseen networks
    env = RandomNetworkEnv(seed=999)    # reproducible unseen network
    state = env.reset()

    # Generalisation test
    python random_network_env.py --test --model results_large/ddqn_final.pt
"""

import numpy as np
import networkx as nx
import argparse
import os
import sys
from network_env_large import EDGES_LARGE, NODES_LARGE, NUM_NODES_L

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

MIN_NODES              = 10
MAX_NODES              = 18
TARGET_NODE_ID         = "target"   # always last node
ENTRY_NODE_ID          = 0          # always node 0
BACKBONE_INHERIT_RATIO = 0.95
# State size — fixed regardless of network size
# We pad/truncate to MAX_NODES for consistent state vector
STATE_NODES    = MAX_NODES
STATE_SIZE_R   = STATE_NODES * 3 + 3   # 57 floats

# Action space — fixed at MAX_NODES × 2
NUM_ACTIONS_R  = MAX_NODES * 2          # 36

MAX_STEPS_R    = 200

# Rewards
R_TARGET_R         = +800
R_NEW_NODE_R       = +200
R_FAILED_EXPLOIT_R = -4
R_HIGH_ALERT_R     = -7
R_CAUGHT_R         = -200
R_STEP_R           = -10
R_HONEYPOT_R       = -50

DETECTION_DANGER_R  = 0.8
DETECTION_CAUGHT_R  = 1.0
PROBE_NOISE_R       = 0.06
EXPLOIT_NOISE_MIN_R = 0.05
EXPLOIT_NOISE_MAX_R = 0.12
DETECTION_DECAY_R   = 0.08

# Node type names for readable output
NODE_TYPE_NAMES = [
    "Web Server", "App Server", "Mail Server", "Dev Box",
    "File Server", "DB Backup", "Admin Panel", "Dev Server",
    "DMZ Server", "VPN Gateway", "Cloud Node", "Proxy Server",
    "Build Server", "Auth Server", "Log Server", "Backup Node",
    "Internal DB",   # always target
]


class RandomNetworkEnv:
    """
    Penetration testing environment with randomly generated network topology.

    Each call to reset() can optionally regenerate the network (if seed=None),
    creating a new unseen topology for the agent to navigate.

    For generalisation TRAINING: use seed=None (new network each episode)
    For generalisation TESTING:  use specific seeds (reproducible networks)
    """

    def __init__(
        self,
        seed: int       = None,
        num_nodes: int  = None,
        fixed_network: bool = False,
    ):
        """
        Args:
            seed         : random seed. None = new random network each reset().
                           Integer = reproducible fixed network.
            num_nodes    : number of nodes. None = random between MIN and MAX.
            fixed_network: if True, regenerate network only on __init__,
                           not on each reset(). Good for testing.
        """
        self.base_seed    = seed
        self.fixed_num    = num_nodes
        self.fixed_network= fixed_network

        # Will be set by _generate_network()
        self.G            = None
        self.num_nodes    = 0
        self.target_node  = 0
        self.honeypot_node= None
        self._episode_rng = np.random.default_rng(seed)

        # Generate initial network
        self._generate_network(seed)

        # Episode state
        self._init_episode_state()

        self.stats = {
            "reward": 0.0, "steps": 0,
            "success": False, "caught": False, "timeout": False,
            "path": [],
        }

    # ─────────────────────────────────────────
    #  NETWORK GENERATOR
    # ─────────────────────────────────────────
    def _generate_network(self, seed=None):
        rng = np.random.default_rng(seed)

        n = (self.fixed_num if self.fixed_num
            else int(rng.integers(MIN_NODES, MAX_NODES + 1)))
        self.num_nodes = n

        G = nx.DiGraph()

        target_idx   = n - 1
        # FIX 3: keep honeypot away from high-traffic fixed nodes (8,10,12,14)
        # to avoid excluding large edge clusters
        bad_honeypot_slots = {8, 10, 12, 13, 14}
        honeypot_candidates = [
            i for i in range(2, max(3, n - 2))
            if i not in bad_honeypot_slots
        ]
        if not honeypot_candidates:
            honeypot_candidates = list(range(2, max(3, n - 2)))
        honeypot_idx = int(rng.choice(honeypot_candidates))

        # Node properties (unchanged)
        fixed_node_names = {i: NODES_LARGE[i]["name"] for i in range(min(n, NUM_NODES_L))}
        for i in range(n):
            if i == 0:
                name, vuln, val, is_hp = "Entry Point", float(rng.uniform(0.85, 0.98)), 0, False
            elif i == target_idx:
                name, vuln, val, is_hp = "Internal DB", float(rng.uniform(0.15, 0.35)), 100, False
            elif i == honeypot_idx:
                name, vuln, val, is_hp = "Honeypot", float(rng.uniform(0.80, 0.95)), 0, True
            else:
                name = fixed_node_names.get(i, NODE_TYPE_NAMES[min(i, len(NODE_TYPE_NAMES)-2)])
                vuln, val, is_hp = float(rng.uniform(0.25, 0.85)), int(rng.integers(5, 35)), False
            G.add_node(i, name=name, vulnerability=round(vuln, 2),
                    value=val, is_honeypot=is_hp)

        # ── Step 1: Inherit backbone edges ───────────────────────
        valid_fixed_edges = [
            (u, v, diff, noise)
            for (u, v, diff, noise) in EDGES_LARGE
            if u < n and v < n
            and u != honeypot_idx
            and v != honeypot_idx        # still exclude honeypot edges
            # FIX 2: DO NOT exclude target edges — agent needs this knowledge
        ]

        LARGE_TARGET_IDX = NUM_NODES_L - 1  # = 8

        # Separate edges that approach the large-graph target (node 8)
        target_approach = [(u, v, d, ns) for u, v, d, ns in valid_fixed_edges if v == LARGE_TARGET_IDX]
        regular_edges   = [(u, v, d, ns) for u, v, d, ns in valid_fixed_edges if v != LARGE_TARGET_IDX]

        # Shuffle regular edges and take backbone ratio of them
        indices = list(range(len(regular_edges)))
        rng.shuffle(indices)
        # Use instance variable so train.py can anneal it
        ratio = getattr(self, 'backbone_ratio', BACKBONE_INHERIT_RATIO)
        n_inherit = max(int(len(regular_edges) * ratio), min(3, len(regular_edges)))
        inherited_regular = [regular_edges[i] for i in indices[:n_inherit]]

        # Combine: all target-approach + sampled regular
        inherited_edges = target_approach + inherited_regular

        for u, v, base_diff, base_noise in inherited_edges:
            actual_v = target_idx if v == LARGE_TARGET_IDX else v
            if actual_v >= n:
                continue
            diff  = float(np.clip(base_diff  + rng.uniform(-0.10, 0.10), 0.05, 0.90))
            noise = float(np.clip(base_noise + rng.uniform(-0.08, 0.08), 0.03, 0.60))
            G.add_edge(u, actual_v,
                    base_difficulty=round(diff, 2),
                    difficulty=round(diff, 2),
                    noise=round(noise, 2))

        # ── Step 2: Backbone path guarantee (unchanged) ──────────
        if not nx.has_path(G, 0, target_idx):
            backbone = self._build_backbone(n, target_idx, honeypot_idx, rng)
            for u, v in backbone:
                if not G.has_edge(u, v):
                    diff  = float(rng.uniform(0.15, 0.55))
                    noise = float(rng.uniform(0.08, 0.40))
                    G.add_edge(u, v, base_difficulty=round(diff, 2),
                            difficulty=round(diff, 2), noise=round(noise, 2))

        # ── Step 3: Random cross-edges for variety ────────────
        n_cross  = int(rng.integers(n // 4, n // 2))
        attempts = 0
        while attempts < n_cross * 3:
            u = int(rng.integers(0, n - 1))
            v = int(rng.integers(u + 1, n))
            if (not G.has_edge(u, v)
                    and v != honeypot_idx
                    and u != honeypot_idx):
                diff  = float(rng.uniform(0.20, 0.70))
                noise = float(rng.uniform(0.10, 0.50))
                G.add_edge(u, v,
                        base_difficulty=round(diff, 2),
                        difficulty=round(diff, 2),
                        noise=round(noise, 2))
            attempts += 1

        # ── Step 4: Honeypot trap edges ───────────────────────
        # Give honeypot 1-2 attractive-looking edges from nearby nodes
        hp_sources = [0] + [
            i for i in range(1, min(4, n))
            if i != honeypot_idx and G.has_edge(0, i)
        ]
        for src in hp_sources[:2]:
            if not G.has_edge(src, honeypot_idx):
                G.add_edge(src, honeypot_idx,
                        base_difficulty=0.05,
                        difficulty=0.05,
                        noise=0.03)

        # ── Step 5: Ensure no isolated non-entry nodes ────────
        for i in range(1, n):
            if G.in_degree(i) == 0 and i != honeypot_idx:
                # Connect from a random earlier node
                src  = int(rng.integers(0, i))
                diff  = float(rng.uniform(0.20, 0.60))
                noise = float(rng.uniform(0.10, 0.40))
                G.add_edge(src, i,
                        base_difficulty=round(diff, 2),
                        difficulty=round(diff, 2),
                        noise=round(noise, 2))

        self.G            = G
        self.target_node  = target_idx
        self.honeypot_node= honeypot_idx

        # Final reachability guarantee
        if not nx.has_path(G, 0, target_idx):
            G.add_edge(0, target_idx,
                    base_difficulty=0.70,
                    difficulty=0.70,
                    noise=0.50)
        

    def _build_backbone(self, n, target_idx, honeypot_idx, rng):
        """Build a guaranteed path from 0 to target, avoiding honeypot."""
        # Create a path through non-honeypot nodes
        middle_nodes = [i for i in range(1, n - 1) if i != honeypot_idx]
        rng.shuffle(middle_nodes)

        # Pick 2-4 intermediate hops
        n_hops = min(len(middle_nodes), int(rng.integers(2, 5)))
        path   = [0] + list(middle_nodes[:n_hops]) + [target_idx]

        edges = []
        for i in range(len(path) - 1):
            edges.append((path[i], path[i+1]))
        return edges

    # ─────────────────────────────────────────
    #  EPISODE STATE
    # ─────────────────────────────────────────

    def _init_episode_state(self):
        self.compromised        = np.zeros(MAX_NODES, dtype=np.float32)
        self.patched            = np.zeros(MAX_NODES, dtype=np.float32)
        self.position           = ENTRY_NODE_ID
        self.detection          = 0.0
        self.timestep           = 0
        self.done               = False
        self.episode_reward     = 0.0
        self.caught             = False
        self.success            = False
        self.high_alert_steps   = 0
        self.recently_exploited = []
        self.honeypot_triggered = False
        self.compromised[ENTRY_NODE_ID] = 1.0

    # ─────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """
        Reset for new episode.
        If not fixed_network, generates a NEW random network topology.
        """
        if not self.fixed_network and self.base_seed is None:
            # New random network every episode
            new_seed = int(self._episode_rng.integers(0, 999999))
            self._generate_network(new_seed)

        self._init_episode_state()
        self.stats = {
            "reward": 0.0, "steps": 0,
            "success": False, "caught": False, "timeout": False,
            "path": [ENTRY_NODE_ID],
        }
        return self._get_state()

    # ─────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────

    def step(self, action: int):
        assert not self.done
        assert 0 <= action < NUM_ACTIONS_R

        self.timestep += 1
        reward = R_STEP_R

        if action < MAX_NODES:
            reward += self._probe(action)
        else:
            reward += self._exploit(action - MAX_NODES)

        self.detection = max(0.0, self.detection - DETECTION_DECAY_R)
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
            "compromised_nodes":  int((self.compromised[:self.num_nodes] > 0).sum()),
            "success":            self.success,
            "caught":             self.caught,
            "num_nodes":          self.num_nodes,
            "honeypot_triggered": self.honeypot_triggered,
        }
        return self._get_state(), reward, self.done, info

    # ─────────────────────────────────────────
    #  PROBE / EXPLOIT
    # ─────────────────────────────────────────

    def _probe(self, node: int) -> float:
        if node >= self.num_nodes:
            return 0.0
        self.detection = min(1.0, self.detection + PROBE_NOISE_R)
        return 0.0

    def _exploit(self, node: int) -> float:
        reward = 0.0

        if node >= self.num_nodes:
            return reward
        if node == self.position:
            return reward
        if self.compromised[node] == 1.0:
            return reward
        if not self.G.has_edge(self.position, node):
            reward -= 2.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_R)
            return reward
        if self.patched[node] == 1.0:
            reward -= 3.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_R)
            return reward

        # Honeypot
        if self.G.nodes[node].get("is_honeypot", False):
            self.honeypot_triggered = True
            self.detection = min(1.0, self.detection + 0.85)
            reward += R_HONEYPOT_R
            return reward

        edge       = self.G[self.position][node]
        noise      = edge["noise"]
        difficulty = edge["difficulty"]
        noise_amt  = EXPLOIT_NOISE_MIN_R + noise * (EXPLOIT_NOISE_MAX_R - EXPLOIT_NOISE_MIN_R)
        self.detection = min(1.0, self.detection + noise_amt)

        vuln         = self.G.nodes[node]["vulnerability"]
        success_prob = max(0.05, vuln - difficulty * 0.5)
        success      = np.random.random() < success_prob

        if success:
            self.compromised[node] = 1.0
            self.position          = node
            self.recently_exploited.append(node)
            self.stats["path"].append(node)
            reward += R_NEW_NODE_R
            reward += self.G.nodes[node]["value"] * 0.1
            try:
                dist = nx.shortest_path_length(self.G, node, self.target_node)
                if dist <= 2:
                    reward += max(0, (3 - dist) * 20)
            except nx.NetworkXNoPath:
                pass
        else:
            reward += R_FAILED_EXPLOIT_R

        if self.detection >= DETECTION_DANGER_R:
            reward += R_HIGH_ALERT_R

        return reward

    # ─────────────────────────────────────────
    #  DEFENDER
    # ─────────────────────────────────────────

    def _defender_step(self):
        patch_prob = 0.12
        if self.timestep < 30:
            patch_prob *= 0.3

        exploited = [
            n for n in range(self.num_nodes)
            if self.compromised[n] == 1.0
            and self.patched[n]    == 0.0
            and n != self.position
            and n != ENTRY_NODE_ID
        ]
        if exploited and np.random.random() < patch_prob:
            self.patched[np.random.choice(exploited)] = 1.0

        if self.detection > 0.93:
            self.high_alert_steps += 1
        else:
            self.high_alert_steps = 0

    # ─────────────────────────────────────────
    #  TERMINATION
    # ─────────────────────────────────────────

    def _check_termination(self):
        if self.high_alert_steps >= 25:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT_R
        if self.detection >= DETECTION_CAUGHT_R:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT_R
        if self.position == self.target_node:
            self.success = True
            self.stats["success"] = True
            return True, R_TARGET_R
        if self.timestep >= MAX_STEPS_R:
            self.stats["timeout"] = True
            return True, 0.0
        return False, 0.0

    # ─────────────────────────────────────────
    #  STATE
    # ─────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Fixed-size state vector regardless of actual network size.
        Pads with zeros for unused node slots.
        Positions active nodes at indices 0..num_nodes-1.
        Target node always at index MAX_NODES-1 in state vector
        so the agent can always find it regardless of network size.
        """
        pos_onehot = np.zeros(MAX_NODES, dtype=np.float32)
        if self.position < MAX_NODES:
            pos_onehot[self.position] = 1.0

        # Remap target to fixed slot MAX_NODES-1
        compromised_state = self.compromised[:MAX_NODES].copy()
        patched_state     = self.patched[:MAX_NODES].copy()

        # Ensure target is always visible at last slot
        if self.target_node != MAX_NODES - 1:
            # Swap target info to last slot
            compromised_state[MAX_NODES-1] = compromised_state[self.target_node]
            patched_state[MAX_NODES-1]     = patched_state[self.target_node]

        state = np.concatenate([
            compromised_state,
            patched_state,
            pos_onehot,
            [np.float32(self.detection)],
            [np.float32(self.timestep / MAX_STEPS_R)],
            [np.float32(self.num_nodes / MAX_NODES)],  # network size hint
        ])
        return state.astype(np.float32)

    # ─────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────

    def get_valid_actions(self) -> list:
        valid = list(range(self.num_nodes))  # all probes
        neighbours = list(self.G.successors(self.position))
        for node in neighbours:
            if (node < MAX_NODES
                    and self.compromised[node] == 0.0
                    and self.patched[node]    == 0.0):
                valid.append(node + MAX_NODES)
        return valid 
    def get_node_name(self, node_id: int) -> str:
        if self.G and node_id < self.num_nodes:
            return self.G.nodes[node_id].get("name", f"Node {node_id}")
        return f"Node {node_id}"

    def get_graph(self)        -> nx.DiGraph: return self.G
    def get_render_state(self) -> dict:
        return {
            "graph":       self.G,
            "position":    self.position,
            "compromised": self.compromised.copy(),
            "patched":     self.patched.copy(),
            "detection":   self.detection,
            "timestep":    self.timestep,
            "target":      self.target_node,
            "entry":       ENTRY_NODE_ID,
            "success":     self.success,
            "caught":      self.caught,
            "path":        self.stats["path"].copy(),
        }

    @property
    def state_size(self):  return STATE_SIZE_R
    @property
    def action_size(self): return NUM_ACTIONS_R
    @property
    def num_nodes_prop(self): return self.num_nodes


# ─────────────────────────────────────────────
#  GENERALISATION TEST
# ─────────────────────────────────────────────

def run_generalisation_test(model_path: str, n_test_networks: int = 20):
    """
    Test a trained agent on N unseen random networks.
    Prints per-network results and overall generalisation score.
    """
    from agent.ddqn_agent import DDQNAgent

    print(f"\n{'='*60}")
    print(f"  GENERALISATION TEST — {n_test_networks} unseen networks")
    print(f"  Model: {model_path}")
    print(f"{'='*60}\n")

    # Load agent — uses random env's state/action sizes
    dummy_env = RandomNetworkEnv(seed=0)
    agent     = DDQNAgent(
        state_size  = dummy_env.state_size,
        action_size = dummy_env.action_size,
    )

    if os.path.exists(model_path):
        agent.load(model_path, partial_transfer=True)  # same behaviour, more explicit
        print(f"Epsilon after load: {agent.epsilon}")
        agent.epsilon = 0.0
        print(f"Epsilon after override: {agent.epsilon}")
    else:
        print(f"  Model not found: {model_path}")
        print(f"  Running with untrained agent for baseline.\n")

    agent.epsilon = 0.0   # pure exploitation

    results = []

    # Test seeds — chosen to be far from training seeds
    test_seeds = [1000 + i * 37 for i in range(n_test_networks)]

    for i, seed in enumerate(test_seeds):
        env   = RandomNetworkEnv(seed=seed, fixed_network=True)
        state = env.reset()

        for _ in range(MAX_STEPS_R):
            valid  = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid)
            state, _, done, info = env.step(action)
            if done:
                break

        results.append({
            "seed":      seed,
            "nodes":     env.num_nodes,
            "success":   env.success,
            "caught":    env.caught,
            "steps":     env.timestep,
            "path_len":  len(env.stats["path"]),
        })

        status = "✅" if env.success else "🚨" if env.caught else "⏱"
        print(f"  Network {i+1:2d} (seed={seed}, nodes={env.num_nodes:2d}): "
              f"{status}  steps={env.timestep:3d}  "
              f"path_len={len(env.stats['path'])}")

    # Summary
    n_success = sum(1 for r in results if r["success"])
    n_caught  = sum(1 for r in results if r["caught"])
    n_timeout = sum(1 for r in results if not r["success"] and not r["caught"])

    print(f"\n{'='*60}")
    print(f"  GENERALISATION RESULTS")
    print(f"{'='*60}")
    print(f"  Success : {n_success}/{n_test_networks}  ({n_success/n_test_networks*100:.0f}%)")
    print(f"  Caught  : {n_caught}/{n_test_networks}  ({n_caught/n_test_networks*100:.0f}%)")
    print(f"  Timeout : {n_timeout}/{n_test_networks}  ({n_timeout/n_test_networks*100:.0f}%)")
    print(f"\n  Generalisation Score: {n_success/n_test_networks*100:.0f}%")

    if n_success / n_test_networks >= 0.40:
        print(f"\n  ✅ GENERALISES WELL — agent learned a general attack strategy")
    elif n_success / n_test_networks >= 0.20:
        print(f"\n  🟡 PARTIAL GENERALISATION — some transfer, not robust")
    else:
        print(f"\n  ❌ POOR GENERALISATION — agent memorised specific topology")
    print(f"{'='*60}\n")

    return n_success / n_test_networks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",     action="store_true",
                        help="Run generalisation test")
    parser.add_argument("--model",    default="results_random/checkpoints/ddqn_ep00500.pt",
                        help="Model path for testing")
    parser.add_argument("--networks", type=int, default=20,
                        help="Number of test networks")
    parser.add_argument("--nodes",    type=int, default=None,
                        help="Fix number of nodes (default: random)")
    args = parser.parse_args()

    if args.test:
        run_generalisation_test(args.model, args.networks)
    else:
        # Demo: generate and show 3 random networks
        print("RANDOM NETWORK GENERATOR — Demo\n")
        for trial in range(3):
            seed = trial * 100
            env  = RandomNetworkEnv(seed=seed)
            state = env.reset()

            print(f"Network {trial+1} (seed={seed}):")
            print(f"  Nodes     : {env.num_nodes}")
            print(f"  Target    : node {env.target_node} "
                  f"({env.get_node_name(env.target_node)})")
            print(f"  Honeypot  : node {env.honeypot_node} "
                  f"({env.get_node_name(env.honeypot_node)})")
            print(f"  Edges     : {env.G.number_of_edges()}")
            print(f"  State size: {env.state_size}")

            # Show all paths
            paths = list(nx.all_simple_paths(
                env.G, 0, env.target_node, cutoff=6))
            print(f"  Paths to target ({len(paths)} found):")
            for p in sorted(paths, key=len)[:3]:
                names = [env.get_node_name(n) for n in p]
                print(f"    {' → '.join(names)}")
            print()