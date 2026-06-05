# network_env_20.py
# -----------------
# Fixed Pool Network Environment for DDQN Training and Generalisation Testing

import numpy as np
import networkx as nx
import argparse
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MIN_NODES              = 10
MAX_NODES              = 18
ENTRY_NODE_ID          = 0          # always node 0
STATE_NODES    = MAX_NODES
STATE_SIZE_R   = STATE_NODES * 3 + 3   # 57 floats
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


class RandomNetworkEnv:
    """
    Penetration testing environment driven by a stored file containing 20 networks.
    - During TRAINING (seed=None): sequential rotation through the 20 stored networks.
    - During TESTING (seed=0 to 19): loads that specific index out of the file layout.
    """

    def __init__(
        self,
        seed: int       = None,
        num_nodes: int  = None,
        fixed_network: bool = False,
    ):
        self.base_seed    = seed
        self.fixed_num    = num_nodes
        self.fixed_network= fixed_network

        self.G            = None
        self.num_nodes    = 0
        self.target_nodes = []
        self.honeypot_node= None
        self._pool_index  = 0

        # Load file pool from root filesystem execution context
        pool_path = "env/networks/network_pool.pkl"
        if os.path.exists(pool_path):
            with open(pool_path, "rb") as f:
                self._network_pool = pickle.load(f)
        else:
            raise FileNotFoundError(
                f"Missing critical data dependency: '{pool_path}'. "
                f"Please execute 'generate_and_store_pool.py' first to initialize the file."
            )

        # Environment index assignment logic
        if seed is not None and 0 <= seed < 20:
            self._pool_index = seed
            self._load_network_from_pool(self._pool_index)
        else:
            self._pool_index = 0
            self._load_network_from_pool(self._pool_index)

        self._init_episode_state()
        self.stats = {"reward": 0.0, "steps": 0, "success": False, "caught": False, "timeout": False, "path": []}

    def _load_network_from_pool(self, index: int):
        """Loads a static pre-generated topology context out of our file cache."""
        config = self._network_pool[index]
        self.G = config["graph"]
        self.num_nodes = config["num_nodes"]
        self.target_nodes = config["target_nodes"]
        self.honeypot_node = config["honeypot_node"]

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

    def reset(self) -> np.ndarray:
        """
        Reset for new episode.
        If training (seed=None), cleanly advances to the next network configuration in our 20-graph pool.
        """
        if not self.fixed_network and self.base_seed is None:
            # Advance index sequentially across episodes during training
            self._pool_index = (self._pool_index + 1) % 20
            self._load_network_from_pool(self._pool_index)

        self._init_episode_state()
        self.stats = {
            "reward": 0.0, "steps": 0,
            "success": False, "caught": False, "timeout": False,
            "path": [ENTRY_NODE_ID],
        }
        return self._get_state()

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

    def _probe(self, node: int) -> float:
        if node >= self.num_nodes:
            return 0.0
        self.detection = min(1.0, self.detection + PROBE_NOISE_R)
        return 0.0

    def _exploit(self, node: int) -> float:
        reward = 0.0
        if node >= self.num_nodes or node == self.position or self.compromised[node] == 1.0:
            return reward
        if not self.G.has_edge(self.position, node):
            reward -= 2.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_R)
            return reward
        if self.patched[node] == 1.0:
            reward -= 3.0
            self.detection = min(1.0, self.detection + PROBE_NOISE_R)
            return reward

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
            
            min_dist = 999
            for t_idx in self.target_nodes:
                try:
                    dist = nx.shortest_path_length(self.G, node, t_idx)
                    if dist < min_dist:
                        min_dist = dist
                except nx.NetworkXNoPath:
                    pass
            if min_dist <= 2:
                reward += max(0, (3 - min_dist) * 20)
        else:
            reward += R_FAILED_EXPLOIT_R

        if self.detection >= DETECTION_DANGER_R:
            reward += R_HIGH_ALERT_R

        return reward

    def _defender_step(self):
        patch_prob = 0.12
        if self.timestep < 30:
            patch_prob *= 0.3

        exploited = [
            n for n in range(self.num_nodes)
            if self.compromised[n] == 1.0 and self.patched[n] == 0.0 and n != self.position and n != ENTRY_NODE_ID
        ]
        if exploited and np.random.random() < patch_prob:
            self.patched[np.random.choice(exploited)] = 1.0

        if self.detection > 0.93:
            self.high_alert_steps += 1
        else:
            self.high_alert_steps = 0

    def _check_termination(self):
        if self.high_alert_steps >= 25 or self.detection >= DETECTION_CAUGHT_R:
            self.caught = True
            self.stats["caught"] = True
            return True, R_CAUGHT_R
        if self.position in self.target_nodes:
            self.success = True
            self.stats["success"] = True
            return True, R_TARGET_R
        if self.timestep >= MAX_STEPS_R:
            self.stats["timeout"] = True
            return True, 0.0
        return False, 0.0

    def _get_state(self) -> np.ndarray:
        pos_onehot = np.zeros(MAX_NODES, dtype=np.float32)
        if self.position < MAX_NODES:
            pos_onehot[self.position] = 1.0

        compromised_state = self.compromised[:MAX_NODES].copy()
        patched_state     = self.patched[:MAX_NODES].copy()

        if len(self.target_nodes) >= 1:
            t1 = self.target_nodes[0]
            if t1 != MAX_NODES - 2:
                compromised_state[MAX_NODES - 2] = compromised_state[t1]
                patched_state[MAX_NODES - 2]     = patched_state[t1]
                
        if len(self.target_nodes) == 2:
            t2 = self.target_nodes[1]
            if t2 != MAX_NODES - 1:
                compromised_state[MAX_NODES - 1] = compromised_state[t2]
                patched_state[MAX_NODES - 1]     = patched_state[t2]

        state = np.concatenate([
            compromised_state,
            patched_state,
            pos_onehot,
            [np.float32(self.detection)],
            [np.float32(self.timestep / MAX_STEPS_R)],
            [np.float32(self.num_nodes / MAX_NODES)],
        ])
        return state.astype(np.float32)

    def get_valid_actions(self) -> list:
        valid = list(range(self.num_nodes))
        neighbours = list(self.G.successors(self.position))
        for node in neighbours:
            if node < MAX_NODES and self.compromised[node] == 0.0 and self.patched[node] == 0.0:
                valid.append(node + MAX_NODES)
        return valid 
        
    def get_node_name(self, node_id: int) -> str:
        if self.G and node_id < self.num_nodes:
            return self.G.nodes[node_id].get("name", f"Node {node_id}")
        return f"Node {node_id}"

    def get_graph(self) -> nx.DiGraph: 
        return self.G

    # ADDED METHOD FOR VISUALISER INTERACTION
    def get_render_state(self) -> dict:
        """Returns the dictionary tracking metrics needed by the Phase 1 and 2 UI layout."""
        return {
            "graph":       self.G,
            "position":    self.position,
            "compromised": self.compromised.copy(),
            "patched":     self.patched.copy(),
            "detection":   self.detection,
            "timestep":    self.timestep,
            "targets":     self.target_nodes,
            "entry":       ENTRY_NODE_ID,
            "success":     self.success,
            "caught":      self.caught,
            "path":        self.stats["path"].copy(),
        }

    @property
    def state_size(self):  return STATE_SIZE_R
    @property
    def action_size(self): return NUM_ACTIONS_R


# ─────────────────────────────────────────────
#  GENERALISATION TEST ENTRY POINT
# ─────────────────────────────────────────────
def run_generalisation_test(model_path: str, n_test_networks: int = 20):
    from agent.ddqn_agent import DDQNAgent

    print(f"\n{'='*60}")
    print(f"  EVALUATION TEST — Stored 20-Graph Persistent Pool Verification")
    print(f"  Model Target: {model_path}")
    print(f"{'='*60}\n")

    dummy_env = RandomNetworkEnv(seed=0)
    agent     = DDQNAgent(state_size=dummy_env.state_size, action_size=dummy_env.action_size)

    if os.path.exists(model_path):
        agent.load(model_path, partial_transfer=True)
        agent.epsilon = 0.0
    else:
        print(f"  Model not found: {model_path}. Running with uninitialized baseline agent.\n")

    agent.epsilon = 0.0
    results = []

    for i in range(min(20, n_test_networks)):
        env   = RandomNetworkEnv(seed=i, fixed_network=True)
        state = env.reset()

        for _ in range(MAX_STEPS_R):
            valid  = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid)
            state, _, done, info = env.step(action)
            if done:
                break

        results.append({"index": i, "nodes": env.num_nodes, "success": env.success, "caught": env.caught})
        status = "✅" if env.success else "🚨" if env.caught else "⏱"
        print(f"  Pool Network {i:2d} (nodes={env.num_nodes:2d}): {status} steps={env.timestep:3d}")

    n_success = sum(1 for r in results if r["success"])
    print(f"\nEvaluation Complete: Score = {n_success}/{len(results)} matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run generalisation test")
    parser.add_argument("--model", default="results_random/checkpoints/ddqn_final.pt", help="Model path")
    parser.add_argument("--networks", type=int, default=20, help="Number of test networks")
    args = parser.parse_args()

    if args.test:
        run_generalisation_test(args.model, args.networks)
    else:
        print("RANDOM NETWORK GENERATOR — File Cache Verification Demo\n")
        env = RandomNetworkEnv(seed=None)
        for trial in range(3):
            state = env.reset()
            print(f"Training Reset Trigger {trial+1} (Pulled Pool Index: {env._pool_index}): Nodes Count = {env.num_nodes}")