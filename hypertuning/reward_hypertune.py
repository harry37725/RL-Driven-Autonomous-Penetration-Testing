"""
reward_hypertune.py
-------------------
Automated reward hypertuning for the Pentest DDQN agent.

Runs 3 reward experiments in sequence, each for N episodes:
  Exp 1 — R_TARGET vs R_NEW_NODE ratio  (3 configs)
  Exp 2 — R_CAUGHT penalty              (3 configs)
  Exp 3 — Distance reward shape         (3 configs)

After all experiments, generates a full comparison dashboard:
  - Success % curves per experiment group
  - Caught % curves per experiment group
  - Reward curves per experiment group
  - Summary bar chart — best config per group
  - Heatmap — all configs ranked

Run:
    python reward_hypertune.py

Results saved to:
    results/reward_tuning/
        plots/
            group1_target_vs_node.png
            group2_caught_penalty.png
            group3_distance_shape.png
            summary_dashboard.png
        logs/
            experiment_log.csv
"""

import os
import sys
import copy
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from env.network_env  import NetworkEnv, NUM_NODES
from agent.ddqn_agent import DDQNAgent


# ─────────────────────────────────────────────
#  TUNING CONFIG
# ─────────────────────────────────────────────

EPISODES_PER_CONFIG = 500   # episodes per config — increase for more reliable results
ROLLING_WINDOW      = 30     # rolling average window for plots
SAVE_DIR            = "results/reward_tuning"

# Base agent config — same for all experiments
AGENT_CFG = {
    "lr":                 1e-3,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      0.9995,
    "batch_size":         64,
    "buffer_capacity":    50_000,
    "target_update_freq": 200,
}

# ─────────────────────────────────────────────
#  EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────

EXPERIMENTS = {

    # ── Group 1: R_TARGET vs R_NEW_NODE ratio ──────────────────
    "group1_target_ratio": {
        "title":       "Group 1 — R_TARGET vs R_NEW_NODE Ratio",
        "description": "How strongly should reaching the target be rewarded vs compromising nodes?",
        "configs": [
            {
                "label":      "Balanced (500/50 = 10:1)",
                "color":      "#63B3ED",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -200,
                "dist_shape": "current",
            },
            {
                "label":      "Target-heavy (800/30 = 27:1)",
                "color":      "#68D391",
                "R_TARGET":   800,
                "R_NEW_NODE": 30,
                "R_CAUGHT":   -200,
                "dist_shape": "current",
            },
            {
                "label":      "Node-heavy (300/80 = 4:1)",
                "color":      "#F6AD55",
                "R_TARGET":   300,
                "R_NEW_NODE": 80,
                "R_CAUGHT":   -200,
                "dist_shape": "current",
            },
        ]
    },

    # ── Group 2: R_CAUGHT penalty ───────────────────────────────
    "group2_caught_penalty": {
        "title":       "Group 2 — R_CAUGHT Penalty",
        "description": "How much should the agent fear getting caught?",
        "configs": [
            {
                "label":      "Current (-200)",
                "color":      "#63B3ED",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -200,
                "dist_shape": "current",
            },
            {
                "label":      "Fearful (-350)",
                "color":      "#FC8181",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -350,
                "dist_shape": "current",
            },
            {
                "label":      "Reckless (-100)",
                "color":      "#F6E05E",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -100,
                "dist_shape": "current",
            },
        ]
    },

    # ── Group 3: Distance reward shape ─────────────────────────
    "group3_distance_shape": {
        "title":       "Group 3 — Distance Reward Shape",
        "description": "How far from the target should the agent get a reward signal?",
        "configs": [
            {
                "label":      "Current (≤2 hops)",
                "color":      "#63B3ED",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -200,
                "dist_shape": "current",   # 2 hops=+20, 1 hop=+40
            },
            {
                "label":      "Extended (≤3 hops)",
                "color":      "#B794F4",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -200,
                "dist_shape": "extended",  # 3 hops=+15, 2 hops=+30, 1 hop=+45
            },
            {
                "label":      "No distance reward",
                "color":      "#FC8181",
                "R_TARGET":   500,
                "R_NEW_NODE": 50,
                "R_CAUGHT":   -200,
                "dist_shape": "none",      # only terminal reward
            },
        ]
    },
}


# ─────────────────────────────────────────────
#  PATCHED ENVIRONMENT
# ─────────────────────────────────────────────

class PatchedEnv(NetworkEnv):
    """
    NetworkEnv with reward values overridden per experiment config.
    Also supports different distance reward shapes.
    """

    def __init__(self, cfg: dict, seed: int = 42):
        super().__init__(defender_level=1, seed=seed)
        self.cfg = cfg

    def _exploit(self, node: int) -> float:
        """Override _exploit to use experiment-specific reward values."""
        import networkx as nx

        reward = 0.0

        if node == self.position:
            return reward
        if self.compromised[node] == 1.0:
            return reward
        if not self.G.has_edge(self.position, node):
            reward -= 2.0
            self.detection = min(1.0, self.detection + 0.06)
            return reward
        if self.patched[node] == 1.0:
            reward -= 3.0
            self.detection = min(1.0, self.detection + 0.06)
            return reward

        edge       = self.G[self.position][node]
        noise      = edge["noise"]
        difficulty = edge["difficulty"]

        noise_amount = 0.05 + noise * (0.12 - 0.05)
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

            # ── Experiment reward values ──
            reward += self.cfg["R_NEW_NODE"]
            reward += self.G.nodes[node]["value"] * 0.1

            # ── Distance reward shape ──
            try:
                from env.network_env import TARGET_NODE
                dist = nx.shortest_path_length(self.G, node, TARGET_NODE)
                shape = self.cfg["dist_shape"]

                if shape == "current":
                    if dist <= 2:
                        reward += max(0, (3 - dist) * 20)
                elif shape == "extended":
                    if dist <= 3:
                        reward += max(0, (4 - dist) * 15)
                elif shape == "none":
                    pass  # no distance reward

            except Exception:
                pass

        else:
            reward += -4  # R_FAILED_EXPLOIT

        if self.detection >= 0.8:
            reward += -7  # R_HIGH_ALERT

        return reward

    def _check_termination(self):
        from env.network_env import TARGET_NODE
        if self.high_alert_steps >= 25:
            self.caught = True
            self.stats["caught"] = True
            return True, self.cfg["R_CAUGHT"]

        if self.detection >= 1.0:
            self.caught = True
            self.stats["caught"] = True
            return True, self.cfg["R_CAUGHT"]

        if self.position == TARGET_NODE:
            self.success = True
            self.stats["success"] = True
            return True, self.cfg["R_TARGET"]

        if self.timestep >= 100:
            self.stats["timeout"] = True
            return True, 0.0

        return False, 0.0


# ─────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────

def run_config(cfg: dict, episodes: int, seed: int = 42) -> dict:
    """
    Run one reward config for N episodes.
    Returns history dict with rewards, successes, caught flags.
    """
    env   = PatchedEnv(cfg, seed=seed)
    agent = DDQNAgent(
        state_size         = env.state_size,
        action_size        = env.action_size,
        **AGENT_CFG
    )

    history = {
        "rewards":   [],
        "successes": [],
        "caught":    [],
        "steps":     [],
    }

    for ep in range(episodes):
        state     = env.reset()
        ep_reward = 0.0

        for _ in range(100):
            valid      = env.get_valid_actions()
            action     = agent.select_action(state, valid_actions=valid)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            ep_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()

        history["rewards"].append(ep_reward)
        history["successes"].append(1 if env.stats["success"] else 0)
        history["caught"].append(1 if env.stats["caught"] else 0)
        history["steps"].append(env.stats["steps"])

    return history


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

C_BG    = "#1A202C"
C_PANEL = "#2D3748"
C_TEXT  = "#E2E8F0"
C_MUTED = "#718096"


def rolling(data, window=30):
    arr = np.array(data, dtype=np.float32)
    w   = min(window, len(arr))
    if w < 2:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def style_ax(ax, title="", xlabel="Episode", ylabel=""):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#4A5568")
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=8)


def plot_group(group_key: str, group_data: dict, results: dict, save_path: str):
    """Plot 3 subplots for one experiment group: reward, success, caught."""
    configs = group_data["configs"]
    title   = group_data["title"]
    desc    = group_data["description"]

    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    fig.suptitle(
        f"{title}\n{desc}",
        color=C_TEXT, fontsize=12, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_reward  = fig.add_subplot(gs[0, :])   # full width top
    ax_success = fig.add_subplot(gs[1, 0])
    ax_caught  = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    eps = list(range(1, EPISODES_PER_CONFIG + 1))

    for cfg in configs:
        label  = cfg["label"]
        color  = cfg["color"]
        hist   = results[group_key][label]

        # Raw + rolling reward
        ax_reward.plot(eps, hist["rewards"], color=color, alpha=0.15, linewidth=0.7)
        roll_r = rolling(hist["rewards"], ROLLING_WINDOW)
        x_r    = eps[len(eps) - len(roll_r):]
        ax_reward.plot(x_r, roll_r, color=color, linewidth=2.2, label=label)

        # Success %
        roll_s = rolling(hist["successes"], ROLLING_WINDOW) * 100
        x_s    = eps[len(eps) - len(roll_s):]
        ax_success.plot(x_s, roll_s, color=color, linewidth=2, label=label)
        ax_success.fill_between(x_s, 0, roll_s, color=color, alpha=0.08)

        # Caught %
        roll_c = rolling(hist["caught"], ROLLING_WINDOW) * 100
        x_c    = eps[len(eps) - len(roll_c):]
        ax_caught.plot(x_c, roll_c, color=color, linewidth=2, label=label)
        ax_caught.fill_between(x_c, 0, roll_c, color=color, alpha=0.08)

    # Summary bar — final 50 episodes success rate per config
    labels       = [cfg["label"] for cfg in configs]
    colors       = [cfg["color"] for cfg in configs]
    final_success = [
        np.mean(results[group_key][cfg["label"]]["successes"][-50:]) * 100
        for cfg in configs
    ]
    final_caught  = [
        np.mean(results[group_key][cfg["label"]]["caught"][-50:]) * 100
        for cfg in configs
    ]

    x      = np.arange(len(labels))
    width  = 0.35
    bars_s = ax_summary.bar(x - width/2, final_success, width, color=colors, alpha=0.85,
                             label="Success %", edgecolor="#4A5568")
    bars_c = ax_summary.bar(x + width/2, final_caught,  width, color=colors, alpha=0.4,
                             label="Caught %",  edgecolor="#4A5568")

    for bar, val in zip(bars_s, final_success):
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.0f}%", ha="center", va="bottom",
            color=C_TEXT, fontsize=7, fontweight="bold"
        )

    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(
        [f"Cfg {i+1}" for i in range(len(labels))],
        color=C_MUTED, fontsize=7
    )
    ax_summary.set_ylim(0, 100)

    # Styling
    ax_reward.axhline(0, color=C_MUTED, linewidth=0.6, linestyle="--")
    style_ax(ax_reward,  "Episode Reward (Rolling Avg)", ylabel="Reward")
    style_ax(ax_success, "Success % (Rolling Avg)",      ylabel="%")
    style_ax(ax_caught,  "Caught % (Rolling Avg)",       ylabel="%")
    style_ax(ax_summary, "Final 50 Episodes Summary",    xlabel="Config", ylabel="%")

    ax_success.set_ylim(0, 100)
    ax_caught.set_ylim(0, 100)

    ax_reward.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT,
                     edgecolor="#4A5568", loc="lower right")
    ax_success.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
                      edgecolor="#4A5568")
    ax_caught.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
                     edgecolor="#4A5568")

    # Best config annotation
    best_idx = int(np.argmax(final_success))
    best_cfg = configs[best_idx]["label"]
    fig.text(
        0.99, 0.01,
        f"✅ Best: {best_cfg}  ({final_success[best_idx]:.0f}% success)",
        color=configs[best_idx]["color"],
        fontsize=9, fontweight="bold",
        ha="right", va="bottom"
    )

    fig.savefig(save_path, dpi=110, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_summary_dashboard(results: dict, save_path: str):
    """
    Final dashboard comparing the best config from each group
    and a heatmap of all configs ranked by success rate.
    """
    fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
    fig.suptitle(
        "REWARD HYPERTUNING — FULL SUMMARY",
        color="#F6AD55", fontsize=14, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    ax_best    = fig.add_subplot(gs[0, :2])   # best configs comparison
    ax_heatmap = fig.add_subplot(gs[0, 2])    # heatmap
    ax_reward  = fig.add_subplot(gs[1, 0])    # best reward curves
    ax_success = fig.add_subplot(gs[1, 1])    # best success curves
    ax_rec     = fig.add_subplot(gs[1, 2])    # recommendation panel

    eps = list(range(1, EPISODES_PER_CONFIG + 1))

    # ── Find best config from each group ──────────
    group_colors = {"group1_target_ratio": "#63B3ED",
                    "group2_caught_penalty": "#68D391",
                    "group3_distance_shape": "#B794F4"}

    group_labels = {"group1_target_ratio":    "G1: Target Ratio",
                    "group2_caught_penalty":  "G2: Caught Penalty",
                    "group3_distance_shape":  "G3: Distance Shape"}

    best_configs = {}
    all_rows     = []

    for group_key, group_data in EXPERIMENTS.items():
        configs = group_data["configs"]
        best_success = -1
        best_label   = ""
        best_hist    = None

        for cfg in configs:
            label = cfg["label"]
            hist  = results[group_key][label]
            fs    = np.mean(hist["successes"][-50:]) * 100
            fc    = np.mean(hist["caught"][-50:])    * 100
            fr    = np.mean(hist["rewards"][-50:])
            all_rows.append({
                "group": group_labels[group_key],
                "label": label,
                "success": fs,
                "caught":  fc,
                "reward":  fr,
            })
            if fs > best_success:
                best_success = fs
                best_label   = label
                best_hist    = hist
                best_cfg_obj = cfg

        best_configs[group_key] = {
            "label":   best_label,
            "hist":    best_hist,
            "success": best_success,
            "color":   group_colors[group_key],
            "cfg":     best_cfg_obj,
        }

    # ── Best configs bar chart ─────────────────────
    group_names   = [group_labels[k] for k in EXPERIMENTS]
    best_successes = [best_configs[k]["success"] for k in EXPERIMENTS]
    best_caught    = [
        np.mean(best_configs[k]["hist"]["caught"][-50:]) * 100
        for k in EXPERIMENTS
    ]
    bar_colors = [group_colors[k] for k in EXPERIMENTS]

    x     = np.arange(len(group_names))
    width = 0.35
    ax_best.bar(x - width/2, best_successes, width, color=bar_colors,
                alpha=0.9, label="Success %", edgecolor="#4A5568")
    ax_best.bar(x + width/2, best_caught,    width, color=bar_colors,
                alpha=0.4, label="Caught %",  edgecolor="#4A5568")

    for i, (s, c) in enumerate(zip(best_successes, best_caught)):
        ax_best.text(i - width/2, s + 0.5, f"{s:.0f}%",
                     ha="center", color=C_TEXT, fontsize=9, fontweight="bold")

    ax_best.set_xticks(x)
    ax_best.set_xticklabels(group_names, color=C_MUTED, fontsize=8)
    ax_best.set_ylim(0, 100)
    ax_best.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")
    style_ax(ax_best, "Best Config from Each Group", xlabel="", ylabel="%")

    # ── Heatmap ────────────────────────────────────
    row_labels = [f"{r['group'][:10]}\n{r['label'][:15]}" for r in all_rows]
    success_vals = np.array([r["success"] for r in all_rows])

    colors_heat = plt.cm.RdYlGn(success_vals / 100)
    bars = ax_heatmap.barh(range(len(all_rows)), success_vals,
                           color=colors_heat, edgecolor="#4A5568")
    ax_heatmap.set_yticks(range(len(all_rows)))
    ax_heatmap.set_yticklabels(row_labels, fontsize=6, color=C_MUTED)
    ax_heatmap.set_xlim(0, 100)
    for bar, val in zip(bars, success_vals):
        ax_heatmap.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.0f}%", va="center", color=C_TEXT, fontsize=7)
    style_ax(ax_heatmap, "All Configs Ranked", xlabel="Success %", ylabel="")

    # ── Best reward curves ─────────────────────────
    for group_key in EXPERIMENTS:
        bc    = best_configs[group_key]
        color = bc["color"]
        label = group_labels[group_key]
        roll  = rolling(bc["hist"]["rewards"], ROLLING_WINDOW)
        x_r   = eps[len(eps) - len(roll):]
        ax_reward.plot(x_r, roll, color=color, linewidth=2, label=label)

    ax_reward.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")
    style_ax(ax_reward, "Best Configs — Reward", ylabel="Reward")
    ax_reward.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")

    # ── Best success curves ────────────────────────
    for group_key in EXPERIMENTS:
        bc    = best_configs[group_key]
        color = bc["color"]
        label = group_labels[group_key]
        roll  = rolling(bc["hist"]["successes"], ROLLING_WINDOW) * 100
        x_r   = eps[len(eps) - len(roll):]
        ax_success.plot(x_r, roll, color=color, linewidth=2, label=label)
        ax_success.fill_between(x_r, 0, roll, color=color, alpha=0.08)

    ax_success.set_ylim(0, 100)
    style_ax(ax_success, "Best Configs — Success %", ylabel="%")
    ax_success.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")

    # ── Recommendation panel ───────────────────────
    ax_rec.axis("off")
    ax_rec.set_facecolor(C_PANEL)

    lines = [("RECOMMENDATIONS", "#F6AD55", 10, "bold"), ("", C_TEXT, 5, "normal")]

    for group_key in EXPERIMENTS:
        bc     = best_configs[group_key]
        cfg    = bc["cfg"]
        gname  = group_labels[group_key]
        color  = bc["color"]
        lines += [
            (gname, color, 8, "bold"),
            (f"  Best: {bc['label']}", C_TEXT, 7, "normal"),
            (f"  Success: {bc['success']:.0f}%", C_TEXT, 7, "normal"),
            (f"  R_TARGET={cfg['R_TARGET']} R_NEW_NODE={cfg['R_NEW_NODE']}", C_MUTED, 6, "normal"),
            (f"  R_CAUGHT={cfg['R_CAUGHT']} dist={cfg['dist_shape']}", C_MUTED, 6, "normal"),
            ("", C_TEXT, 4, "normal"),
        ]

    # Overall best
    overall_best_key = max(EXPERIMENTS.keys(), key=lambda k: best_configs[k]["success"])
    ob = best_configs[overall_best_key]
    lines += [
        ("─" * 25, C_MUTED, 6, "normal"),
        ("OVERALL BEST CONFIG", "#68D391", 9, "bold"),
        (f"  {ob['label']}", C_TEXT, 7, "normal"),
        (f"  Success: {ob['success']:.0f}%", "#68D391", 8, "bold"),
        (f"  R_TARGET   = {ob['cfg']['R_TARGET']}", C_TEXT, 7, "normal"),
        (f"  R_NEW_NODE = {ob['cfg']['R_NEW_NODE']}", C_TEXT, 7, "normal"),
        (f"  R_CAUGHT   = {ob['cfg']['R_CAUGHT']}", C_TEXT, 7, "normal"),
        (f"  dist_shape = {ob['cfg']['dist_shape']}", C_TEXT, 7, "normal"),
    ]

    y = 0.97
    for (text, color, size, weight) in lines:
        ax_rec.text(0.05, y, text, transform=ax_rec.transAxes,
                    color=color, fontsize=size, fontweight=weight, va="top",
                    fontfamily="monospace")
        y -= 0.045 + (size - 6) * 0.006

    fig.savefig(save_path, dpi=110, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def save_csv(results: dict, path: str):
    """Save all experiment results to CSV."""
    rows = []
    for group_key, group_data in EXPERIMENTS.items():
        for cfg in group_data["configs"]:
            label = cfg["label"]
            hist  = results[group_key][label]
            rows.append({
                "group":          group_key,
                "config":         label,
                "R_TARGET":       cfg["R_TARGET"],
                "R_NEW_NODE":     cfg["R_NEW_NODE"],
                "R_CAUGHT":       cfg["R_CAUGHT"],
                "dist_shape":     cfg["dist_shape"],
                "final50_success": f"{np.mean(hist['successes'][-50:]) * 100:.1f}",
                "final50_caught":  f"{np.mean(hist['caught'][-50:])    * 100:.1f}",
                "final50_reward":  f"{np.mean(hist['rewards'][-50:]):.1f}",
                "avg_steps":       f"{np.mean(hist['steps']):.1f}",
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {path}")


# ─────────────────────────────────────────────
#  LIVE PROGRESS WINDOW
# ─────────────────────────────────────────────

class LiveProgress:
    """Shows a simple live window during tuning so you can watch progress."""

    def __init__(self, total_configs: int, episodes_per: int):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4), facecolor=C_BG)
        self.fig.canvas.manager.set_window_title("Reward Hypertuning — Live Progress")
        self.ax.set_facecolor(C_PANEL)
        self.ax.tick_params(colors=C_MUTED, labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#4A5568")
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, episodes_per)
        self.ax.axhline(25, color=C_MUTED, linewidth=0.5, linestyle="--", alpha=0.5)
        self.ax.set_title("Live Success % — Current Config", color=C_TEXT,
                          fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Episode", color=C_MUTED, fontsize=8)
        self.ax.set_ylabel("Success %", color=C_MUTED, fontsize=8)
        self.line, = self.ax.plot([], [], color="#68D391", linewidth=2)
        self.info  = self.fig.text(0.99, 0.02, "", color=C_TEXT,
                                   fontsize=8, ha="right", va="bottom")
        plt.show(block=False)
        plt.pause(0.1)

    def update(self, successes: list, label: str, group: str, config_num: int, total: int):
        eps  = list(range(1, len(successes) + 1))
        roll = rolling(successes, ROLLING_WINDOW) * 100
        x_r  = eps[len(eps) - len(roll):]
        self.line.set_data(x_r, roll)
        self.ax.set_title(
            f"[{config_num}/{total}] {group}\n{label}",
            color=C_TEXT, fontsize=9, fontweight="bold"
        )
        cur_success = roll[-1] if len(roll) > 0 else 0
        self.info.set_text(f"Current success: {cur_success:.0f}%")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/logs",  exist_ok=True)

    total_configs = sum(len(g["configs"]) for g in EXPERIMENTS.values())
    total_eps     = total_configs * EPISODES_PER_CONFIG

    print("\n" + "="*60)
    print("  REWARD HYPERTUNING")
    print("="*60)
    print(f"  Configs       : {total_configs}")
    print(f"  Episodes each : {EPISODES_PER_CONFIG}")
    print(f"  Total episodes: {total_eps}")
    print(f"  Groups        : {len(EXPERIMENTS)}")
    print("="*60 + "\n")

    live = LiveProgress(total_configs, EPISODES_PER_CONFIG)
    results = {}
    config_num = 0

    for group_key, group_data in EXPERIMENTS.items():
        print(f"\n{'─'*50}")
        print(f"  {group_data['title']}")
        print(f"{'─'*50}")
        results[group_key] = {}

        for cfg in group_data["configs"]:
            config_num += 1
            label = cfg["label"]
            print(f"\n  [{config_num}/{total_configs}] {label}")

            successes_live = []

            pbar = tqdm(range(EPISODES_PER_CONFIG),
                        desc=f"    {label[:30]}", unit="ep", leave=False)

            # Run with live update every 10 episodes
            env   = PatchedEnv(cfg, seed=42)
            agent = DDQNAgent(
                state_size  = env.state_size,
                action_size = env.action_size,
                **AGENT_CFG
            )
            history = {"rewards": [], "successes": [], "caught": [], "steps": []}

            for ep in pbar:
                state     = env.reset()
                ep_reward = 0.0

                for _ in range(100):
                    valid      = env.get_valid_actions()
                    action     = agent.select_action(state, valid_actions=valid)
                    next_state, reward, done, _ = env.step(action)
                    agent.store(state, action, reward, next_state, done)
                    agent.train_step()
                    ep_reward += reward
                    state = next_state
                    if done:
                        break

                agent.decay_epsilon()
                history["rewards"].append(ep_reward)
                history["successes"].append(1 if env.stats["success"] else 0)
                history["caught"].append(1 if env.stats["caught"] else 0)
                history["steps"].append(env.stats["steps"])
                successes_live.append(1 if env.stats["success"] else 0)

                # Live update every 10 eps
                if ep % 10 == 0:
                    live.update(successes_live, label, group_data["title"],
                                config_num, total_configs)

                pbar.set_postfix({
                    "success": f"{np.mean(history['successes'][-50:])*100:.0f}%",
                    "ε":       f"{agent.epsilon:.3f}",
                })

            results[group_key][label] = history
            final_s = np.mean(history["successes"][-50:]) * 100
            final_c = np.mean(history["caught"][-50:])    * 100
            print(f"    Final 50 eps → Success: {final_s:.0f}%  Caught: {final_c:.0f}%")

        # Save group plot immediately after group finishes
        group_plot_path = f"{SAVE_DIR}/plots/{group_key}.png"
        print(f"\n  Plotting group...")
        plot_group(group_key, group_data, results, group_plot_path)

    live.close()

    # ── Final summary dashboard ────────────────────
    print("\n  Generating summary dashboard...")
    plot_summary_dashboard(results, f"{SAVE_DIR}/plots/summary_dashboard.png")

    # ── Save CSV log ───────────────────────────────
    save_csv(results, f"{SAVE_DIR}/logs/experiment_log.csv")

    print("\n" + "="*60)
    print("  TUNING COMPLETE")
    print("="*60)
    print(f"  Group plots  → {SAVE_DIR}/plots/")
    print(f"  Summary      → {SAVE_DIR}/plots/summary_dashboard.png")
    print(f"  CSV log      → {SAVE_DIR}/logs/experiment_log.csv")
    print("="*60)
    print("\n  Open summary_dashboard.png for the recommended config.")
    print("  Copy those values into network_env.py and re-run train.py\n")


if __name__ == "__main__":
    main()