"""
hyperparam_tune.py
------------------
Automated hyperparameter tuning for the Pentest DDQN agent.

5 groups tuned in priority order:
  Group 1 — Learning Rate          (biggest impact)
  Group 2 — Gamma                  (planning horizon)
  Group 3 — Epsilon Decay          (exploration schedule)
  Group 4 — Batch Size             (gradient stability)
  Group 5 — Target Update Freq     (Q-value stability)

Each config trains from scratch for 1000 episodes.
Results saved to results/hyperparam_tuning/

Run:
    python hyperparam_tune.py

    # Run only specific groups (saves time):
    python hyperparam_tune.py --groups 1 2
    python hyperparam_tune.py --groups 3

Output:
    results/hyperparam_tuning/
        plots/
            group1_lr.png
            group2_gamma.png
            group3_epsilon_decay.png
            group4_batch_size.png
            group5_target_update.png
            summary_dashboard.png
        logs/
            experiment_log.csv
            best_config.txt
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from env.network_env  import NetworkEnv
from agent.ddqn_agent import DDQNAgent


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

EPISODES_PER_CONFIG = 1000
ROLLING_WINDOW      = 50
SAVE_DIR            = "results/hyperparam_tuning"

# ── Baseline — keep all others fixed while tuning one ──
BASELINE = {
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
#  EXPERIMENT GROUPS
# ─────────────────────────────────────────────

GROUPS = {

    1: {
        "key":         "group1_lr",
        "title":       "Group 1 — Learning Rate",
        "description": "Controls how fast the network weights update. Too high = unstable. Too low = slow.",
        "param":       "lr",
        "configs": [
            {"label": "Slow    (1e-4)",  "color": "#FC8181", "lr": 1e-4},
            {"label": "Default (1e-3)",  "color": "#63B3ED", "lr": 1e-3},
            {"label": "Fast    (5e-4)",  "color": "#68D391", "lr": 5e-4},
            {"label": "Aggr.   (3e-3)",  "color": "#F6AD55", "lr": 3e-3},
        ],
    },

    2: {
        "key":         "group2_gamma",
        "title":       "Group 2 — Gamma (Discount Factor)",
        "description": "How much the agent values future rewards. Higher = longer planning horizon.",
        "param":       "gamma",
        "configs": [
            {"label": "Short-sighted (0.95)",  "color": "#FC8181", "gamma": 0.95},
            {"label": "Default      (0.99)",   "color": "#63B3ED", "gamma": 0.99},
            {"label": "Far-sighted  (0.995)",  "color": "#68D391", "gamma": 0.995},
            {"label": "Max          (0.999)",  "color": "#F6AD55", "gamma": 0.999},
        ],
    },

    3: {
        "key":         "group3_epsilon_decay",
        "title":       "Group 3 — Epsilon Decay",
        "description": "How fast exploration reduces. Too fast = commits too early. Too slow = never exploits.",
        "param":       "epsilon_decay",
        "configs": [
            {"label": "Fast    (0.999)",   "color": "#FC8181", "epsilon_decay": 0.999},
            {"label": "Default (0.9995)",  "color": "#63B3ED", "epsilon_decay": 0.9995},
            {"label": "Slow    (0.9998)",  "color": "#68D391", "epsilon_decay": 0.9998},
            {"label": "Glacial (0.9999)",  "color": "#F6AD55", "epsilon_decay": 0.9999},
        ],
    },

    4: {
        "key":         "group4_batch_size",
        "title":       "Group 4 — Batch Size",
        "description": "Number of transitions per training step. Larger = more stable but slower.",
        "param":       "batch_size",
        "configs": [
            {"label": "Small   (32)",   "color": "#FC8181", "batch_size": 32},
            {"label": "Default (64)",   "color": "#63B3ED", "batch_size": 64},
            {"label": "Large   (128)",  "color": "#68D391", "batch_size": 128},
            {"label": "XLarge  (256)",  "color": "#F6AD55", "batch_size": 256},
        ],
    },

    5: {
        "key":         "group5_target_update",
        "title":       "Group 5 — Target Network Update Frequency",
        "description": "How often online weights copy to target net. Lower = faster but less stable.",
        "param":       "target_update_freq",
        "configs": [
            {"label": "Very frequent (50)",   "color": "#FC8181", "target_update_freq": 50},
            {"label": "Frequent     (100)",   "color": "#68D391", "target_update_freq": 100},
            {"label": "Default      (200)",   "color": "#63B3ED", "target_update_freq": 200},
            {"label": "Infrequent   (500)",   "color": "#F6AD55", "target_update_freq": 500},
        ],
    },
}


# ─────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────

def build_agent_cfg(group: dict, cfg: dict) -> dict:
    """Merge baseline with the one param being tested."""
    agent_cfg = dict(BASELINE)
    param     = group["param"]
    agent_cfg[param] = cfg[param]
    return agent_cfg


def run_config(agent_cfg: dict, seed: int = 42) -> dict:
    """Train one config from scratch for EPISODES_PER_CONFIG episodes."""
    env   = NetworkEnv(defender_level=1, seed=seed)
    agent = DDQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,
        **agent_cfg,
    )

    history = {
        "rewards":   [],
        "successes": [],
        "caught":    [],
        "timeouts":  [],
        "steps":     [],
        "epsilons":  [],
        "losses":    [],
    }

    for ep in range(EPISODES_PER_CONFIG):
        state     = env.reset()
        ep_reward = 0.0
        ep_losses = []

        for _ in range(100):
            valid      = env.get_valid_actions()
            action     = agent.select_action(state, valid_actions=valid)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)
            ep_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()

        history["rewards"].append(ep_reward)
        history["successes"].append(1 if env.stats["success"] else 0)
        history["caught"].append(1 if env.stats["caught"]  else 0)
        history["timeouts"].append(1 if env.stats["timeout"] else 0)
        history["steps"].append(env.stats["steps"])
        history["epsilons"].append(agent.epsilon)
        history["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)

    return history


# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────

C_BG    = "#1A202C"
C_PANEL = "#2D3748"
C_TEXT  = "#E2E8F0"
C_MUTED = "#718096"


def rolling(data, window=50):
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
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=8)


def final_stats(hist, window=100):
    return {
        "success": np.mean(hist["successes"][-window:]) * 100,
        "caught":  np.mean(hist["caught"][-window:])    * 100,
        "timeout": np.mean(hist["timeouts"][-window:])  * 100,
        "reward":  np.mean(hist["rewards"][-window:]),
        "steps":   np.mean(hist["steps"][-window:]),
        "loss":    np.mean(hist["losses"][-window:]),
    }


# ─────────────────────────────────────────────
#  GROUP PLOT
# ─────────────────────────────────────────────

def plot_group(group: dict, results: dict, save_path: str):
    """6-subplot dashboard for one group — saves to PNG."""
    configs = group["configs"]
    eps     = list(range(1, EPISODES_PER_CONFIG + 1))

    fig = plt.figure(figsize=(18, 10), facecolor=C_BG)
    fig.suptitle(
        f"{group['title']}\n{group['description']}",
        color=C_TEXT, fontsize=12, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    ax_reward  = fig.add_subplot(gs[0, :2])
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_success = fig.add_subplot(gs[1, 0])
    ax_caught  = fig.add_subplot(gs[1, 1])
    ax_epsilon = fig.add_subplot(gs[1, 2])
    ax_loss    = fig.add_subplot(gs[2, 0])
    ax_steps   = fig.add_subplot(gs[2, 1])
    ax_rank    = fig.add_subplot(gs[2, 2])

    stats_all = {}

    for cfg in configs:
        label = cfg["label"]
        color = cfg["color"]
        hist  = results[label]
        stats = final_stats(hist)
        stats_all[label] = stats

        # Rolling reward
        ax_reward.plot(eps, hist["rewards"], color=color, alpha=0.12, linewidth=0.6)
        roll = rolling(hist["rewards"], ROLLING_WINDOW)
        x_r  = eps[len(eps) - len(roll):]
        ax_reward.plot(x_r, roll, color=color, linewidth=2, label=f"{label}  ({stats['reward']:+.0f})")

        # Success %
        roll_s = rolling(hist["successes"], ROLLING_WINDOW) * 100
        x_s    = eps[len(eps) - len(roll_s):]
        ax_success.plot(x_s, roll_s, color=color, linewidth=2, label=label)
        ax_success.fill_between(x_s, 0, roll_s, color=color, alpha=0.07)

        # Caught %
        roll_c = rolling(hist["caught"], ROLLING_WINDOW) * 100
        x_c    = eps[len(eps) - len(roll_c):]
        ax_caught.plot(x_c, roll_c, color=color, linewidth=2, label=label)

        # Epsilon curve
        ax_epsilon.plot(eps, hist["epsilons"], color=color, linewidth=1.8, label=label)

        # Loss curve
        roll_l = rolling(hist["losses"], ROLLING_WINDOW)
        x_l    = eps[len(eps) - len(roll_l):]
        ax_loss.plot(x_l, roll_l, color=color, linewidth=1.5, label=label)

        # Steps per episode
        roll_st = rolling(hist["steps"], ROLLING_WINDOW)
        x_st    = eps[len(eps) - len(roll_st):]
        ax_steps.plot(x_st, roll_st, color=color, linewidth=1.5, label=label)

    # ── Summary bar chart ──────────────────────
    labels        = [cfg["label"] for cfg in configs]
    colors        = [cfg["color"] for cfg in configs]
    final_success = [stats_all[l]["success"] for l in labels]
    final_caught  = [stats_all[l]["caught"]  for l in labels]

    x     = np.arange(len(labels))
    width = 0.35
    bars  = ax_summary.bar(x - width/2, final_success, width, color=colors,
                           alpha=0.9, edgecolor="#4A5568", label="Success %")
    ax_summary.bar(x + width/2, final_caught, width, color=colors,
                   alpha=0.35, edgecolor="#4A5568", label="Caught %")

    for bar, val in zip(bars, final_success):
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.0f}%", ha="center", va="bottom",
            color=C_TEXT, fontsize=8, fontweight="bold"
        )

    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels([f"Cfg {i+1}" for i in range(len(labels))],
                                color=C_MUTED, fontsize=7)
    ax_summary.set_ylim(0, 105)
    ax_summary.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")

    # ── Rank bar (horizontal) ──────────────────
    sorted_cfgs = sorted(labels, key=lambda l: stats_all[l]["success"], reverse=True)
    rank_vals   = [stats_all[l]["success"] for l in sorted_cfgs]
    rank_colors = [next(c["color"] for c in configs if c["label"] == l) for l in sorted_cfgs]
    short_labels = [l.split("(")[0].strip() for l in sorted_cfgs]

    bars_r = ax_rank.barh(range(len(sorted_cfgs)), rank_vals,
                           color=rank_colors, edgecolor="#4A5568", alpha=0.85)
    ax_rank.set_yticks(range(len(sorted_cfgs)))
    ax_rank.set_yticklabels(short_labels, fontsize=7, color=C_MUTED)
    ax_rank.set_xlim(0, 100)
    for bar, val in zip(bars_r, rank_vals):
        ax_rank.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}%", va="center", color=C_TEXT, fontsize=8,
                     fontweight="bold")

    # ── Styling ────────────────────────────────
    ax_reward.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")
    style_ax(ax_reward,  "Episode Reward (Rolling Avg)",    ylabel="Reward")
    style_ax(ax_summary, "Final 100 Eps Summary",           xlabel="", ylabel="%")
    style_ax(ax_success, "Success % (Rolling Avg)",         ylabel="%")
    style_ax(ax_caught,  "Caught % (Rolling Avg)",          ylabel="%")
    style_ax(ax_epsilon, "Epsilon Decay Curve",             ylabel="ε")
    style_ax(ax_loss,    "Training Loss (Rolling Avg)",     ylabel="Loss")
    style_ax(ax_steps,   "Avg Steps per Episode",           ylabel="Steps")
    style_ax(ax_rank,    "Configs Ranked by Success %",     xlabel="Success %", ylabel="")

    ax_success.set_ylim(0, 100)
    ax_caught.set_ylim(0, 100)
    ax_epsilon.set_ylim(0, 1.05)

    for ax in [ax_reward, ax_success, ax_caught, ax_epsilon, ax_loss, ax_steps]:
        ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
                  edgecolor="#4A5568", loc="best")

    # ── Best config annotation ─────────────────
    best_label   = sorted_cfgs[0]
    best_success = rank_vals[0]
    best_color   = rank_colors[0]
    fig.text(
        0.99, 0.005,
        f"✅ Best: {best_label}  →  {best_success:.0f}% success",
        color=best_color, fontsize=9, fontweight="bold", ha="right", va="bottom"
    )

    fig.savefig(save_path, dpi=110, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────
#  SUMMARY DASHBOARD
# ─────────────────────────────────────────────

def plot_summary_dashboard(all_results: dict, groups_run: list, save_path: str):
    """Final cross-group summary dashboard."""
    fig = plt.figure(figsize=(18, 11), facecolor=C_BG)
    fig.suptitle(
        "HYPERPARAMETER TUNING — FULL SUMMARY",
        color="#F6AD55", fontsize=14, fontweight="bold", y=0.99
    )

    n_groups  = len(groups_run)
    n_cols    = 3   # always 3 columns — works for 1 to 5 groups
    gs = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    # ── Row 1: best success per group (up to 3 per row) ───
    axes_top = []
    for i in range(n_groups):
        col = i % n_cols
        axes_top.append(fig.add_subplot(gs[0, col]))

    best_per_group   = {}
    eps = list(range(1, EPISODES_PER_CONFIG + 1))

    for i, gnum in enumerate(groups_run):
        group   = GROUPS[gnum]
        results = all_results[group["key"]]
        configs = group["configs"]
        ax      = axes_top[i]

        best_label   = ""
        best_success = -1
        best_hist    = None
        best_color   = ""

        for cfg in configs:
            label = cfg["label"]
            hist  = results[label]
            fs    = np.mean(hist["successes"][-100:]) * 100
            color = cfg["color"]

            roll = rolling(hist["successes"], ROLLING_WINDOW) * 100
            x_r  = eps[len(eps) - len(roll):]
            ax.plot(x_r, roll, color=color, linewidth=1.8,
                    label=f"{label.split('(')[0].strip()} ({fs:.0f}%)")

            if fs > best_success:
                best_success = fs
                best_label   = label
                best_hist    = hist
                best_color   = color

        best_per_group[gnum] = {
            "label":   best_label,
            "success": best_success,
            "hist":    best_hist,
            "color":   best_color,
            "cfg":     next(c for c in configs if c["label"] == best_label),
            "group":   group,
        }

        ax.set_ylim(0, 100)
        ax.fill_between(x_r, 0, rolling(best_hist["successes"], ROLLING_WINDOW) * 100,
                         color=best_color, alpha=0.08)
        style_ax(ax, group["title"].replace("Group ", "G").split("—")[0].strip(),
                 ylabel="Success %")
        ax.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT,
                  edgecolor="#4A5568", loc="upper left")
        ax.text(0.99, 0.97, f"Best: {best_success:.0f}%",
                transform=ax.transAxes, color=best_color,
                fontsize=8, fontweight="bold", ha="right", va="top")

    # ── Row 2: best configs comparison ────────────
    ax_reward  = fig.add_subplot(gs[1, 0])
    ax_success = fig.add_subplot(gs[1, 1])
    ax_heatmap = fig.add_subplot(gs[1, 2])

    for gnum in groups_run:
        bc    = best_per_group[gnum]
        color = bc["color"]
        label = f"G{gnum}: {bc['label'].split('(')[1].rstrip(')')}"

        roll_r = rolling(bc["hist"]["rewards"],   ROLLING_WINDOW)
        roll_s = rolling(bc["hist"]["successes"], ROLLING_WINDOW) * 100
        x_r    = eps[len(eps) - len(roll_r):]
        x_s    = eps[len(eps) - len(roll_s):]

        ax_reward.plot(x_r, roll_r, color=color, linewidth=2, label=label)
        ax_success.plot(x_s, roll_s, color=color, linewidth=2, label=label)
        ax_success.fill_between(x_s, 0, roll_s, color=color, alpha=0.07)

    ax_reward.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")
    style_ax(ax_reward,  "Best Config from Each Group — Reward",    ylabel="Reward")
    style_ax(ax_success, "Best Config from Each Group — Success %", ylabel="%")
    ax_success.set_ylim(0, 100)

    for ax in [ax_reward, ax_success]:
        ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT,
                  edgecolor="#4A5568", loc="best")

    # ── Heatmap — all configs ──────────────────────
    all_rows = []
    for gnum in groups_run:
        group   = GROUPS[gnum]
        results = all_results[group["key"]]
        for cfg in group["configs"]:
            label = cfg["label"]
            hist  = results[label]
            all_rows.append({
                "name":    f"G{gnum} {label.split('(')[0].strip()[:12]}",
                "success": np.mean(hist["successes"][-100:]) * 100,
                "color":   cfg["color"],
            })

    all_rows.sort(key=lambda r: r["success"], reverse=True)
    heat_colors = plt.cm.RdYlGn(np.array([r["success"] for r in all_rows]) / 100)
    bars = ax_heatmap.barh(range(len(all_rows)),
                            [r["success"] for r in all_rows],
                            color=heat_colors, edgecolor="#4A5568")
    ax_heatmap.set_yticks(range(len(all_rows)))
    ax_heatmap.set_yticklabels([r["name"] for r in all_rows], fontsize=6.5, color=C_MUTED)
    ax_heatmap.set_xlim(0, 100)
    for bar, row in zip(bars, all_rows):
        ax_heatmap.text(row["success"] + 0.5,
                         bar.get_y() + bar.get_height() / 2,
                         f"{row['success']:.0f}%", va="center",
                         color=C_TEXT, fontsize=7, fontweight="bold")
    style_ax(ax_heatmap, "All Configs Ranked", xlabel="Success %", ylabel="")

    # ── Row 3: Recommendations ────────────────────
    ax_rec = fig.add_subplot(gs[2, :])
    ax_rec.axis("off")
    ax_rec.set_facecolor(C_PANEL)

    # Build recommendation text
    overall_best = max(groups_run, key=lambda g: best_per_group[g]["success"])
    ob           = best_per_group[overall_best]

    rec_lines = [
        ("RECOMMENDED CONFIG — USE THESE VALUES IN train.py CFG", "#F6AD55", 10, "bold"),
        ("", C_TEXT, 4, "normal"),
    ]

    final_cfg = dict(BASELINE)
    for gnum in groups_run:
        bc    = best_per_group[gnum]
        param = GROUPS[gnum]["param"]
        final_cfg[param] = bc["cfg"][param]
        rec_lines.append((
            f"  G{gnum} {GROUPS[gnum]['title'].split('—')[1].strip():25s}  "
            f"→  {param:20s} = {bc['cfg'][param]}   "
            f"({bc['success']:.0f}% success,  was {BASELINE[param]})",
            bc["color"], 8, "normal"
        ))

    rec_lines += [
        ("", C_TEXT, 4, "normal"),
        ("─" * 100, C_MUTED, 6, "normal"),
        ("FINAL CFG BLOCK — copy this into train.py:", "#68D391", 9, "bold"),
        ("", C_TEXT, 3, "normal"),
    ]

    cfg_lines = [
        f'    "lr":                 {final_cfg["lr"]}',
        f'    "gamma":              {final_cfg["gamma"]}',
        f'    "epsilon_decay":      {final_cfg["epsilon_decay"]}',
        f'    "batch_size":         {final_cfg["batch_size"]}',
        f'    "target_update_freq": {final_cfg["target_update_freq"]}',
    ]
    for line in cfg_lines:
        rec_lines.append((line, C_TEXT, 8, "normal"))

    y = 0.95
    for (text, color, size, weight) in rec_lines:
        ax_rec.text(0.01, y, text, transform=ax_rec.transAxes,
                    color=color, fontsize=size, fontweight=weight,
                    va="top", fontfamily="monospace")
        y -= 0.085 + (size - 6) * 0.01

    fig.savefig(save_path, dpi=110, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────
#  CSV + BEST CONFIG TXT
# ─────────────────────────────────────────────

def save_csv(all_results: dict, groups_run: list, path: str):
    rows = []
    for gnum in groups_run:
        group   = GROUPS[gnum]
        results = all_results[group["key"]]
        for cfg in group["configs"]:
            label = cfg["label"]
            hist  = results[label]
            stats = final_stats(hist)
            row   = {
                "group":       f"G{gnum}",
                "param":       group["param"],
                "config":      label,
                "value":       cfg[group["param"]],
                "success_pct": f"{stats['success']:.1f}",
                "caught_pct":  f"{stats['caught']:.1f}",
                "timeout_pct": f"{stats['timeout']:.1f}",
                "avg_reward":  f"{stats['reward']:.1f}",
                "avg_steps":   f"{stats['steps']:.1f}",
                "avg_loss":    f"{stats['loss']:.4f}",
            }
            rows.append(row)

    rows.sort(key=lambda r: float(r["success_pct"]), reverse=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV → {path}")


def save_best_config(all_results: dict, groups_run: list, path: str):
    lines = ["BEST HYPERPARAMETER CONFIG\n", "=" * 40 + "\n\n"]
    final_cfg = dict(BASELINE)

    for gnum in groups_run:
        group   = GROUPS[gnum]
        results = all_results[group["key"]]
        configs = group["configs"]
        param   = group["param"]

        best_label   = max(configs, key=lambda c: np.mean(
            results[c["label"]]["successes"][-100:]))["label"]
        best_success = np.mean(results[best_label]["successes"][-100:]) * 100
        best_value   = next(c[param] for c in configs if c["label"] == best_label)

        final_cfg[param] = best_value
        lines.append(f"G{gnum} {group['title'].split('—')[1].strip()}\n")
        lines.append(f"  Best : {best_label}\n")
        lines.append(f"  Value: {param} = {best_value}\n")
        lines.append(f"  Score: {best_success:.1f}% success\n\n")

    lines += [
        "=" * 40 + "\n",
        "COPY INTO train.py CFG:\n\n",
        f'  "lr":                 {final_cfg["lr"]},\n',
        f'  "gamma":              {final_cfg["gamma"]},\n',
        f'  "epsilon_decay":      {final_cfg["epsilon_decay"]},\n',
        f'  "batch_size":         {final_cfg["batch_size"]},\n',
        f'  "target_update_freq": {final_cfg["target_update_freq"]},\n',
    ]

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  Best config → {path}")


# ─────────────────────────────────────────────
#  LIVE PROGRESS WINDOW
#  Uses a background thread so the training loop
#  never blocks waiting for matplotlib on Windows.
# ─────────────────────────────────────────────

import threading
import queue


class LiveProgress:
    """
    Live dashboard running in a dedicated GUI thread.
    Training loop calls .push() to send data — non-blocking.
    The GUI thread drains the queue and redraws at its own pace.
    """

    def __init__(self):
        # Shared data queue — training pushes, GUI thread pops
        self._queue   = queue.Queue(maxsize=5)
        self._running = True
        self._label   = ""
        self._title   = ""
        self._cfg_num = 0
        self._total   = 0

        # Start GUI in its own thread
        self._thread = threading.Thread(target=self._gui_loop, daemon=True)
        self._thread.start()

        # Give window time to open before training starts
        import time
        time.sleep(1.5)

    def _gui_loop(self):
        """Runs entirely in the GUI thread — owns all matplotlib objects."""
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=C_BG)
        fig.canvas.manager.set_window_title("Hyperparameter Tuning — Live")

        ax_s, ax_r = axes
        for ax in [ax_s, ax_r]:
            ax.set_facecolor(C_PANEL)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#4A5568")

        ax_s.set_ylim(0, 100)
        ax_s.set_ylabel("Success %", color=C_MUTED, fontsize=8)
        ax_r.set_ylabel("Reward",    color=C_MUTED, fontsize=8)
        ax_r.axhline(0, color=C_MUTED, linewidth=0.6, linestyle="--")

        line_s, = ax_s.plot([], [], color="#68D391", linewidth=2.2)
        line_r, = ax_r.plot([], [], color="#63B3ED", linewidth=2.2)
        info     = fig.text(0.5, 0.01, "Waiting for first update...",
                            color=C_TEXT, fontsize=9, ha="center",
                            va="bottom", fontweight="bold")

        plt.tight_layout(pad=2.5)
        plt.show(block=False)
        fig.canvas.flush_events()

        while self._running:
            # Drain all pending updates — only render the latest one
            latest = None
            try:
                while True:
                    latest = self._queue.get_nowait()
            except queue.Empty:
                pass

            if latest is not None:
                hist, label, group_title, cfg_num, total = latest

                eps    = list(range(1, len(hist["successes"]) + 1))
                roll_s = rolling(hist["successes"], ROLLING_WINDOW) * 100
                roll_r = rolling(hist["rewards"],   ROLLING_WINDOW)
                x_s    = eps[len(eps) - len(roll_s):]
                x_r    = eps[len(eps) - len(roll_r):]

                line_s.set_data(x_s, roll_s)
                line_r.set_data(x_r, roll_r)

                ax_s.set_xlim(0, max(eps) + 5)
                ax_r.set_xlim(0, max(eps) + 5)

                if len(roll_r) > 1:
                    rmin = float(roll_r.min())
                    rmax = float(roll_r.max())
                    pad  = max(abs(rmin), abs(rmax)) * 0.15 + 10
                    ax_r.set_ylim(rmin - pad, rmax + pad)

                cur_s = float(roll_s[-1]) if len(roll_s) > 0 else 0.0
                cur_r = float(roll_r[-1]) if len(roll_r) > 0 else 0.0
                cur_e = hist["epsilons"][-1] if hist["epsilons"] else 1.0

                ax_s.set_title(
                    f"[{cfg_num}/{total}]  {group_title.split('—')[0].strip()}\n{label}",
                    color=C_TEXT, fontsize=8, fontweight="bold", pad=4
                )
                ax_r.set_title(
                    f"Reward (rolling avg {ROLLING_WINDOW})",
                    color=C_TEXT, fontsize=8, fontweight="bold", pad=4
                )
                info.set_text(
                    f"Success: {cur_s:.0f}%   |   "
                    f"Reward: {cur_r:+.0f}   |   "
                    f"ε: {cur_e:.4f}   |   "
                    f"Episode: {len(hist['successes'])}"
                )

                try:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                except Exception:
                    pass

            # Process Tk events to keep window responsive
            try:
                fig.canvas.flush_events()
            except Exception:
                pass

            import time
            time.sleep(0.05)   # 20 fps max — keeps CPU sane

        # Cleanup
        try:
            plt.ioff()
            plt.close(fig)
        except Exception:
            pass

    def update(self, hist: dict, label: str, group_title: str,
               cfg_num: int, total: int):
        """
        Called from training loop — puts data on queue, never blocks.
        If queue is full (GUI is slow), drops oldest item first.
        """
        payload = (
            {k: list(v) for k, v in hist.items()},  # shallow copy
            label, group_title, cfg_num, total
        )
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()   # drop oldest
                self._queue.put_nowait(payload)
            except Exception:
                pass

    def close(self):
        """Signal GUI thread to stop and wait for it."""
        self._running = False
        try:
            self._thread.join(timeout=3.0)
        except Exception:
            pass


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main(groups_to_run: list):
    os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/logs",  exist_ok=True)

    total_configs = sum(len(GROUPS[g]["configs"]) for g in groups_to_run)
    total_eps     = total_configs * EPISODES_PER_CONFIG

    print("\n" + "=" * 65)
    print("  HYPERPARAMETER TUNING")
    print("=" * 65)
    print(f"  Groups        : {groups_to_run}")
    print(f"  Configs       : {total_configs}")
    print(f"  Episodes each : {EPISODES_PER_CONFIG}")
    print(f"  Total episodes: {total_eps:,}")
    print(f"  Est. time     : ~{total_configs * 15} min")
    print("=" * 65 + "\n")

    live        = LiveProgress()
    all_results = {}
    cfg_num     = 0

    for gnum in groups_to_run:
        group     = GROUPS[gnum]
        group_key = group["key"]
        configs   = group["configs"]

        print(f"\n{'─' * 55}")
        print(f"  {group['title']}")
        print(f"  {group['description']}")
        print(f"{'─' * 55}")

        all_results[group_key] = {}

        for cfg in configs:
            cfg_num += 1
            label    = cfg["label"]
            param    = group["param"]

            print(f"\n  [{cfg_num}/{total_configs}] {label}")
            print(f"    {param} = {cfg[param]}  (baseline = {BASELINE[param]})")

            agent_cfg = build_agent_cfg(group, cfg)
            hist      = {"rewards": [], "successes": [], "caught": [],
                         "timeouts": [], "steps": [], "epsilons": [], "losses": []}

            env   = NetworkEnv(defender_level=1, seed=42)
            agent = DDQNAgent(
                state_size  = env.state_size,
                action_size = env.action_size,
                **agent_cfg,
            )

            pbar = tqdm(range(EPISODES_PER_CONFIG),
                        desc=f"    {label[:35]}", unit="ep", leave=False)

            for ep in pbar:
                state     = env.reset()
                ep_reward = 0.0
                ep_losses = []

                for _ in range(100):
                    valid  = env.get_valid_actions()
                    action = agent.select_action(state, valid_actions=valid)
                    ns, reward, done, _ = env.step(action)
                    agent.store(state, action, reward, ns, done)
                    loss = agent.train_step()
                    if loss > 0:
                        ep_losses.append(loss)
                    ep_reward += reward
                    state = ns
                    if done:
                        break

                agent.decay_epsilon()

                hist["rewards"].append(ep_reward)
                hist["successes"].append(1 if env.stats["success"] else 0)
                hist["caught"].append(1 if env.stats["caught"]  else 0)
                hist["timeouts"].append(1 if env.stats["timeout"] else 0)
                hist["steps"].append(env.stats["steps"])
                hist["epsilons"].append(agent.epsilon)
                hist["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)

                if ep % 10 == 0:
                    live.update(hist, label, group["title"], cfg_num, total_configs)

                pbar.set_postfix({
                    "success": f"{np.mean(hist['successes'][-50:])*100:.0f}%",
                    "reward":  f"{np.mean(hist['rewards'][-50:]):+.0f}",
                    "ε":       f"{agent.epsilon:.4f}",
                })

            all_results[group_key][label] = hist
            stats = final_stats(hist)
            print(f"    Final 100 eps → "
                  f"Success: {stats['success']:.0f}%  "
                  f"Caught: {stats['caught']:.0f}%  "
                  f"Reward: {stats['reward']:+.0f}")

    # ── Close live window BEFORE any further matplotlib calls ──
    print("\n  Closing live window...")
    live.close()
    import time; time.sleep(0.8)   # let thread fully exit before matplotlib reuse

    # ── Now safe to plot — all on main thread with no GUI thread running ──
    for gnum in groups_to_run:
        group           = GROUPS[gnum]
        group_plot_path = f"{SAVE_DIR}/plots/{group['key']}.png"
        print(f"\n  Plotting group {gnum} — {group['title']}...")
        plot_group(group, all_results[group["key"]], group_plot_path)

    # ── Summary ───────────────────────────────────
    print("\n  Generating summary dashboard...")
    plot_summary_dashboard(all_results, groups_to_run,
                           f"{SAVE_DIR}/plots/summary_dashboard.png")

    save_csv(all_results, groups_to_run,
             f"{SAVE_DIR}/logs/experiment_log.csv")

    save_best_config(all_results, groups_to_run,
                     f"{SAVE_DIR}/logs/best_config.txt")

    print("\n" + "=" * 65)
    print("  TUNING COMPLETE")
    print("=" * 65)
    print(f"  Group plots  → {SAVE_DIR}/plots/")
    print(f"  Summary      → {SAVE_DIR}/plots/summary_dashboard.png")
    print(f"  CSV log      → {SAVE_DIR}/logs/experiment_log.csv")
    print(f"  Best config  → {SAVE_DIR}/logs/best_config.txt")
    print("=" * 65)
    print("\n  Open best_config.txt and copy the CFG block into train.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Pentest DDQN")
    parser.add_argument(
        "--groups", nargs="+", type=int,
        default=list(GROUPS.keys()),
        help="Which groups to run e.g. --groups 1 2 3  (default: all)"
    )
    args = parser.parse_args()

    valid = [g for g in args.groups if g in GROUPS]
    if not valid:
        print(f"Invalid groups. Choose from {list(GROUPS.keys())}")
        sys.exit(1)

    main(sorted(valid))