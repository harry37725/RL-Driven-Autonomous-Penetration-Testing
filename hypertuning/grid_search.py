"""
grid_search.py
--------------
Exhaustive grid search over 5 hyperparameters — 243 combinations.
Each combination trains from scratch for 300 episodes.

Parameters searched:
  lr                : [1e-4, 5e-4, 1e-3]
  gamma             : [0.99, 0.995, 0.999]
  epsilon_decay     : [0.999, 0.9995, 0.9999]
  batch_size        : [64, 128, 256]
  target_update_freq: [200, 350, 500]

Scoring metric: True Score = Success% - Caught%
  Rewards both finding the target AND staying stealthy.

Run:
    python grid_search.py

Resume after crash:
    python grid_search.py --resume

Run specific slice (for parallelism):
    python grid_search.py --start 0   --end 81
    python grid_search.py --start 81  --end 162
    python grid_search.py --start 162 --end 243

Output:
    results/grid_search/
        logs/
            results.csv          ← all 243 results
            best_config.txt      ← top 10 configs + recommended
            progress.json        ← checkpoint for resume
        plots/
            top10_comparison.png
            heatmaps/            ← param interaction heatmaps
            summary_dashboard.png
"""

import os
import sys
import csv
import json
import time
import argparse
import itertools
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import threading
import queue

sys.path.insert(0, os.path.dirname(__file__))

from env.network_env  import NetworkEnv
from agent.ddqn_agent import DDQNAgent


# ─────────────────────────────────────────────
#  GRID DEFINITION
# ─────────────────────────────────────────────

GRID = {
    "lr":                 [1e-4,  5e-4,  1e-3],
    "gamma":              [0.99,  0.995, 0.999],
    "epsilon_decay":      [0.999, 0.9995, 0.9999],
    "batch_size":         [64,    128,   256],
    "target_update_freq": [200,   350,   500],
}

FIXED = {
    "epsilon_start":  1.0,
    "epsilon_end":    0.05,
    "buffer_capacity": 50_000,
}

EPISODES       = 500
EVAL_WINDOW    = 100     # last N episodes for scoring
ROLLING_WINDOW = 30
SAVE_DIR       = "results/grid_search"
PROGRESS_FILE  = f"{SAVE_DIR}/logs/progress.json"


# ─────────────────────────────────────────────
#  BUILD ALL COMBINATIONS
# ─────────────────────────────────────────────

def build_grid():
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = []
    for vals in itertools.product(*values):
        cfg = dict(zip(keys, vals))
        cfg.update(FIXED)
        combos.append(cfg)
    return combos


# ─────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────

def run_combo(cfg: dict, combo_id: int) -> dict:
    """Train one combination from scratch. Returns result dict."""
    env   = NetworkEnv(defender_level=1, seed=42)
    agent = DDQNAgent(
        state_size  = env.state_size,
        action_size = env.action_size,
        lr                 = cfg["lr"],
        gamma              = cfg["gamma"],
        epsilon_start      = cfg["epsilon_start"],
        epsilon_end        = cfg["epsilon_end"],
        epsilon_decay      = cfg["epsilon_decay"],
        batch_size         = cfg["batch_size"],
        buffer_capacity    = cfg["buffer_capacity"],
        target_update_freq = cfg["target_update_freq"],
    )

    rewards   = []
    successes = []
    caught    = []
    timeouts  = []
    losses    = []

    for ep in range(EPISODES):
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
        rewards.append(ep_reward)
        successes.append(1 if env.stats["success"] else 0)
        caught.append(1 if env.stats["caught"]  else 0)
        timeouts.append(1 if env.stats["timeout"] else 0)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

    # Score on last EVAL_WINDOW episodes
    success_rate = float(np.mean(successes[-EVAL_WINDOW:]) * 100)
    caught_rate  = float(np.mean(caught[-EVAL_WINDOW:])    * 100)
    timeout_rate = float(np.mean(timeouts[-EVAL_WINDOW:])  * 100)
    avg_reward   = float(np.mean(rewards[-EVAL_WINDOW:]))
    true_score   = success_rate - caught_rate
    final_loss   = float(np.mean(losses[-EVAL_WINDOW:])) if losses else 0.0
    final_eps    = float(agent.epsilon)

    return {
        "combo_id":    combo_id,
        "lr":          cfg["lr"],
        "gamma":       cfg["gamma"],
        "epsilon_decay": cfg["epsilon_decay"],
        "batch_size":  cfg["batch_size"],
        "target_update_freq": cfg["target_update_freq"],
        "success_rate": success_rate,
        "caught_rate":  caught_rate,
        "timeout_rate": timeout_rate,
        "avg_reward":   avg_reward,
        "true_score":   true_score,
        "final_loss":   final_loss,
        "final_epsilon": final_eps,
        # Store curves for plotting top configs
        "_rewards":   rewards,
        "_successes": successes,
        "_caught":    caught,
    }


# ─────────────────────────────────────────────
#  LIVE DASHBOARD  (background thread)
# ─────────────────────────────────────────────

C_BG    = "#1A202C"
C_PANEL = "#2D3748"
C_TEXT  = "#E2E8F0"
C_MUTED = "#718096"
C_GREEN = "#68D391"
C_BLUE  = "#63B3ED"
C_RED   = "#FC8181"
C_GOLD  = "#F6AD55"


class GridSearchDashboard:
    """
    Live dashboard showing:
      - Current combo progress (success/reward curves)
      - Leaderboard of top 10 so far
      - Progress bar + ETA
    """

    def __init__(self, total: int):
        self.total   = total
        self._queue  = queue.Queue(maxsize=3)
        self._running = True
        self._thread  = threading.Thread(target=self._gui_loop, daemon=True)
        self._thread.start()
        time.sleep(1.5)

    def _gui_loop(self):
        plt.ion()
        fig = plt.figure(figsize=(15, 8), facecolor=C_BG)
        fig.canvas.manager.set_window_title("Grid Search — Live")

        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
        ax_s   = fig.add_subplot(gs[0, 0])   # success curve
        ax_r   = fig.add_subplot(gs[0, 1])   # reward curve
        ax_lb  = fig.add_subplot(gs[0, 2])   # leaderboard
        ax_prog= fig.add_subplot(gs[1, :2])  # progress bar area
        ax_info= fig.add_subplot(gs[1, 2])   # current config info

        for ax in [ax_s, ax_r, ax_lb, ax_prog, ax_info]:
            ax.set_facecolor(C_PANEL)
            ax.tick_params(colors=C_MUTED, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#4A5568")

        ax_s.set_ylim(0, 100)
        ax_r.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")
        ax_prog.set_xlim(0, self.total)
        ax_prog.set_ylim(0, 1)
        ax_prog.axis("off")

        line_s, = ax_s.plot([], [], color=C_GREEN, linewidth=2)
        line_r, = ax_r.plot([], [], color=C_BLUE,  linewidth=2)
        prog_bar = ax_prog.barh([0], [0], color=C_GREEN, height=0.4)[0]
        prog_text = ax_prog.text(self.total / 2, 0, "", color=C_TEXT,
                                  fontsize=11, ha="center", va="center",
                                  fontweight="bold")
        eta_text  = ax_prog.text(self.total / 2, -0.35, "", color=C_MUTED,
                                  fontsize=8, ha="center", va="center")

        ax_s.set_title("Success % — Current Combo", color=C_TEXT, fontsize=8,
                        fontweight="bold")
        ax_r.set_title("Reward — Current Combo",    color=C_TEXT, fontsize=8,
                        fontweight="bold")
        ax_lb.set_title("🏆 Top 10 So Far",          color=C_GOLD, fontsize=8,
                         fontweight="bold")
        ax_info.set_title("Current Config",           color=C_TEXT, fontsize=8,
                           fontweight="bold")

        fig.suptitle("GRID SEARCH — 243 Combinations",
                     color=C_GOLD, fontsize=12, fontweight="bold")

        plt.show(block=False)
        fig.canvas.flush_events()

        leaderboard = []

        while self._running:
            latest = None
            try:
                while True:
                    latest = self._queue.get_nowait()
            except queue.Empty:
                pass

            if latest is not None:
                data = latest

                # ── Success curve ──────────────────────
                successes = data.get("successes", [])
                rewards   = data.get("rewards",   [])
                done_n    = data.get("done_combos", 0)
                cfg       = data.get("cfg", {})
                eta_str   = data.get("eta", "")
                best_so_far = data.get("best_so_far", None)

                if successes:
                    eps    = list(range(1, len(successes) + 1))
                    roll_s = _rolling(successes, ROLLING_WINDOW) * 100
                    roll_r = _rolling(rewards,   ROLLING_WINDOW)
                    x_s    = eps[len(eps) - len(roll_s):]
                    x_r    = eps[len(eps) - len(roll_r):]

                    line_s.set_data(x_s, roll_s)
                    line_r.set_data(x_r, roll_r)
                    ax_s.set_xlim(0, len(eps) + 5)
                    ax_r.set_xlim(0, len(eps) + 5)
                    if len(roll_r) > 1:
                        pad = max(abs(float(roll_r.min())),
                                  abs(float(roll_r.max()))) * 0.2 + 10
                        ax_r.set_ylim(float(roll_r.min()) - pad,
                                       float(roll_r.max()) + pad)
                    cur_s = float(roll_s[-1]) if len(roll_s) else 0
                    ax_s.set_title(
                        f"Success %: {cur_s:.0f}% — Combo {done_n}/{self.total}",
                        color=C_TEXT, fontsize=8, fontweight="bold"
                    )

                # ── Progress bar ────────────────────────
                prog_bar.set_width(done_n)
                pct = done_n / self.total * 100
                prog_text.set_text(f"{done_n}/{self.total} combos  ({pct:.0f}%)")
                eta_text.set_text(f"ETA: {eta_str}")
                ax_prog.set_xlim(0, self.total)

                # ── Leaderboard ─────────────────────────
                if best_so_far:
                    leaderboard = best_so_far[:10]
                    ax_lb.clear()
                    ax_lb.set_facecolor(C_PANEL)
                    ax_lb.tick_params(colors=C_MUTED, labelsize=6)
                    for sp in ax_lb.spines.values():
                        sp.set_edgecolor("#4A5568")

                    scores = [r["true_score"] for r in leaderboard]
                    colors = [C_GOLD if i == 0 else C_GREEN if i < 3 else C_BLUE
                              for i in range(len(leaderboard))]
                    bars   = ax_lb.barh(range(len(leaderboard)), scores,
                                         color=colors, edgecolor="#4A5568", alpha=0.85)
                    labels = [
                        f"lr={r['lr']} γ={r['gamma']} ε={r['epsilon_decay']}"
                        for r in leaderboard
                    ]
                    ax_lb.set_yticks(range(len(leaderboard)))
                    ax_lb.set_yticklabels(labels, fontsize=5.5, color=C_MUTED)
                    for bar, r in zip(bars, leaderboard):
                        ax_lb.text(
                            max(r["true_score"] - 0.5, 0.5),
                            bar.get_y() + bar.get_height() / 2,
                            f"✓{r['success_rate']:.0f}% ✗{r['caught_rate']:.0f}%",
                            va="center", color=C_TEXT, fontsize=5.5
                        )
                    ax_lb.set_title("🏆 Top 10 (Success−Caught)",
                                     color=C_GOLD, fontsize=8, fontweight="bold")

                # ── Config info panel ───────────────────
                ax_info.clear()
                ax_info.set_facecolor(C_PANEL)
                ax_info.axis("off")
                if cfg:
                    lines = [
                        ("CURRENT CONFIG", C_GOLD, 8, "bold"),
                        ("", C_TEXT, 4, "normal"),
                        (f"lr          = {cfg.get('lr', '')}", C_TEXT, 7, "normal"),
                        (f"gamma       = {cfg.get('gamma', '')}", C_TEXT, 7, "normal"),
                        (f"eps_decay   = {cfg.get('epsilon_decay', '')}", C_TEXT, 7, "normal"),
                        (f"batch_size  = {cfg.get('batch_size', '')}", C_TEXT, 7, "normal"),
                        (f"target_upd  = {cfg.get('target_update_freq', '')}", C_TEXT, 7, "normal"),
                        ("", C_TEXT, 4, "normal"),
                    ]
                    if best_so_far:
                        best = best_so_far[0]
                        lines += [
                            ("BEST SO FAR", C_GREEN, 8, "bold"),
                            (f"Score: {best['true_score']:.1f}", C_GREEN, 9, "bold"),
                            (f"✓ {best['success_rate']:.0f}%  ✗ {best['caught_rate']:.0f}%", C_TEXT, 7, "normal"),
                            (f"lr={best['lr']}", C_MUTED, 6, "normal"),
                            (f"γ={best['gamma']}", C_MUTED, 6, "normal"),
                            (f"ε={best['epsilon_decay']}", C_MUTED, 6, "normal"),
                            (f"b={best['batch_size']}", C_MUTED, 6, "normal"),
                            (f"t={best['target_update_freq']}", C_MUTED, 6, "normal"),
                        ]
                    y = 0.95
                    for (text, color, size, weight) in lines:
                        ax_info.text(0.05, y, text, transform=ax_info.transAxes,
                                     color=color, fontsize=size, fontweight=weight,
                                     va="top", fontfamily="monospace")
                        y -= 0.085 + (size - 6) * 0.01

                try:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                except Exception:
                    pass

            try:
                fig.canvas.flush_events()
            except Exception:
                pass
            time.sleep(0.05)

        try:
            plt.ioff()
            plt.close(fig)
        except Exception:
            pass

    def update(self, successes, rewards, done_combos, cfg, eta, best_so_far):
        payload = {
            "successes":   list(successes),
            "rewards":     list(rewards),
            "done_combos": done_combos,
            "cfg":         dict(cfg),
            "eta":         eta,
            "best_so_far": list(best_so_far[:10]) if best_so_far else [],
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(payload)
            except Exception:
                pass

    def close(self):
        self._running = False
        try:
            self._thread.join(timeout=3.0)
        except Exception:
            pass


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

def _rolling(data, window=30):
    arr = np.array(data, dtype=np.float32)
    w   = min(window, len(arr))
    if w < 2:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#4A5568")
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=8, fontweight="bold", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=7)


def plot_top10(results: list, save_path: str):
    """Compare top 10 configs side by side."""
    top10 = sorted(results, key=lambda r: r["true_score"], reverse=True)[:10]
    eps   = list(range(1, EPISODES + 1))

    fig = plt.figure(figsize=(18, 12), facecolor=C_BG)
    fig.suptitle("TOP 10 CONFIGURATIONS — Grid Search Results",
                 color=C_GOLD, fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35)

    ax_score   = fig.add_subplot(gs[0, :2])
    ax_scatter = fig.add_subplot(gs[0, 2:])
    ax_success = fig.add_subplot(gs[1, :2])
    ax_caught  = fig.add_subplot(gs[1, 2:])
    ax_reward  = fig.add_subplot(gs[2, :2])
    ax_table   = fig.add_subplot(gs[2, 2:])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (res, color) in enumerate(zip(top10, colors)):
        label = f"#{i+1} lr={res['lr']} γ={res['gamma']}"
        roll_s = _rolling(res["_successes"], ROLLING_WINDOW) * 100
        roll_r = _rolling(res["_rewards"],   ROLLING_WINDOW)
        roll_c = _rolling(res["_caught"],    ROLLING_WINDOW) * 100
        x_s    = eps[len(eps) - len(roll_s):]

        ax_success.plot(x_s, roll_s, color=color, linewidth=1.8,
                        label=label, alpha=0.85)
        ax_caught.plot(x_s, roll_c,  color=color, linewidth=1.8, alpha=0.85)
        ax_reward.plot(x_s, roll_r,  color=color, linewidth=1.8, alpha=0.85)

    # Score bar chart
    labels = [f"#{i+1}" for i in range(len(top10))]
    scores = [r["true_score"]   for r in top10]
    s_rate = [r["success_rate"] for r in top10]
    c_rate = [r["caught_rate"]  for r in top10]

    x     = np.arange(len(top10))
    width = 0.3
    ax_score.bar(x - width, s_rate, width, color=[c for c in colors],
                  alpha=0.9,  label="Success %", edgecolor="#4A5568")
    ax_score.bar(x,          c_rate, width, color=[c for c in colors],
                  alpha=0.4,  label="Caught %",  edgecolor="#4A5568")
    ax_score.bar(x + width,  scores, width, color=[c for c in colors],
                  alpha=0.7,  label="True Score", edgecolor="#4A5568")
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(labels, color=C_MUTED, fontsize=7)
    ax_score.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
                     edgecolor="#4A5568")

    # Scatter: success vs caught
    all_results_scatter = results
    ax_scatter.scatter(
        [r["caught_rate"]  for r in all_results_scatter],
        [r["success_rate"] for r in all_results_scatter],
        c=[r["true_score"] for r in all_results_scatter],
        cmap="RdYlGn", alpha=0.6, s=20, edgecolors="none"
    )
    for i, res in enumerate(top10):
        ax_scatter.scatter(res["caught_rate"], res["success_rate"],
                            color=colors[i], s=80, zorder=5,
                            edgecolors="white", linewidths=0.8)
        ax_scatter.annotate(f"#{i+1}", (res["caught_rate"], res["success_rate"]),
                             textcoords="offset points", xytext=(4, 4),
                             color=colors[i], fontsize=6)
    ax_scatter.set_xlabel("Caught %", color=C_MUTED, fontsize=7)
    ax_scatter.set_ylabel("Success %", color=C_MUTED, fontsize=7)

    # Table of top 10
    ax_table.axis("off")
    ax_table.set_facecolor(C_PANEL)
    headers = ["#", "lr", "γ", "ε_decay", "batch", "t_upd", "✓%", "✗%", "Score"]
    rows = []
    for i, r in enumerate(top10):
        rows.append([
            f"#{i+1}",
            str(r["lr"]),
            str(r["gamma"]),
            str(r["epsilon_decay"]),
            str(r["batch_size"]),
            str(r["target_update_freq"]),
            f"{r['success_rate']:.0f}%",
            f"{r['caught_rate']:.0f}%",
            f"{r['true_score']:.1f}",
        ])

    tbl = ax_table.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#2D3748" if row % 2 == 0 else "#1A202C")
        cell.set_text_props(color=C_GOLD if row == 0 else C_TEXT)
        cell.set_edgecolor("#4A5568")
    ax_table.set_title("Top 10 Full Config Table",
                        color=C_TEXT, fontsize=8, fontweight="bold")

    style_ax(ax_score,   "Score Breakdown",       xlabel="Rank", ylabel="%")
    style_ax(ax_scatter, "All Combos: Success vs Caught (green=better)")
    style_ax(ax_success, "Success % Curves",       ylabel="%")
    style_ax(ax_caught,  "Caught % Curves",        ylabel="%")
    style_ax(ax_reward,  "Reward Curves",          ylabel="Reward")

    ax_success.set_ylim(0, 100)
    ax_caught.set_ylim(0, 60)
    ax_success.legend(fontsize=5.5, facecolor=C_PANEL, labelcolor=C_TEXT,
                       edgecolor="#4A5568", loc="upper left",
                       ncol=2)
    ax_reward.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")

    fig.savefig(save_path, dpi=110, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top10 plot → {save_path}")


def plot_heatmaps(results: list, save_path: str):
    """
    2D heatmaps showing interaction between each pair of parameters.
    Cell colour = average true score for that parameter combination.
    """
    params = list(GRID.keys())
    pairs  = [(params[i], params[j])
              for i in range(len(params))
              for j in range(i + 1, len(params))]

    n_pairs = len(pairs)   # 10 pairs
    cols    = 4
    rows    = (n_pairs + cols - 1) // cols

    fig = plt.figure(figsize=(18, rows * 4), facecolor=C_BG)
    fig.suptitle("PARAMETER INTERACTION HEATMAPS\n"
                 "(Avg True Score = Success% − Caught% for each pair)",
                 color=C_GOLD, fontsize=12, fontweight="bold", y=1.0)

    for idx, (p1, p2) in enumerate(pairs):
        ax  = fig.add_subplot(rows, cols, idx + 1)
        ax.set_facecolor(C_PANEL)

        v1 = GRID[p1]
        v2 = GRID[p2]

        matrix = np.zeros((len(v1), len(v2)))
        counts = np.zeros((len(v1), len(v2)))

        for r in results:
            i = v1.index(r[p1])
            j = v2.index(r[p2])
            matrix[i, j] += r["true_score"]
            counts[i, j]  += 1

        counts[counts == 0] = 1
        matrix /= counts

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                       vmin=-20, vmax=50)
        ax.set_xticks(range(len(v2)))
        ax.set_yticks(range(len(v1)))
        ax.set_xticklabels([str(v) for v in v2], fontsize=6.5, color=C_MUTED)
        ax.set_yticklabels([str(v) for v in v1], fontsize=6.5, color=C_MUTED)
        ax.set_xlabel(p2, color=C_MUTED, fontsize=7)
        ax.set_ylabel(p1, color=C_MUTED, fontsize=7)
        ax.set_title(f"{p1[:6]} vs {p2[:6]}", color=C_TEXT,
                     fontsize=7, fontweight="bold")

        for i in range(len(v1)):
            for j in range(len(v2)):
                ax.text(j, i, f"{matrix[i, j]:.0f}",
                        ha="center", va="center",
                        color="black" if matrix[i, j] > 20 else C_TEXT,
                        fontsize=6.5, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=100, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmaps   → {save_path}")


def save_results_csv(results: list, path: str):
    if not results:
        return
    keys = ["combo_id", "lr", "gamma", "epsilon_decay", "batch_size",
            "target_update_freq", "success_rate", "caught_rate",
            "timeout_rate", "avg_reward", "true_score",
            "final_loss", "final_epsilon"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["true_score"], reverse=True):
            writer.writerow({k: r[k] for k in keys})
    print(f"  CSV        → {path}")


def save_best_config(results: list, path: str):
    top10 = sorted(results, key=lambda r: r["true_score"], reverse=True)[:10]
    best  = top10[0]

    lines = [
        "GRID SEARCH — BEST HYPERPARAMETER CONFIG\n",
        f"Total combinations tested : {len(results)}\n",
        f"Episodes per combo        : {EPISODES}\n",
        f"Scoring metric            : True Score = Success% - Caught%\n",
        "=" * 50 + "\n\n",
        "TOP 10 CONFIGS:\n\n",
    ]
    for i, r in enumerate(top10):
        lines.append(
            f"  #{i+1:2d}  Score={r['true_score']:+.1f}  "
            f"✓{r['success_rate']:.0f}%  ✗{r['caught_rate']:.0f}%  "
            f"lr={r['lr']}  γ={r['gamma']}  "
            f"ε={r['epsilon_decay']}  b={r['batch_size']}  "
            f"t={r['target_update_freq']}\n"
        )

    lines += [
        "\n" + "=" * 50 + "\n",
        "RECOMMENDED CONFIG — COPY INTO train.py CFG:\n\n",
        f'    "lr":                 {best["lr"]},\n',
        f'    "gamma":              {best["gamma"]},\n',
        f'    "epsilon_start":      1.0,\n',
        f'    "epsilon_end":        0.05,\n',
        f'    "epsilon_decay":      {best["epsilon_decay"]},\n',
        f'    "batch_size":         {best["batch_size"]},\n',
        f'    "buffer_capacity":    50_000,\n',
        f'    "target_update_freq": {best["target_update_freq"]},\n',
        f'\n    # True Score: {best["true_score"]:+.1f}  '
        f'(Success: {best["success_rate"]:.0f}%  Caught: {best["caught_rate"]:.0f}%)\n',
    ]

    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  Best cfg   → {path}")


# ─────────────────────────────────────────────
#  PROGRESS / RESUME
# ─────────────────────────────────────────────

def save_progress(done_ids: list, results: list):
    data = {
        "done_ids": done_ids,
        "results":  [{k: v for k, v in r.items()
                      if not k.startswith("_")} for r in results],
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return [], []
    with open(PROGRESS_FILE) as f:
        data = json.load(f)
    return data.get("done_ids", []), data.get("results", [])


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main(start: int = 0, end: int = 243, resume: bool = False):
    os.makedirs(f"{SAVE_DIR}/logs",    exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/plots/heatmaps", exist_ok=True)

    combos = build_grid()
    total  = len(combos)

    # Apply slice
    combos_slice = combos[start:end]
    n_slice      = len(combos_slice)

    # Resume
    done_ids    = []
    all_results = []

    if resume:
        done_ids, all_results = load_progress()
        print(f"  Resuming — {len(done_ids)} combos already done")

    # Filter already done
    combos_todo = [(i + start, cfg) for i, cfg in enumerate(combos_slice)
                   if (i + start) not in done_ids]

    print("\n" + "=" * 65)
    print("  GRID SEARCH — HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)
    print(f"  Total combos   : {total}")
    print(f"  This slice     : {start}–{end}  ({n_slice} combos)")
    print(f"  Remaining      : {len(combos_todo)}")
    print(f"  Episodes each  : {EPISODES}")
    print(f"  Scoring metric : True Score = Success% − Caught%")
    print(f"  Est. time      : ~{len(combos_todo) * 2.5 / 60:.1f} hrs")
    print("=" * 65 + "\n")

    dashboard  = GridSearchDashboard(total=len(combos_todo))
    start_time = time.time()

    pbar = tqdm(combos_todo, desc="Grid Search", unit="combo")

    for done_count, (combo_id, cfg) in enumerate(pbar, 1):
        # ETA calculation
        elapsed  = time.time() - start_time
        avg_time = elapsed / done_count if done_count > 0 else 0
        remaining = (len(combos_todo) - done_count) * avg_time
        eta_str  = str(timedelta(seconds=int(remaining)))

        # Run combo — stream updates to dashboard every 20 episodes
        env   = NetworkEnv(defender_level=1, seed=42)
        agent = DDQNAgent(
            state_size  = env.state_size,
            action_size = env.action_size,
            lr                 = cfg["lr"],
            gamma              = cfg["gamma"],
            epsilon_start      = cfg["epsilon_start"],
            epsilon_end        = cfg["epsilon_end"],
            epsilon_decay      = cfg["epsilon_decay"],
            batch_size         = cfg["batch_size"],
            buffer_capacity    = cfg["buffer_capacity"],
            target_update_freq = cfg["target_update_freq"],
        )

        rewards   = []
        successes = []
        caught    = []
        timeouts  = []
        losses    = []

        for ep in range(EPISODES):
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
            rewards.append(ep_reward)
            successes.append(1 if env.stats["success"] else 0)
            caught.append(1 if env.stats["caught"]  else 0)
            timeouts.append(1 if env.stats["timeout"] else 0)
            losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

            if ep % 20 == 0:
                best_so_far = sorted(all_results,
                                     key=lambda r: r["true_score"],
                                     reverse=True)
                dashboard.update(successes, rewards, done_count,
                                 cfg, eta_str, best_so_far)

        # Score
        success_rate = float(np.mean(successes[-EVAL_WINDOW:]) * 100)
        caught_rate  = float(np.mean(caught[-EVAL_WINDOW:])    * 100)
        true_score   = success_rate - caught_rate

        result = {
            "combo_id":    combo_id,
            "lr":          cfg["lr"],
            "gamma":       cfg["gamma"],
            "epsilon_decay": cfg["epsilon_decay"],
            "batch_size":  cfg["batch_size"],
            "target_update_freq": cfg["target_update_freq"],
            "success_rate": success_rate,
            "caught_rate":  caught_rate,
            "timeout_rate": float(np.mean(timeouts[-EVAL_WINDOW:]) * 100),
            "avg_reward":   float(np.mean(rewards[-EVAL_WINDOW:])),
            "true_score":   true_score,
            "final_loss":   float(np.mean(losses[-EVAL_WINDOW:])) if losses else 0.0,
            "final_epsilon": float(agent.epsilon),
            "_rewards":     rewards,
            "_successes":   successes,
            "_caught":      caught,
        }

        all_results.append(result)
        done_ids.append(combo_id)

        # Save progress every 10 combos
        if done_count % 10 == 0:
            save_progress(done_ids, all_results)

        best = max(all_results, key=lambda r: r["true_score"])
        pbar.set_postfix({
            "✓":     f"{success_rate:.0f}%",
            "✗":     f"{caught_rate:.0f}%",
            "score": f"{true_score:+.0f}",
            "best":  f"{best['true_score']:+.0f}",
        })

    # ── All done ─────────────────────────────
    dashboard.close()
    time.sleep(0.8)

    print("\n  Saving results...")
    save_results_csv(all_results,
                     f"{SAVE_DIR}/logs/results.csv")
    save_best_config(all_results,
                     f"{SAVE_DIR}/logs/best_config.txt")
    save_progress(done_ids, all_results)

    print("\n  Generating plots...")
    plot_top10(all_results,
               f"{SAVE_DIR}/plots/top10_comparison.png")
    plot_heatmaps(all_results,
                  f"{SAVE_DIR}/plots/heatmaps/param_interactions.png")

    best = max(all_results, key=lambda r: r["true_score"])

    print("\n" + "=" * 65)
    print("  GRID SEARCH COMPLETE")
    print("=" * 65)
    print(f"  Combos tested  : {len(all_results)}")
    print(f"  Best score     : {best['true_score']:+.1f}")
    print(f"  Best success   : {best['success_rate']:.0f}%")
    print(f"  Best caught    : {best['caught_rate']:.0f}%")
    print(f"  Best config    :")
    print(f"    lr={best['lr']}  gamma={best['gamma']}")
    print(f"    epsilon_decay={best['epsilon_decay']}")
    print(f"    batch_size={best['batch_size']}")
    print(f"    target_update_freq={best['target_update_freq']}")
    print(f"\n  Results  → {SAVE_DIR}/logs/results.csv")
    print(f"  Best cfg → {SAVE_DIR}/logs/best_config.txt")
    print(f"  Top10    → {SAVE_DIR}/plots/top10_comparison.png")
    print(f"  Heatmaps → {SAVE_DIR}/plots/heatmaps/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for Pentest DDQN")
    parser.add_argument("--start",  type=int, default=0,   help="Start combo index")
    parser.add_argument("--end",    type=int, default=243, help="End combo index")
    parser.add_argument("--resume", action="store_true",   help="Resume from checkpoint")
    args = parser.parse_args()

    main(start=args.start, end=args.end, resume=args.resume)