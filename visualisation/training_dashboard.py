"""
training_dashboard.py
---------------------
Live updating training dashboard — 4 subplots updated every N episodes.

Plots:
  1. Episode reward (raw + rolling average)
  2. Success rate % (last 100 episodes)
  3. Detection level per episode (avg)
  4. Epsilon decay curve
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


C_BG      = "#1A202C"
C_PANEL   = "#2D3748"
C_TEXT    = "#E2E8F0"
C_MUTED   = "#718096"
C_GREEN   = "#68D391"
C_YELLOW  = "#F6E05E"
C_RED     = "#FC8181"
C_BLUE    = "#63B3ED"
C_ORANGE  = "#F6AD55"


class TrainingDashboard:
    """
    Saves a training dashboard PNG every `update_every` episodes.

    Usage:
        dashboard = TrainingDashboard(save_dir="results/plots")
        # inside training loop:
        dashboard.update(episode, reward, success, caught, detection_avg, epsilon)
        # at end:
        dashboard.save_final()
    """

    def __init__(self, save_dir: str = "results/plots", update_every: int = 50):
        self.save_dir     = save_dir
        self.update_every = update_every
        os.makedirs(save_dir, exist_ok=True)

        # History
        self.episodes       = []
        self.rewards        = []
        self.successes      = []   # 1 = success, 0 = not
        self.caught_flags   = []   # 1 = caught
        self.detections     = []   # avg detection per episode
        self.epsilons       = []

    def update(
        self,
        episode:       int,
        reward:        float,
        success:       bool,
        caught:        bool,
        detection_avg: float,
        epsilon:       float,
    ):
        """Log one episode and optionally redraw the dashboard."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.successes.append(1 if success else 0)
        self.caught_flags.append(1 if caught else 0)
        self.detections.append(detection_avg)
        self.epsilons.append(epsilon)

        if episode % self.update_every == 0:
            self._draw(episode)

    def _rolling(self, data: list, window: int = 100) -> np.ndarray:
        arr = np.array(data, dtype=np.float32)
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window) / window, mode="valid")

    def _draw(self, episode: int):
        fig = plt.figure(figsize=(14, 8), facecolor=C_BG)
        plt.ion() 
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(C_PANEL)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#4A5568")

        eps = self.episodes
        n   = len(eps)

        # ── Plot 1: Episode Reward ─────────────────────
        ax1.plot(eps, self.rewards, color=C_BLUE, alpha=0.25, linewidth=0.8, label="Raw")
        if n >= 20:
            roll = self._rolling(self.rewards, 50)
            x_roll = eps[len(eps) - len(roll):]
            ax1.plot(x_roll, roll, color=C_GREEN, linewidth=2, label="Rolling avg (50)")
        ax1.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")
        ax1.set_title("Episode Reward", color=C_TEXT, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Episode", color=C_MUTED, fontsize=8)
        ax1.set_ylabel("Reward", color=C_MUTED, fontsize=8)
        ax1.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")

        # ── Plot 2: Success Rate ───────────────────────
        window = min(100, n)
        if n >= 10:
            roll_success = self._rolling(self.successes, window) * 100
            roll_caught  = self._rolling(self.caught_flags, window) * 100
            x_roll = eps[len(eps) - len(roll_success):]
            ax2.plot(x_roll, roll_success, color=C_GREEN, linewidth=2, label="Success %")
            ax2.plot(x_roll, roll_caught,  color=C_RED,   linewidth=2, label="Caught %")
            ax2.fill_between(x_roll, 0, roll_success, color=C_GREEN, alpha=0.1)
            ax2.fill_between(x_roll, 0, roll_caught,  color=C_RED,   alpha=0.1)
        ax2.set_ylim(0, 100)
        ax2.set_title(f"Success vs Caught Rate (last {window} eps)",
                      color=C_TEXT, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Episode", color=C_MUTED, fontsize=8)
        ax2.set_ylabel("%", color=C_MUTED, fontsize=8)
        ax2.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")

        # ── Plot 3: Detection Level ────────────────────
        ax3.plot(eps, self.detections, color=C_ORANGE, alpha=0.4, linewidth=0.8)
        if n >= 20:
            roll_det = self._rolling(self.detections, 50)
            x_roll   = eps[len(eps) - len(roll_det):]
            ax3.plot(x_roll, roll_det, color=C_RED, linewidth=2, label="Rolling avg")
        ax3.axhline(0.8, color=C_RED, linewidth=1, linestyle="--", alpha=0.5, label="Danger (0.8)")
        ax3.set_ylim(0, 1.05)
        ax3.set_title("Avg Detection Level per Episode",
                      color=C_TEXT, fontsize=10, fontweight="bold")
        ax3.set_xlabel("Episode", color=C_MUTED, fontsize=8)
        ax3.set_ylabel("Detection", color=C_MUTED, fontsize=8)
        ax3.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, edgecolor="#4A5568")

        # ── Plot 4: Epsilon Decay ──────────────────────
        ax4.plot(eps, self.epsilons, color=C_YELLOW, linewidth=2)
        ax4.fill_between(eps, 0, self.epsilons, color=C_YELLOW, alpha=0.15)
        ax4.set_ylim(0, 1.05)
        ax4.set_title("Epsilon (Exploration Rate)",
                      color=C_TEXT, fontsize=10, fontweight="bold")
        ax4.set_xlabel("Episode", color=C_MUTED, fontsize=8)
        ax4.set_ylabel("ε", color=C_MUTED, fontsize=8)

        # Main title
        success_rate = np.mean(self.successes[-100:]) * 100 if n >= 100 else np.mean(self.successes) * 100
        fig.suptitle(
            f"PENTEST DDQN — Episode {episode}   |   "
            f"Success Rate: {success_rate:.1f}%   |   "
            f"ε: {self.epsilons[-1]:.3f}",
            color=C_TEXT, fontsize=12, fontweight="bold", y=0.98,
        )

        plt.pause(0.001)
        plt.draw()
        plt.show(block=False)
        path = os.path.join(self.save_dir, f"dashboard_ep{episode:05d}.png")
        fig.savefig(path, dpi=100, facecolor=C_BG, bbox_inches="tight")

    def save_final(self):
        """Save final dashboard at end of training."""
        if self.episodes:
            self._draw(self.episodes[-1])
            # Also save a copy named 'final'
            import shutil
            latest = os.path.join(self.save_dir, f"dashboard_ep{self.episodes[-1]:05d}.png")
            final  = os.path.join(self.save_dir, "dashboard_final.png")
            shutil.copy(latest, final)
            print(f"Final dashboard saved → {final}")