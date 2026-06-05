"""
visualiser.py
-------------
Interactive 3-phase visualiser for Pentest DDQN.

Phase 1 — Network Preview
    Shows the full network graph before training starts.
    Displays node names, vulnerability scores, edge noise levels.
    Press ENTER in terminal to begin training.

Phase 2 — Live Training
    Left panel  : network graph — agent moves in real time
    Right panel : 4 live training stats (reward, success %, detection, epsilon)
    Speed slider: drag to control how fast the simulation renders
    You can see exactly which path the agent is taking each episode.

Phase 3 — Test Replay (after training)
    Step-by-step replay of the trained agent.
    Slower speed so you can follow every decision.

Controls during training:
    Speed slider  → drag left (slow) or right (fast)
    Close window  → stops training and saves model

Usage:
    from visualisation.visualiser import Visualiser
    vis = Visualiser(env, agent)
    vis.show_network()           # Phase 1
    vis.start_training_view()    # Phase 2  (call inside training loop)
    vis.show_replay(states)      # Phase 3
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
import networkx as nx
import os
import time


# ── Colour palette ───────────────────────────────────────────────
C_BG          = "#1A202C"
C_PANEL       = "#2D3748"
C_TEXT        = "#E2E8F0"
C_MUTED       = "#718096"
C_GREEN       = "#68D391"
C_YELLOW      = "#F6E05E"
C_RED         = "#FC8181"
C_BLUE        = "#63B3ED"
C_ORANGE      = "#F6AD55"
C_PURPLE      = "#B794F4"

C_NODE_UNKNOWN     = "#4A5568"
C_NODE_ENTRY       = "#63B3ED"
C_NODE_COMPROMISED = "#68D391"
C_NODE_POSITION    = "#F6E05E"
C_NODE_PATCHED     = "#FC8181"
C_NODE_TARGET      = "#F6AD55"

C_EDGE_DEFAULT  = "#4A5568"
C_EDGE_TRAVELLED = "#68D391"


class Visualiser:
    """
    Three-phase interactive visualiser.

    Args:
        env   : NetworkEnv instance
        agent : DDQNAgent instance (can be None for Phase 1 only)
    """

    def __init__(self, env, agent=None):
        self.env   = env
        self.agent = agent
        self.G     = env.get_graph()
        self.pos   = self._get_layout(self.G)   # ← dynamic, not FIXED_LAYOUT

        # Training history for live plots
        self.ep_rewards   = []
        self.ep_successes = []
        self.ep_caught    = []
        self.ep_detections= []
        self.ep_epsilons  = []
        self.episodes     = []

        # Speed control (seconds between frames)
        self.render_delay = 0.05   # default: medium speed
        self.paused       = False

        # Internal figure handles
        self._fig  = None
        self._axes = {}

        os.makedirs("results/plots",   exist_ok=True)
        os.makedirs("results/replays", exist_ok=True)

    # ══════════════════════════════════════════════════════
    #  PHASE 1 — NETWORK PREVIEW
    # ══════════════════════════════════════════════════════
    def _get_layout(self, G):
        """Compute a fresh spring layout for any graph size."""
        # Use a fixed seed for visual consistency within an episode,
        # but it will naturally adapt to any number of nodes.
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=50)
        # Normalise to 0–1 range for consistent rendering
        xs = np.array([v[0] for v in pos.values()])
        ys = np.array([v[1] for v in pos.values()])
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)
        for k in pos:
            pos[k] = ((pos[k][0] - x_min) / x_range,
                    (pos[k][1] - y_min) / y_range)
        return pos

    def show_network(self):
        """
        Phase 1: Show the full network graph before training.
        Blocks until the window is closed.
        """
        self.G   = self.env.get_graph()          # ← refresh graph (may change each reset)
        self.pos = self._get_layout(self.G)      # ← recompute layout

        fig, ax = plt.subplots(figsize=(14, 7), facecolor=C_BG)
        ax.set_facecolor(C_BG)
        ax.axis("off")
        fig.suptitle(
            "PENTEST NETWORK — Close this window to start training",
            color=C_ORANGE, fontsize=13, fontweight="bold"
        )

        self._draw_network_preview(ax)
        self._draw_legend(ax)

        plt.tight_layout()
        plt.show(block=True)   # blocks here until user closes window
        plt.close(fig)

    def _draw_network_preview(self, ax):
        G = self.G
        target_nodes = self.env.target_nodes        # ← UPDATE: handled as list
        entry_node  = 0

        node_colors = []
        node_sizes  = []
        for node in G.nodes():
            if node in target_nodes:                # ← UPDATE: check membership in list
                node_colors.append(C_NODE_TARGET)
                node_sizes.append(2200)
            elif node == entry_node:
                node_colors.append(C_NODE_ENTRY)
                node_sizes.append(1800)
            elif G.nodes[node].get("is_honeypot", False):
                node_colors.append(C_PURPLE)      # ← distinct honeypot colour
                node_sizes.append(1500)
            else:
                node_colors.append(C_NODE_UNKNOWN)
                node_sizes.append(1500)

        nx.draw_networkx_edges(
            G, self.pos, ax=ax,
            edge_color=C_EDGE_DEFAULT, width=1.5,
            arrows=True, arrowsize=18, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08", alpha=0.7,
        )
        nx.draw_networkx_nodes(
            G, self.pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            linewidths=2, edgecolors=C_TEXT,
        )

        labels = {}
        for n in G.nodes():
            vuln = G.nodes[n]["vulnerability"]
            name = G.nodes[n]["name"]
            labels[n] = f"{n}: {name}\nvuln: {vuln:.0%}"
        nx.draw_networkx_labels(
            G, self.pos, labels, ax=ax,
            font_size=6, font_color=C_BG, font_weight="bold"
        )

        edge_labels = {
            (u, v): f"noise:{d['noise']:.2f}\ndiff:{d['difficulty']:.2f}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, self.pos, edge_labels, ax=ax,
            font_size=5, font_color=C_MUTED,
            bbox=dict(alpha=0, pad=0),
        )

        # Annotations use dynamic positions
        entry_xy  = self.pos[entry_node]
        ax.annotate("ENTRY POINT", xy=entry_xy,
                    xytext=(entry_xy[0] - 0.05, entry_xy[1] + 0.06),
                    color=C_BLUE, fontsize=8, fontweight="bold")
        
        # UPDATE: Loop through and annotate all dynamic targets from the list
        for t_idx in target_nodes:
            if t_idx in self.pos:
                target_xy = self.pos[t_idx]
                ax.annotate("★ TARGET", xy=target_xy,
                            xytext=(target_xy[0] + 0.02, target_xy[1] + 0.06),
                            color=C_ORANGE, fontsize=9, fontweight="bold")

    def _draw_legend(self, ax):
        legend_elements = [
            mpatches.Patch(color=C_NODE_ENTRY,       label="Entry Point"),
            mpatches.Patch(color=C_NODE_UNKNOWN,     label="Unknown Node"),
            mpatches.Patch(color=C_NODE_COMPROMISED, label="Compromised"),
            mpatches.Patch(color=C_NODE_POSITION,    label="Agent Position"),
            mpatches.Patch(color=C_NODE_PATCHED,     label="Patched (Defender)"),
            mpatches.Patch(color=C_NODE_TARGET,      label="Target Node"),
            mpatches.Patch(color=C_PURPLE,           label="Honeypot"),  
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower left", fontsize=8,
            facecolor=C_PANEL, labelcolor=C_TEXT,
            edgecolor=C_MUTED, framealpha=0.9,
        )

    # ══════════════════════════════════════════════════════
    #  PHASE 2 — LIVE TRAINING VIEW
    # ══════════════════════════════════════════════════════

    def init_training_view(self, render_state=None, speed=0.5):
        """
        Call once before the training loop starts.
        Sets up the live training window with speed slider.
        """
        plt.ion()
        self._fig = plt.figure(figsize=(16, 8), facecolor=C_BG)
        self._fig.canvas.manager.set_window_title("Pentest DDQN — Live Training")

        # Layout: network (left 60%) | stats (right 40%) | slider at bottom
        gs = gridspec.GridSpec(
            3, 2,
            figure=self._fig,
            height_ratios=[10, 10, 1],
            hspace=0.4, wspace=0.3,
        )

        self._axes["network"]   = self._fig.add_subplot(gs[0:2, 0])  # left, full height
        self._axes["reward"]    = self._fig.add_subplot(gs[0, 1])
        self._axes["success"]   = self._fig.add_subplot(gs[1, 1])

        # Speed slider at bottom
        ax_slider = self._fig.add_axes([0.1, 0.03, 0.55, 0.025], facecolor=C_PANEL)
        self._slider = Slider(
            ax_slider, "Speed", 0.0, 1.0,
            valinit=speed,
            color=C_GREEN,
        )
        self._slider.label.set_color(C_TEXT)
        self._slider.valtext.set_color(C_TEXT)

        # Speed label
        self._fig.text(
            0.02, 0.035, "← Slow    Fast →",
            color=C_MUTED, fontsize=8
        )

        # Episode info text
        self._ep_text = self._fig.text(
            0.68, 0.03, "Episode: 0",
            color=C_TEXT, fontsize=9, fontweight="bold"
        )

        self._slider.on_changed(self._on_speed_change)
        self._on_speed_change(speed)

        plt.show(block=False)
        plt.pause(0.1)

    def _on_speed_change(self, val):
        """Slider callback — maps 0–1 to delay 0.5s–0s."""
        self.render_delay = (1.0 - val) * 0.5

    def update_training_view(self, render_state: dict):
        """
        Wrapper keeping compatibility with updated step metrics framework updates.
        """
        if self._fig is None or len(self.episodes) == 0:
            # Fallback to direct frame updater layout logic if called before history compiles
            self.update_step(episode=1, step=render_state.get("timestep", 0), render_state=render_state)
            return
            
        self.update_step(episode=self.episodes[-1], step=render_state.get("timestep", 0), render_state=render_state)

    def update_step(self, episode: int, step: int, render_state: dict):
        """
        Call every step during training to update the network graph.
        This shows the agent moving through the network in real time.
        """
        if self._fig is None:
            return

        ax = self._axes["network"]
        ax.clear()
        ax.set_facecolor(C_BG)
        ax.axis("off")

        self._draw_live_network(ax, render_state, episode, step)

        eps_val = self.agent.epsilon if self.agent else 0.0
        self._ep_text.set_text(
            f"Episode: {episode}  |  Step: {step}  |  "
            f"Detection: {render_state['detection']:.2f}  |  "
            f"ε: {eps_val:.3f}"
        )

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        if self.render_delay > 0:
            time.sleep(self.render_delay)

    def _draw_live_network(self, ax, state: dict, episode: int, step: int):
        """Draw the network with agent's current position highlighted."""
        G           = state["graph"]
        position    = state["position"]
        compromised = state["compromised"]
        patched     = state["patched"]
        path        = state["path"]
        detection   = state["detection"]
        target_nodes = state.get("targets", [])    # ← UPDATE: Safe check for list key instead of "target"

        # Recompute layout if graph changed (new episode with new random network)
        if set(G.nodes()) != set(self.pos.keys()):
            self.G   = G
            self.pos = self._get_layout(G)

        node_colors = []
        node_sizes  = []
        for node in G.nodes():
            if node == position:
                node_colors.append(C_NODE_POSITION)
                node_sizes.append(2000)
            elif node in target_nodes:              # ← UPDATE: Check against target nodes list
                node_colors.append(C_NODE_TARGET)
                node_sizes.append(1800)
            elif G.nodes[node].get("is_honeypot", False):
                node_colors.append(C_PURPLE)
                node_sizes.append(1300)
            elif patched[node] == 1.0:
                node_colors.append(C_NODE_PATCHED)
                node_sizes.append(1300)
            elif compromised[node] == 1.0:
                node_colors.append(C_NODE_COMPROMISED)
                node_sizes.append(1500)
            elif node == 0:
                node_colors.append(C_NODE_ENTRY)
                node_sizes.append(1500)
            else:
                node_colors.append(C_NODE_UNKNOWN)
                node_sizes.append(1300)

        # Edge colours — highlight travelled path
        path_edges = set(zip(path[:-1], path[1:]))
        edge_colors = [C_EDGE_TRAVELLED if (u, v) in path_edges else C_EDGE_DEFAULT
                       for u, v in G.edges()]
        edge_widths = [3.0 if (u, v) in path_edges else 1.2
                       for u, v in G.edges()]

        nx.draw_networkx_edges(
            G, self.pos, ax=ax,
            edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowsize=14, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08", alpha=0.85,
        )
        nx.draw_networkx_nodes(
            G, self.pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            linewidths=2, edgecolors=C_TEXT,
        )

        labels = {n: f"{n}\n{G.nodes[n]['name']}" for n in G.nodes()}
        nx.draw_networkx_labels(
            G, self.pos, labels, ax=ax,
            font_size=6, font_color=C_BG, font_weight="bold"
        )

        # Detection bar title
        bar_len   = 20
        filled    = int(detection * bar_len)
        bar       = "█" * filled + "░" * (bar_len - filled)
        bar_color = C_RED if detection > 0.8 else C_ORANGE if detection > 0.5 else C_GREEN
        alert     = "  ⚠ HIGH ALERT" if detection > 0.8 else ""

        ax.set_title(
            f"Ep {episode} | Step {step} | Detection: [{bar}] {detection:.2f}{alert}",
            color=bar_color, fontsize=9, fontweight="bold", pad=6,
        )

        # Path annotation bottom left
        path_names = " → ".join(G.nodes[n]["name"] for n in path)
        ax.text(
            0.01, 0.01, f"Path: {path_names}",
            transform=ax.transAxes,
            color=C_GREEN, fontsize=6.5, va="bottom",
            wrap=True,
        )

        self._draw_legend(ax)

    def update_episode(self, episode: int, reward: float, success: bool,
                       caught: bool, detection_avg: float, epsilon: float):
        """
        Call at the end of each episode to update the stats plots.
        """
        if self._fig is None:
            return

        self.episodes.append(episode)
        self.ep_rewards.append(reward)
        self.ep_successes.append(1 if success else 0)
        self.ep_caught.append(1 if caught else 0)
        self.ep_detections.append(detection_avg)
        self.ep_epsilons.append(epsilon)

        self._draw_stats()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _rolling(self, data, window=50):
        arr = np.array(data, dtype=np.float32)
        if len(arr) < 2:
            return arr
        w = min(window, len(arr))
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    def _draw_stats(self):
        """Redraw the two stats subplots on the right."""
        eps = self.episodes
        n   = len(eps)

        # ── Reward plot ──────────────────────────────
        ax1 = self._axes["reward"]
        ax1.clear()
        ax1.set_facecolor(C_PANEL)
        ax1.tick_params(colors=C_MUTED, labelsize=7)
        for sp in ax1.spines.values():
            sp.set_edgecolor("#4A5568")

        ax1.plot(eps, self.ep_rewards, color=C_BLUE, alpha=0.2, linewidth=0.8)
        if n >= 10:
            roll = self._rolling(self.ep_rewards, 20)
            x_r  = eps[len(eps) - len(roll):]
            ax1.plot(x_r, roll, color=C_GREEN, linewidth=2, label="Avg reward")
        ax1.axhline(0, color=C_MUTED, linewidth=0.5, linestyle="--")
        ax1.set_title("Episode Reward", color=C_TEXT, fontsize=9, fontweight="bold")
        ax1.set_ylabel("Reward", color=C_MUTED, fontsize=7)

        # ── Success / Caught rate ────────────────────
        ax2 = self._axes["success"]
        ax2.clear()
        ax2.set_facecolor(C_PANEL)
        ax2.tick_params(colors=C_MUTED, labelsize=7)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#4A5568")

        if n >= 10:
            roll_s = self._rolling(self.ep_successes, 20) * 100
            roll_c = self._rolling(self.ep_caught,    20) * 100
            x_r    = eps[len(eps) - len(roll_s):]
            ax2.plot(x_r, roll_s, color=C_GREEN, linewidth=2, label="Success %")
            ax2.plot(x_r, roll_c, color=C_RED,   linewidth=2, label="Caught %")
            ax2.fill_between(x_r, 0, roll_s, color=C_GREEN, alpha=0.1)
            ax2.fill_between(x_r, 0, roll_c, color=C_RED,   alpha=0.1)
            ax2.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT,
                       edgecolor="#4A5568", loc="upper left")

        ax2.set_ylim(0, 105)
        ax2.set_title("Success vs Caught %", color=C_TEXT, fontsize=9, fontweight="bold")
        ax2.set_ylabel("%", color=C_MUTED, fontsize=7)
        ax2.set_xlabel("Episode", color=C_MUTED, fontsize=7)

    def save_training_snapshot(self):
        """Save current training view as PNG."""
        if self._fig and len(self.episodes) > 0:
            path = f"results/plots/training_ep{self.episodes[-1]:05d}.png"
            self._fig.savefig(path, dpi=100, facecolor=C_BG, bbox_inches="tight")

    def close_training_view(self):
        """Call after training loop ends."""
        if self._fig:
            self.save_training_snapshot()
            plt.ioff()
            plt.close(self._fig)
            self._fig = None

    # ══════════════════════════════════════════════════════
    #  PHASE 3 — TEST REPLAY
    # ══════════════════════════════════════════════════════

    def show_replay(self, model_path: str = "results/ddqn_final.pt", episodes: int = 3):
        """
        Phase 3: Step-by-step replay of the trained agent.
        Slower, narrated, saves a GIF.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from env.network_env_20 import RandomNetworkEnv
        import imageio, io

        env   = RandomNetworkEnv(seed=0, fixed_network=True)
        agent = self.agent

        if agent is None:
            print("No agent provided for replay.")
            return

        agent.epsilon = 0.0   # pure exploitation

        for ep in range(1, episodes + 1):
            frames = []
            state  = env.reset()

            fig, (ax_net, ax_info) = plt.subplots(
                1, 2, figsize=(14, 6),
                gridspec_kw={"width_ratios": [3, 1]},
                facecolor=C_BG,
            )
            fig.canvas.manager.set_window_title(f"Replay — Episode {ep}")
            plt.ion()
            plt.show(block=False)

            for step in range(200):
                valid  = env.get_valid_actions()
                action = agent.select_action(state, valid_actions=valid)
                next_state, reward, done, info = env.step(action)

                render_state = env.get_render_state()

                # Draw network
                ax_net.clear()
                ax_net.set_facecolor(C_BG)
                ax_net.axis("off")
                self._draw_live_network(ax_net, render_state, ep, step + 1)

                # Draw info panel
                ax_info.clear()
                ax_info.set_facecolor(C_PANEL)
                ax_info.axis("off")
                self._draw_replay_info(ax_info, env, action, reward, info, step + 1)

                plt.tight_layout()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                # Capture frame
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=80, facecolor=C_BG)
                buf.seek(0)
                import imageio as iio
                frames.append(iio.imread(buf))
                buf.close()

                time.sleep(0.3)   # slow for replay
                state = next_state

                if done:
                    time.sleep(1.0)  # pause on final frame
                    break

            # Save GIF
            gif_path = f"results/replays/replay_ep{ep:02d}.gif"
            import imageio as iio
            iio.mimsave(gif_path, frames, fps=3, loop=0)
            print(f"GIF saved → {gif_path}")

            plt.ioff()
            plt.close(fig)

    def _draw_replay_info(self, ax, env, action, reward, info, step):
        """Info sidebar for the replay view."""
        
        from env.network_env_20 import MAX_NODES as MAX_NODES_R
        act_type = "PROBE"  if action < MAX_NODES_R else "EXPLOIT"
        act_node = action   if action < MAX_NODES_R else action - MAX_NODES_R
        act_name = env.get_node_name(act_node)
        pos_name = env.get_node_name(info["position"])
        det      = info["detection"]

        lines = [
            ("REPLAY", C_ORANGE, 12, "bold"),
            ("", C_TEXT, 6, "normal"),
            ("ACTION", C_MUTED, 7, "normal"),
            (f"{act_type}", C_YELLOW if act_type == "PROBE" else C_GREEN, 10, "bold"),
            (act_name, C_TEXT, 8, "normal"),
            ("", C_TEXT, 5, "normal"),
            ("STEP", C_MUTED, 7, "normal"),
            (str(step), C_TEXT, 10, "bold"),
            ("", C_TEXT, 5, "normal"),
            ("REWARD", C_MUTED, 7, "normal"),
            (f"{reward:+.1f}", C_GREEN if reward > 0 else C_RED, 10, "bold"),
            ("", C_TEXT, 5, "normal"),
            ("POSITION", C_MUTED, 7, "normal"),
            (pos_name, C_TEXT, 8, "bold"),
            ("", C_TEXT, 5, "normal"),
            ("DETECTION", C_MUTED, 7, "normal"),
            (f"{det:.0%}", C_RED if det > 0.8 else C_ORANGE if det > 0.5 else C_GREEN, 10, "bold"),
        ]

        if env.success:
            lines += [("", C_TEXT, 6, "normal"), ("✅ SUCCESS!", C_GREEN, 12, "bold")]
        elif env.caught:
            lines += [("", C_TEXT, 6, "normal"), ("🚨 CAUGHT!", C_RED, 12, "bold")]

        y = 0.97
        for (text, color, size, weight) in lines:
            ax.text(0.1, y, text, transform=ax.transAxes,
                    color=color, fontsize=size, fontweight=weight, va="top")
            y -= 0.05 + (size - 7) * 0.004