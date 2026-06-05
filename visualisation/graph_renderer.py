"""
graph_renderer.py
-----------------
Live network graph visualisation for the pentest DDQN environment.

Shows:
  ⚪ Grey    → unvisited node
  🟡 Yellow  → agent's current position
  🟢 Green   → compromised by agent
  🔴 Red     → patched by defender
  🌟 Gold    → target node
  🔵 Blue    → entry node

Usage:
    renderer = GraphRenderer(env)
    renderer.render(env.get_render_state())   # call each step
    renderer.save_gif("results/replays/episode.gif")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — switch to "TkAgg" for live window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import imageio
import io


# ── Colour palette ──────────────────────────────────────────────
C_UNKNOWN    = "#2D3748"   # dark grey   — not visited
C_COMPROMISED= "#48BB78"   # green       — agent owns it
C_POSITION   = "#F6E05E"   # yellow      — current position
C_PATCHED    = "#FC8181"   # red         — defender closed it
C_TARGET     = "#F6AD55"   # gold        — the goal
C_ENTRY      = "#63B3ED"   # blue        — starting point
C_EDGE       = "#4A5568"   # edge colour
C_EDGE_PATH  = "#68D391"   # travelled edge colour
C_BG         = "#1A202C"   # dark background
C_TEXT       = "#E2E8F0"   # light text


# Fixed layout so the graph doesn't jump around between frames
FIXED_LAYOUT = {
    0: (0.0,  0.5),   # Entry
    1: (1.5,  0.5),   # Web Server
    2: (3.0,  0.8),   # App Server
    3: (3.0,  0.2),   # Mail Server
    4: (4.5,  0.9),   # Dev Box
    5: (5.5,  0.7),   # File Server
    6: (4.5,  0.1),   # DB Backup
    7: (6.5,  0.8),   # Admin Panel
    8: (8.0,  0.5),   # Internal DB (TARGET)
    9: (5.5,  1.0),   # Dev Server
    10: (1.5, -0.2),  # DMZ Server
    11: (3.0, -0.2),  # VPN Gateway
    12: (4.5, -0.2),  # Cloud Instance
    13: (2.2, -0.5),  # Honeypot
    14: (6.0, -0.2),  # Secondary DB
}


class GraphRenderer:
    """
    Renders the network graph as matplotlib figures.
    Captures frames to build a GIF replay.
    """

    def __init__(self, env, figsize=(14, 7)):
        self.env     = env
        self.G       = env.get_graph()
        self.pos     = FIXED_LAYOUT
        self.figsize = figsize
        self.frames  = []   # stores PNG bytes for GIF export

    def render(self, state: dict, save_frame: bool = True, title: str = "") -> plt.Figure:
        """
        Draw current state of the network.

        Args:
            state      : dict from env.get_render_state()
            save_frame : if True, captures frame for GIF
            title      : optional episode title string

        Returns:
            matplotlib Figure
        """
        fig, (ax_graph, ax_info) = plt.subplots(
            1, 2,
            figsize=self.figsize,
            gridspec_kw={"width_ratios": [3, 1]},
            facecolor=C_BG
        )

        self._draw_graph(ax_graph, state)
        self._draw_info_panel(ax_info, state, title)

        plt.tight_layout(pad=1.5)

        if save_frame:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80, facecolor=C_BG)
            buf.seek(0)
            self.frames.append(imageio.imread(buf))
            buf.close()

        return fig

    def _draw_graph(self, ax, state: dict):
        """Draw the network graph with coloured nodes."""
        ax.set_facecolor(C_BG)
        ax.axis("off")

        G           = state["graph"]
        position    = state["position"]
        compromised = state["compromised"]
        patched     = state["patched"]
        target      = state["target"]
        entry       = state["entry"]
        path        = state["path"]

        # ── Node colours ──────────────────────────────
        node_colors = []
        node_sizes  = []
        for node in G.nodes():
            if node == position:
                node_colors.append(C_POSITION)
                node_sizes.append(1800)
            elif node == target:
                node_colors.append(C_TARGET)
                node_sizes.append(2000)
            elif patched[node] == 1.0:
                node_colors.append(C_PATCHED)
                node_sizes.append(1200)
            elif compromised[node] == 1.0:
                node_colors.append(C_COMPROMISED)
                node_sizes.append(1400)
            elif node == entry:
                node_colors.append(C_ENTRY)
                node_sizes.append(1400)
            else:
                node_colors.append(C_UNKNOWN)
                node_sizes.append(1200)

        # ── Edge colours — highlight the path taken ──
        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i+1]))

        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if (u, v) in path_edges:
                edge_colors.append(C_EDGE_PATH)
                edge_widths.append(3.0)
            else:
                edge_colors.append(C_EDGE)
                edge_widths.append(1.2)

        # ── Draw ─────────────────────────────────────
        nx.draw_networkx_edges(
            G, self.pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            alpha=0.8,
        )

        nx.draw_networkx_nodes(
            G, self.pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            linewidths=2,
            edgecolors="#E2E8F0",
        )

        # Node labels — name + id
        labels = {n: f"{n}\n{G.nodes[n]['name']}" for n in G.nodes()}
        nx.draw_networkx_labels(
            G, self.pos, labels, ax=ax,
            font_size=6.5,
            font_color=C_BG,
            font_weight="bold",
        )

        # Edge noise labels
        edge_labels = {
            (u, v): f"n:{d['noise']:.2f}" for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, self.pos, edge_labels, ax=ax,
            font_size=5,
            font_color="#A0AEC0",
            bbox=dict(alpha=0),
        )

        # Detection bar at top
        detection = state["detection"]
        bar_color = "#FC8181" if detection > 0.8 else "#F6AD55" if detection > 0.5 else "#68D391"
        ax.set_title(
            f"Detection Level: {detection:.2f}  {'⚠ HIGH ALERT' if detection > 0.8 else ''}",
            color=bar_color,
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color=C_ENTRY,      label="Entry"),
            mpatches.Patch(color=C_UNKNOWN,    label="Unknown"),
            mpatches.Patch(color=C_COMPROMISED,label="Compromised"),
            mpatches.Patch(color=C_POSITION,   label="Agent Here"),
            mpatches.Patch(color=C_PATCHED,    label="Patched"),
            mpatches.Patch(color=C_TARGET,     label="Target"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower left",
            fontsize=7,
            facecolor="#2D3748",
            labelcolor=C_TEXT,
            edgecolor="#4A5568",
            framealpha=0.9,
        )

    def _draw_info_panel(self, ax, state: dict, title: str):
        """Draw the info sidebar."""
        ax.set_facecolor("#2D3748")
        ax.axis("off")

        G        = state["graph"]
        pos_name = G.nodes[state["position"]]["name"]
        step     = state["timestep"]
        detect   = state["detection"]
        path     = state["path"]

        lines = [
            ("PENTEST AGENT", "#F6AD55", 13, "bold"),
            ("", C_TEXT, 8, "normal"),
        ]

        if title:
            lines.append((title, "#63B3ED", 9, "bold"))
            lines.append(("", C_TEXT, 8, "normal"))

        lines += [
            ("POSITION", "#A0AEC0", 7, "normal"),
            (pos_name, C_TEXT, 9, "bold"),
            ("", C_TEXT, 6, "normal"),
            ("STEP", "#A0AEC0", 7, "normal"),
            (str(step), C_TEXT, 9, "bold"),
            ("", C_TEXT, 6, "normal"),
            ("DETECTION", "#A0AEC0", 7, "normal"),
            (f"{detect:.0%}", "#FC8181" if detect > 0.8 else C_TEXT, 9, "bold"),
            ("", C_TEXT, 6, "normal"),
            ("PATH TAKEN", "#A0AEC0", 7, "normal"),
        ]

        for node in path:
            lines.append((f"  → {G.nodes[node]['name']}", "#68D391", 7, "normal"))

        # Terminal result
        if state["success"]:
            lines += [
                ("", C_TEXT, 6, "normal"),
                ("✅ TARGET REACHED", "#68D391", 10, "bold"),
            ]
        elif state["caught"]:
            lines += [
                ("", C_TEXT, 6, "normal"),
                ("🚨 CAUGHT!", "#FC8181", 10, "bold"),
            ]

        y = 0.97
        for (text, color, size, weight) in lines:
            ax.text(
                0.1, y, text,
                transform=ax.transAxes,
                color=color,
                fontsize=size,
                fontweight=weight,
                va="top",
            )
            y -= 0.045 + (size - 7) * 0.003

    # ─────────────────────────────────────────────
    #  GIF EXPORT
    # ─────────────────────────────────────────────

    def save_gif(self, path: str, fps: int = 3):
        """Save all captured frames as an animated GIF."""
        if not self.frames:
            print("No frames to save.")
            return
        imageio.mimsave(path, self.frames, fps=fps, loop=0)
        print(f"GIF saved → {path}  ({len(self.frames)} frames)")

    def clear_frames(self):
        """Clear captured frames (call between episodes)."""
        self.frames = []

    def save_static(self, state: dict, path: str, title: str = ""):
        """Save a single frame as PNG."""
        fig = self.render(state, save_frame=False, title=title)
        fig.savefig(path, dpi=120, facecolor=C_BG, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}")


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.network_env import NetworkEnv

    env      = NetworkEnv(seed=42)
    renderer = GraphRenderer(env)

    state = env.reset()
    render_state = env.get_render_state()

    # Save initial state
    os.makedirs("results/plots", exist_ok=True)
    renderer.save_static(render_state, "results/plots/initial_state.png", title="Episode Start")
    print("Initial state saved to results/plots/initial_state.png")

    # Run a short random episode and save GIF
    os.makedirs("results/replays", exist_ok=True)
    renderer.clear_frames()

    for step in range(30):
        renderer.render(env.get_render_state(), save_frame=True)
        action = np.random.randint(0, env.action_size)
        _, _, done, _ = env.step(action)
        if done:
            renderer.render(env.get_render_state(), save_frame=True)
            break

    renderer.save_gif("results/replays/random_agent.gif", fps=2)