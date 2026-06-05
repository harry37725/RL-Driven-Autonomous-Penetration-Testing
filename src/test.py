"""
test.py
-------
Load a trained DDQN agent and watch it navigate the network.

Run:
    python test.py                        # uses results/ddqn_final.pt
    python test.py --model path/to/model  # use specific checkpoint
    python test.py --episodes 10          # run 10 test episodes

What happens:
  1. Loads trained model (epsilon set to 0 — pure exploitation, no random moves)
  2. Runs N test episodes
  3. Renders each step to the terminal AND saves a GIF
  4. Prints final stats
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

# from env.network_env              import NetworkEnv, NUM_NODES
from env.network_env_large import NetworkEnvLarge as NetworkEnv
from env.network_env_large import NUM_NODES_L as NUM_NODES
from agent.ddqn_agent             import DDQNAgent
from visualisation.graph_renderer import GraphRenderer


# ── ANSI colours for terminal output ──────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

BAR_FULL  = "█"
BAR_EMPTY = "░"


def detection_bar(level: float, width: int = 20) -> str:
    filled = int(level * width)
    bar    = BAR_FULL * filled + BAR_EMPTY * (width - filled)
    color  = RED if level > 0.8 else YELLOW if level > 0.5 else GREEN
    return f"{color}{bar}{RESET}"


def print_step(step, action, node_name, action_type, reward, info, env):
    pos_name = env.get_node_name(info["position"])
    det      = info["detection"]
    bar      = detection_bar(det)

    action_str = f"{BLUE}PROBE  {RESET}" if action_type == "probe" else f"{YELLOW}EXPLOIT{RESET}"

    print(
        f"  Step {step:3d} | {action_str} [{node_name:15s}] "
        f"| Reward: {reward:+7.1f} "
        f"| Pos: {pos_name:15s} "
        f"| Detection: {bar} {det:.2f}"
    )


def run_test(model_path: str, num_episodes: int = 5, save_gif: bool = True, render_delay: float = 0.0):
    print(f"\n{'='*60}")
    print(f"  {BOLD}PENTEST DDQN — TEST MODE{RESET}")
    print(f"{'='*60}")
    print(f"  Model    : {model_path}")
    print(f"  Episodes : {num_episodes}")
    print(f"{'='*60}\n")

    env   = NetworkEnv(seed=0)
    agent = DDQNAgent(state_size=env.state_size, action_size=env.action_size)

    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"{RED}Model not found at {model_path}. Running with untrained agent.{RESET}")

    # Pure exploitation — no random moves during testing
    agent.epsilon = 0.0

    renderer = GraphRenderer(env) if save_gif else None

    os.makedirs("results_large/replays", exist_ok=True)
    os.makedirs("results_large/plots",   exist_ok=True)

    # ── Test loop ─────────────────────────────
    results = {"success": 0, "caught": 0, "timeout": 0, "rewards": []}

    for ep in range(1, num_episodes + 1):
        print(f"\n{BOLD}── Episode {ep}/{num_episodes} ─────────────────────────────{RESET}")

        state = env.reset()
        if renderer:
            renderer.clear_frames()
            renderer.render(env.get_render_state(), save_frame=True, title=f"Episode {ep} — Start")

        ep_reward = 0.0

        for step in range(200):
            valid  = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid)

            action_type = "probe"   if action < NUM_NODES else "exploit"
            node_id     = action    if action < NUM_NODES else action - NUM_NODES
            node_name   = env.get_node_name(node_id)

            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            print_step(step + 1, action, node_name, action_type, reward, info, env)

            if renderer:
                renderer.render(
                    env.get_render_state(),
                    save_frame=True,
                    title=f"Episode {ep} — Step {step+1}"
                )

            if render_delay > 0:
                time.sleep(render_delay)

            state = next_state

            if done:
                # Final frame
                if renderer:
                    renderer.render(
                        env.get_render_state(),
                        save_frame=True,
                        title=f"Episode {ep} — {'✅ SUCCESS' if env.success else '🚨 CAUGHT'}"
                    )
                break

        # Episode result
        results["rewards"].append(ep_reward)
        if env.success:
            results["success"] += 1
            print(f"\n  {GREEN}{BOLD}✅ TARGET REACHED{RESET} in {step+1} steps | Total reward: {ep_reward:+.1f}")
            print(f"  Path: {' → '.join(env.get_node_name(n) for n in env.stats['path'])}")
        elif env.caught:
            results["caught"] += 1
            print(f"\n  {RED}{BOLD}🚨 CAUGHT{RESET} at step {step+1} | Total reward: {ep_reward:+.1f}")
        else:
            results["timeout"] += 1
            print(f"\n  {YELLOW}⏱  TIMEOUT{RESET} | Total reward: {ep_reward:+.1f}")

        if renderer and renderer.frames:
            gif_path = f"results_large/replays/episode_{ep:02d}.gif"
            renderer.save_gif(gif_path, fps=3)

    # ── Summary ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {BOLD}TEST RESULTS — {num_episodes} episodes{RESET}")
    print(f"{'='*60}")
    print(f"  {GREEN}✅ Success : {results['success']:3d}  ({results['success']/num_episodes*100:.1f}%){RESET}")
    print(f"  {RED}🚨 Caught  : {results['caught']:3d}  ({results['caught']/num_episodes*100:.1f}%){RESET}")
    print(f"  {YELLOW}⏱  Timeout : {results['timeout']:3d}  ({results['timeout']/num_episodes*100:.1f}%){RESET}")
    print(f"  Avg reward : {np.mean(results['rewards']):+.1f}")
    print(f"{'='*60}\n")

    # Save summary bar chart
    _save_result_chart(results, num_episodes)


def _save_result_chart(results: dict, num_episodes: int):
    """Save a simple result breakdown bar chart."""
    C_BG    = "#1A202C"
    C_PANEL = "#2D3748"
    C_TEXT  = "#E2E8F0"

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=C_BG)
    ax.set_facecolor(C_PANEL)

    categories = ["Success", "Caught", "Timeout"]
    values     = [results["success"], results["caught"], results["timeout"]]
    colors     = ["#68D391", "#FC8181", "#F6AD55"]

    bars = ax.bar(categories, values, color=colors, edgecolor="#4A5568", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val}\n({val/num_episodes*100:.0f}%)",
            ha="center", va="bottom",
            color=C_TEXT, fontsize=9, fontweight="bold"
        )

    ax.set_ylim(0, max(values) * 1.3 + 1)
    ax.set_title("Test Results", color=C_TEXT, fontsize=12, fontweight="bold")
    ax.tick_params(colors="#718096")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A5568")

    path = "results_large/plots/test_results.png"
    fig.savefig(path, dpi=100, facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Result chart saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained DDQN pentest agent")
    parser.add_argument("--model",    default="results_large/checkpoints/ddqn_ep00500.pt", help="Path to model file")
    parser.add_argument("--episodes", type=int, default=5,             help="Number of test episodes")
    parser.add_argument("--no-gif",   action="store_true",             help="Skip GIF rendering")
    parser.add_argument("--delay",    type=float, default=0.0,         help="Seconds between steps (for slow-mo)")
    args = parser.parse_args()

    run_test(
        model_path   = args.model,
        num_episodes = args.episodes,
        save_gif     = not args.no_gif,
        render_delay = args.delay,
    )