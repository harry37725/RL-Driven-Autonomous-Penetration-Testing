"""
test_generalisation.py
----------------------
Runs the trained agent across 100 episodes (20 stored networks x 5 repeats).
Saves per-episode results and a summary to results_random/eval/.

Usage:
    python test_generalisation.py
    python test_generalisation.py --model results_random/ddqn_final.pt
    python test_generalisation.py --model results_random/checkpoints/ddqn_ep02500.pt --repeats 10
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from env.network_env_20 import RandomNetworkEnv, MAX_STEPS_R
from agent.ddqn_agent import DDQNAgent


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DEFAULT_MODEL   = "results_random/ddqn_final.pt"
DEFAULT_REPEATS = 5          # 20 networks x 5 repeats = 100 episodes
EVAL_DIR        = "results_random/eval"

# Verdict labels — ASCII only so Windows cp1252 never chokes
VERDICT_STRONG   = "[STRONG]   agent generalises well across all topologies"
VERDICT_MODERATE = "[MODERATE] some topologies still challenging"
VERDICT_WEAK     = "[WEAK]     partial transfer only"
VERDICT_POOR     = "[POOR]     agent has not generalised"

# Console icons — safe on all platforms
ICON_SUCCESS = "OK "
ICON_CAUGHT  = "!! "
ICON_TIMEOUT = "T  "


def _verdict(success_pct: float) -> str:
    if success_pct >= 60:
        return VERDICT_STRONG
    elif success_pct >= 40:
        return VERDICT_MODERATE
    elif success_pct >= 20:
        return VERDICT_WEAK
    else:
        return VERDICT_POOR


def run_eval(model_path: str, repeats: int = 5):
    """
    Test the agent on all 20 stored networks, each repeated `repeats` times.
    Total episodes = 20 x repeats.

    Args:
        model_path : path to .pt model file
        repeats    : how many times to run each network (default 5 -> 100 total)
    """
    os.makedirs(EVAL_DIR, exist_ok=True)

    total_episodes = 20 * repeats
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  GENERALISATION EVAL -- {total_episodes} episodes "
          f"(20 networks x {repeats} repeats)")
    print(f"  Model : {model_path}")
    print(f"  Output: {EVAL_DIR}/")
    print(f"{'='*65}\n")

    # ── Load agent ────────────────────────────────────────────────
    dummy_env = RandomNetworkEnv(seed=0, fixed_network=True)
    agent     = DDQNAgent(
        state_size  = dummy_env.state_size,
        action_size = dummy_env.action_size,
    )

    if os.path.exists(model_path):
        agent.load(model_path, partial_transfer=True)
        print(f"  Loaded weights from {model_path}")
    else:
        print(f"  WARNING: Model not found: {model_path}")
        print(f"    Running with untrained agent for baseline comparison.\n")

    agent.epsilon = 0.0   # pure exploitation

    # ── Run episodes ──────────────────────────────────────────────
    per_episode_results = []
    per_network_stats   = {i: {"success": 0, "caught": 0, "timeout": 0,
                                "steps": [], "path_lens": []}
                           for i in range(20)}

    ep_num = 0
    for repeat in range(1, repeats + 1):
        for net_idx in range(20):
            ep_num += 1

            env   = RandomNetworkEnv(seed=net_idx, fixed_network=True)
            state = env.reset()

            for _ in range(MAX_STEPS_R):
                valid  = env.get_valid_actions()
                action = agent.select_action(state, valid_actions=valid)
                state, _, done, info = env.step(action)
                if done:
                    break

            outcome = ("success" if env.success
                       else "caught"  if env.caught
                       else "timeout")

            result = {
                "episode":   ep_num,
                "repeat":    repeat,
                "network":   net_idx,
                "num_nodes": env.num_nodes,
                "outcome":   outcome,
                "success":   env.success,
                "caught":    env.caught,
                "timeout":   not env.success and not env.caught,
                "steps":     env.timestep,
                "path_len":  len(env.stats["path"]),
                "path":      env.stats["path"],
                "targets":   env.target_nodes,
                "ep_reward": float(env.episode_reward),
            }
            per_episode_results.append(result)

            s = per_network_stats[net_idx]
            s[outcome]       += 1
            s["steps"].append(env.timestep)
            s["path_lens"].append(len(env.stats["path"]))

            icon = (ICON_SUCCESS if env.success
                    else ICON_CAUGHT if env.caught
                    else ICON_TIMEOUT)
            print(f"  Ep {ep_num:3d} | net={net_idx:2d} | repeat={repeat} | "
                  f"nodes={env.num_nodes:2d} | {icon} {outcome:<7} | "
                  f"steps={env.timestep:3d} | path={len(env.stats['path'])}")

    # ── Aggregate summary ─────────────────────────────────────────
    n_success = sum(1 for r in per_episode_results if r["success"])
    n_caught  = sum(1 for r in per_episode_results if r["caught"])
    n_timeout = sum(1 for r in per_episode_results if r["timeout"])
    avg_steps = float(np.mean([r["steps"]     for r in per_episode_results]))
    avg_path  = float(np.mean([r["path_len"]  for r in per_episode_results]))
    avg_rew   = float(np.mean([r["ep_reward"] for r in per_episode_results]))
    sr        = n_success / total_episodes * 100
    verdict   = _verdict(sr)

    per_network_summary = {}
    for net_idx, s in per_network_stats.items():
        per_network_summary[net_idx] = {
            "success_rate": s["success"] / repeats * 100,
            "caught_rate":  s["caught"]  / repeats * 100,
            "timeout_rate": s["timeout"] / repeats * 100,
            "avg_steps":    float(np.mean(s["steps"])),
            "avg_path_len": float(np.mean(s["path_lens"])),
            "raw":          s,
        }

    summary = {
        "model":          model_path,
        "timestamp":      timestamp,
        "total_episodes": total_episodes,
        "repeats":        repeats,
        "n_networks":     20,
        "overall": {
            "success":      n_success,
            "caught":       n_caught,
            "timeout":      n_timeout,
            "success_pct":  round(sr, 1),
            "caught_pct":   round(n_caught  / total_episodes * 100, 1),
            "timeout_pct":  round(n_timeout / total_episodes * 100, 1),
            "avg_steps":    round(avg_steps, 1),
            "avg_path_len": round(avg_path,  1),
            "avg_reward":   round(avg_rew,   1),
        },
        "per_network": per_network_summary,
    }

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  Total episodes : {total_episodes}  (20 nets x {repeats} repeats)")
    print(f"  Success        : {n_success:3d} / {total_episodes}  ({sr:.1f}%)")
    print(f"  Caught         : {n_caught:3d} / {total_episodes}  "
          f"({summary['overall']['caught_pct']:.1f}%)")
    print(f"  Timeout        : {n_timeout:3d} / {total_episodes}  "
          f"({summary['overall']['timeout_pct']:.1f}%)")
    print(f"  Avg steps      : {avg_steps:.1f}")
    print(f"  Avg path len   : {avg_path:.1f}")
    print(f"  Avg reward     : {avg_rew:.1f}")

    print(f"\n  Per-network success rate ({repeats} runs each):")
    print(f"  {'Net':>4}  {'Nodes':>5}  {'Success%':>8}  {'Caught%':>7}  "
          f"{'Timeout%':>8}  {'AvgSteps':>8}")
    print(f"  {'-'*52}")
    for net_idx, s in per_network_summary.items():
        env_tmp = RandomNetworkEnv(seed=net_idx, fixed_network=True)
        n_nodes = env_tmp.num_nodes
        print(f"  {net_idx:4d}  {n_nodes:5d}  "
              f"{s['success_rate']:7.0f}%  "
              f"{s['caught_rate']:6.0f}%  "
              f"{s['timeout_rate']:7.0f}%  "
              f"{s['avg_steps']:8.1f}")

    print(f"\n  Generalisation score : {sr:.1f}%")
    print(f"  Verdict              : {verdict}")
    print(f"{'='*65}\n")

    # ── Save results — all files opened with utf-8 ────────────────
    detail_path = os.path.join(EVAL_DIR, f"eval_detail_{timestamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(per_episode_results, f, indent=2)

    # Clean raw counts for JSON serialisation
    for net_idx in summary["per_network"]:
        raw = summary["per_network"][net_idx]["raw"]
        summary["per_network"][net_idx]["raw"] = {
            k: ([int(x) for x in v] if isinstance(v, list) else v)
            for k, v in raw.items()
        }

    summary_path = os.path.join(EVAL_DIR, f"eval_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(EVAL_DIR, f"eval_report_{timestamp}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GENERALISATION EVAL REPORT\n")
        f.write(f"Generated : {timestamp}\n")
        f.write(f"Model     : {model_path}\n")
        f.write(f"Episodes  : {total_episodes} (20 nets x {repeats} repeats)\n\n")
        f.write("OVERALL\n")
        f.write(f"  Success : {n_success}/{total_episodes} ({sr:.1f}%)\n")
        f.write(f"  Caught  : {n_caught}/{total_episodes} "
                f"({summary['overall']['caught_pct']:.1f}%)\n")
        f.write(f"  Timeout : {n_timeout}/{total_episodes} "
                f"({summary['overall']['timeout_pct']:.1f}%)\n")
        f.write(f"  Verdict : {verdict}\n\n")
        f.write("PER-NETWORK BREAKDOWN\n")
        f.write(f"  {'Net':>3}  {'Success%':>8}  {'Caught%':>7}  "
                f"{'Timeout%':>8}  {'AvgSteps':>8}\n")
        f.write(f"  {'-'*44}\n")
        for net_idx, s in per_network_summary.items():
            f.write(f"  {net_idx:3d}    {s['success_rate']:7.0f}%  "
                    f"{s['caught_rate']:6.0f}%  "
                    f"{s['timeout_rate']:7.0f}%  "
                    f"{s['avg_steps']:8.1f}\n")
        f.write("\nPER-EPISODE LOG\n")
        f.write(f"  {'Ep':>3}  {'Net':>3}  {'Rep':>3}  {'Nodes':>5}  "
                f"{'Outcome':<7}  {'Steps':>5}  {'PathLen':>7}  {'Reward':>8}\n")
        f.write(f"  {'-'*58}\n")
        for r in per_episode_results:
            f.write(f"  {r['episode']:3d}  {r['network']:3d}  {r['repeat']:3d}  "
                    f"{r['num_nodes']:5d}  {r['outcome']:<7}  "
                    f"{r['steps']:5d}  {r['path_len']:7d}  "
                    f"{r['ep_reward']:8.1f}\n")

    print(f"  Files saved:")
    print(f"    {detail_path}")
    print(f"    {summary_path}")
    print(f"    {report_path}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DDQN on 20 stored networks x N repeats."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to model .pt file (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Repeats per network (default: 5 -> 100 total episodes)"
    )
    args = parser.parse_args()

    run_eval(model_path=args.model, repeats=args.repeats)