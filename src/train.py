"""
train.py
--------
Main training loop for the Pentest DDQN agent.
"""

import os
import sys
import random
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from env.network_env_20 import RandomNetworkEnv as NetworkEnv   # ← uses stored pool
from agent.ddqn_agent import DDQNAgent
from visualisation.visualiser import Visualiser


CFG = {
    "num_episodes":       5000,
    "max_steps_per_ep":   200,
    "lr":                 5e-4,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      0.999,
    "batch_size":         256,
    "buffer_capacity":    50_000,
    "target_update_freq": 350,
    "defender_level":     1,
    "seed":               None,
    "render_every_step":  False,
    "update_stats_every": 1,
    "save_dir":           "results_random",
    "checkpoint_dir":     "results_random/checkpoints",
}

# Load the 20 stored networks — seed=i maps to pool index i
NETWORK_POOL = [
    NetworkEnv(seed=i, fixed_network=True)
    for i in range(20)
]


def get_inherit_ratio(episode, total=5000):
    return max(0.30, 0.80 - (episode / total) * 0.50)


def train():
    os.makedirs(CFG["save_dir"], exist_ok=True)
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    env   = random.choice(NETWORK_POOL)
    state = env.reset()

    agent = DDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        lr=CFG["lr"],
        gamma=CFG["gamma"],
        epsilon_start=CFG["epsilon_start"],
        epsilon_end=CFG["epsilon_end"],
        epsilon_decay=CFG["epsilon_decay"],
        batch_size=CFG["batch_size"],
        buffer_capacity=CFG["buffer_capacity"],
        target_update_freq=CFG["target_update_freq"],
    )

    agent.load("results_large/ddqn_transfer.pt")
    agent.epsilon = 0.50

    vis = Visualiser(env, agent)

    print("\n" + "="*55)
    print("  PENTEST DDQN — STARTING")
    print("="*55)
    print(f"  Episodes    : {CFG['num_episodes']}")
    print(f"  Pool size   : {len(NETWORK_POOL)} stored networks")
    print(f"  State size  : {env.state_size}")
    print(f"  Action size : {env.action_size}")
    print("="*55)

    print("\n[Phase 1] Network graph opening...")
    print("          Study the topology, then CLOSE the window to start training.\n")
    vis.show_network()

    print("[Phase 2] Training starting...\n")
    vis.init_training_view()

    history = {"rewards": [], "successes": [], "caught": []}
    pbar    = tqdm(range(1, CFG["num_episodes"] + 1), desc="Training", unit="ep")

    for episode in pbar:
        env   = random.choice(NETWORK_POOL)
        state = env.reset()

        # Sync visualiser to new env
        vis.env = env
        vis.update_step(episode, 0, env.get_render_state())

        ep_reward = 0.0
        det_vals  = []

        for step in range(CFG["max_steps_per_ep"]):
            if CFG["render_every_step"]:
                vis.update_step(episode, step + 1, env.get_render_state())

            valid  = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid)

            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            ep_reward += reward
            det_vals.append(info["detection"])
            state = next_state

            if done:
                vis.update_step(episode, step + 1, env.get_render_state())
                break

        # Always render final state — covers timeout (loop exhausted without done)
        vis.update_step(episode, step + 1, env.get_render_state())

        if episode > 200:
            agent.decay_epsilon()

        if not hasattr(agent, '_last_known_level'):
            agent._last_known_level = 1

        success = env.stats["success"]
        caught  = env.stats["caught"]
        det_avg = float(np.mean(det_vals))

        history["rewards"].append(ep_reward)
        history["successes"].append(int(success))
        history["caught"].append(int(caught))

        if episode % CFG["update_stats_every"] == 0:
            vis.update_episode(episode, ep_reward, success, caught, det_avg, agent.epsilon)

        recent_success = np.mean(history["successes"][-100:]) * 100
        pbar.set_postfix({
            "reward":  f"{ep_reward:+.0f}",
            "success": f"{recent_success:.0f}%",
            "ε":       f"{agent.epsilon:.3f}",
            "net":     f"{env._pool_index}",    # which of the 20 networks is active
        })

        if episode % 500 == 0:
            ckpt = os.path.join(CFG["checkpoint_dir"], f"ddqn_ep{episode:05d}.pt")
            agent.save(ckpt)
            vis.save_training_snapshot()

    vis.close_training_view()

    model_path = os.path.join(CFG["save_dir"], "ddqn_final.pt")
    agent.save(model_path)

    last100_success = np.mean(history["successes"][-100:]) * 100
    last100_caught  = np.mean(history["caught"][-100:])    * 100
    last100_reward  = np.mean(history["rewards"][-100:])

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    print(f"  Last 100 eps success : {last100_success:.1f}%")
    print(f"  Last 100 eps caught  : {last100_caught:.1f}%")
    print(f"  Avg reward           : {last100_reward:+.1f}")
    print(f"  Model saved          → {model_path}")
    print("="*55)

    print("\n[Phase 3] Opening replay of trained agent...\n")
    vis.show_replay(model_path=model_path, episodes=3)
    print("\nDone. Check results/ for all saved outputs.")


if __name__ == "__main__":
    train()