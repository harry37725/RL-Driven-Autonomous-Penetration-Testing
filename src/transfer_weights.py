"""
transfer_weights.py
-------------------
Transfers shared layer weights from the trained 10-node model
to a new 15-node model.

What gets transferred:
  shared.1 (Dense 128→128, ReLU)   ← identical, copy directly
  shared.3 (Dense 128→128, ReLU)   ← identical, copy directly
  shared.5 (Dense 128→64,  ReLU)   ← identical, copy directly
  value_stream     (64→32→1)        ← identical, copy directly
  advantage_stream (64→32→N)        ← partial: copy 64→32 layer only
                                       output layer differs (20 vs 30 actions)

What does NOT transfer:
  shared.0 (input Dense: 32→128 vs 47→128)  ← different input size
  advantage_stream final layer (→20 vs →30)  ← different action count

Run:
    python transfer_weights.py

    # Custom paths:
    python transfer_weights.py --source results/ddqn_final.pt
                               --output results_large/ddqn_transfer.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from agent.ddqn_agent import DDQNAgent, DuelingDQN


def transfer_weights(
    source_path: str = "results/ddqn_final.pt",
    output_path: str = "results_large/ddqn_transfer.pt",
    source_state_size: int = 32,
    source_action_size: int = 20,
    target_state_size: int  = 48,
    target_action_size: int = 30,
    hidden: int = 128,
):
    """
    Transfer shared layer weights from source model to target model.

    Args:
        source_path       : path to trained 10-node model
        output_path       : where to save the new 15-node model
        source_state_size : state size of source model (32)
        source_action_size: action size of source model (20)
        target_state_size : state size of target model (48)
        target_action_size: action size of target model (30)
        hidden            : hidden layer size (must match both models)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("\n" + "="*55)
    print("  TRANSFER LEARNING — Weight Migration")
    print("="*55)
    print(f"  Source : {source_path}")
    print(f"           state={source_state_size}  actions={source_action_size}")
    print(f"  Target : {output_path}")
    print(f"           state={target_state_size}  actions={target_action_size}")
    print("="*55 + "\n")

    # ── Load source model ──────────────────────────────
    if not os.path.exists(source_path):
        print(f"  ❌ Source model not found: {source_path}")
        print(f"     Make sure you've trained the 10-node model first.")
        sys.exit(1)

    checkpoint = torch.load(source_path, map_location="cpu")
    source_state = checkpoint["online_net"]

    print("  Source model layers:")
    for name, param in source_state.items():
        print(f"    {name:45s} {str(list(param.shape)):20s}")

    # ── Build target model ─────────────────────────────
    target_net = DuelingDQN(target_state_size, target_action_size, hidden)

    print("\n  Target model layers:")
    for name, param in target_net.named_parameters():
        print(f"    {name:45s} {str(list(param.shape)):20s}")

    # ── Transfer layer by layer ────────────────────────
    target_state = target_net.state_dict()
    transferred  = []
    skipped      = []

    transfer_map = {
        # source layer name → target layer name
        # shared[0] is input layer — SKIP (different input size)
        "shared.2.weight": "shared.2.weight",   # Dense(128→128)
        "shared.2.bias":   "shared.2.bias",
        "shared.4.weight": "shared.4.weight",   # Dense(128→128)
        "shared.4.bias":   "shared.4.bias",
        "shared.6.weight": "shared.6.weight",   # Dense(128→64)
        "shared.6.bias":   "shared.6.bias",
        # Value stream — fully transferable
        "value_stream.0.weight": "value_stream.0.weight",
        "value_stream.0.bias":   "value_stream.0.bias",
        "value_stream.2.weight": "value_stream.2.weight",
        "value_stream.2.bias":   "value_stream.2.bias",
        # Advantage stream — first layer only (64→32 is same)
        "advantage_stream.0.weight": "advantage_stream.0.weight",
        "advantage_stream.0.bias":   "advantage_stream.0.bias",
        # advantage_stream.2 is output layer (→20 vs →30) — SKIP
    }

    print("\n  Transferring weights:")
    for src_name, tgt_name in transfer_map.items():
        if src_name in source_state and tgt_name in target_state:
            src_param = source_state[src_name]
            tgt_param = target_state[tgt_name]

            if src_param.shape == tgt_param.shape:
                target_state[tgt_name] = src_param.clone()
                transferred.append(f"  ✅  {src_name:40s} → {tgt_name}")
            else:
                skipped.append(f"  ⚠️  {src_name:40s}  shape mismatch: "
                                f"{list(src_param.shape)} vs {list(tgt_param.shape)}")
        else:
            skipped.append(f"  ⚠️  {src_name:40s}  not found in source or target")

    for msg in transferred: print(msg)
    print()
    print("  Skipped (expected — different size):")
    for msg in skipped: print(msg)

    # ── Input layer: smart initialisation ─────────────
    # The new input layer goes from 47→128 instead of 32→128.
    # Strategy: copy the first 32 weights (same meaning as before)
    #           initialise the new 15 weights (new nodes) near zero
    print("\n  Input layer partial transfer (shared.0):")
    src_input_w = source_state.get("shared.0.weight")   # shape: [128, 32]
    src_input_b = source_state.get("shared.0.bias")     # shape: [128]

    if src_input_w is not None:
        tgt_input_w = target_state["shared.0.weight"]    # shape: [128, 47]

        # Copy first 32 columns (original state features)
        tgt_input_w[:, :source_state_size] = src_input_w.clone()

        # Initialise new 15 columns (new node features) with small random values
        # Using Xavier initialisation scaled down so new features start quiet
        new_cols = target_state_size - source_state_size
        nn.init.xavier_uniform_(tgt_input_w[:, source_state_size:])
        tgt_input_w[:, source_state_size:] *= 0.1   # scale down new features

        target_state["shared.0.weight"] = tgt_input_w
        target_state["shared.0.bias"]   = src_input_b.clone()
        print(f"  ✅  Copied first {source_state_size} cols, "
              f"initialised {new_cols} new cols (scaled Xavier)")
    else:
        print("  ⚠️  shared.0.weight not found in source — using random init")

    # ── Output layer: smart initialisation ────────────
    # Advantage stream output: 32→20 → 32→30
    # Copy first 20 action values (same actions as before)
    # Initialise new 10 actions near zero
    print("\n  Output layer partial transfer (advantage_stream.2):")
    src_adv_w = source_state.get("advantage_stream.2.weight")  # [20, 32]
    src_adv_b = source_state.get("advantage_stream.2.bias")    # [20]

    if src_adv_w is not None:
        tgt_adv_w = target_state["advantage_stream.2.weight"]   # [30, 32]
        tgt_adv_b = target_state["advantage_stream.2.bias"]     # [30]

        # Copy first 20 rows (original actions)
        tgt_adv_w[:source_action_size, :] = src_adv_w.clone()
        tgt_adv_b[:source_action_size]    = src_adv_b.clone()

        # New 10 actions: initialise near zero
        nn.init.xavier_uniform_(tgt_adv_w[source_action_size:, :])
        tgt_adv_w[source_action_size:, :] *= 0.05
        tgt_adv_b[source_action_size:]     = 0.0

        target_state["advantage_stream.2.weight"] = tgt_adv_w
        target_state["advantage_stream.2.bias"]   = tgt_adv_b
        print(f"  ✅  Copied first {source_action_size} actions, "
              f"initialised {target_action_size - source_action_size} new actions")
    else:
        print("  ⚠️  advantage_stream.2.weight not found — using random init")

    # ── Load weights into model ────────────────────────
    target_net.load_state_dict(target_state)

    # ── Build full agent and save ──────────────────────
    agent = DDQNAgent(
        state_size         = target_state_size,
        action_size        = target_action_size,
        lr                 = 5e-4,
        gamma              = 0.99,
        epsilon_start      = 0.5,    # start at 50% exploration — not full random
        epsilon_end        = 0.05,
        epsilon_decay      = 0.999,
        batch_size         = 256,
        buffer_capacity    = 50_000,
        target_update_freq = 350,
    )

    # Copy transferred weights into agent's online and target networks
    agent.online_net.load_state_dict(target_net.state_dict())
    agent.target_net.load_state_dict(target_net.state_dict())

    torch.save({
        "online_net":  agent.online_net.state_dict(),
        "target_net":  agent.target_net.state_dict(),
        "optimizer":   agent.optimizer.state_dict(),
        "epsilon":     agent.epsilon,
        "steps_done":  0,
    }, output_path)

    print(f"\n  ✅  Saved transferred model → {output_path}")

    # ── Verify ────────────────────────────────────────
    print("\n  Verification — forward pass with random state:")
    test_state = torch.randn(1, target_state_size)
    with torch.no_grad():
        q_values = agent.online_net(test_state)
    print(f"    Input shape  : {list(test_state.shape)}")
    print(f"    Output shape : {list(q_values.shape)}")
    print(f"    Q-value range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    print(f"    Epsilon set  : {agent.epsilon}")

    # Summary
    n_transferred = len(transferred)
    n_skipped     = len(skipped)
    total_params  = sum(p.numel() for p in agent.online_net.parameters())
    print(f"\n  Summary:")
    print(f"    Layers transferred : {n_transferred}")
    print(f"    Layers reinitialised: {n_skipped + 2}")
    print(f"    Total parameters   : {total_params:,}")
    print(f"\n  ✅  Transfer complete!")
    print(f"\n  Next step: load this model in train.py with:")
    print(f'    agent.load("{output_path}")')
    print(f"    agent.epsilon = 0.5   # already set in saved model")
    print("="*55 + "\n")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer weights from 10-node to 15-node model")
    parser.add_argument("--source", default="results/ddqn_final.pt",
                        help="Path to trained 10-node model")
    parser.add_argument("--output", default="results_large/ddqn_transfer.pt",
                        help="Where to save the transferred model")
    args = parser.parse_args()

    transfer_weights(
        source_path  = args.source,
        output_path  = args.output,
    )