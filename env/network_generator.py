# generate_and_store_pool.py
# ──────────────────────────────────────────────────────────────────────
# Script to generate a persistent pool of 20 deterministic network graphs 
# and save them to a file for consistent training and testing.

import os
import sys
import random
import pickle
import numpy as np
import networkx as nx

# Import the existing large graph layout constants
from network_env_large import EDGES_LARGE, NODES_LARGE, NUM_NODES_L

# ─────────────────────────────────────────────
#  CONSTANTS (Matching your environment)
# ─────────────────────────────────────────────
MIN_NODES              = 10
MAX_NODES              = 18
BACKBONE_INHERIT_RATIO = 0.95

NODE_TYPE_NAMES = [
    "Web Server", "App Server", "Mail Server", "Dev Box",
    "File Server", "DB Backup", "Admin Panel", "Dev Server",
    "DMZ Server", "VPN Gateway", "Cloud Node", "Proxy Server",
    "Build Server", "Auth Server", "Log Server", "Backup Node",
    "Internal DB",
]

def build_backbone_path(n, target_idx, honeypot_idx, rng):
    """Builds a guaranteed path from 0 to a target node, avoiding the honeypot."""
    middle_nodes = [i for i in range(1, n) if i != honeypot_idx and i != target_idx]
    rng.shuffle(middle_nodes)
    n_hops = min(len(middle_nodes), int(rng.integers(1, 4)))
    path = [0] + list(middle_nodes[:n_hops]) + [target_idx]
    
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]

def generate_single_network(seed, num_nodes=None):
    """Generates a single network graph using the exact specifications from your environment."""
    rng = np.random.default_rng(seed)
    
    n = num_nodes if num_nodes is not None else int(rng.integers(MIN_NODES, MAX_NODES + 1))
    G = nx.DiGraph()

    # Dynamic targets (randomly pick 1 or 2 targets)
    num_targets = int(rng.choice([1, 2]))
    
    # Position honeypot carefully away from high-traffic nodes
    bad_honeypot_slots = {8, 10, 12, 13, 14}
    honeypot_candidates = [i for i in range(2, max(3, n - 1)) if i not in bad_honeypot_slots]
    if not honeypot_candidates:
        honeypot_candidates = list(range(2, max(3, n - 1)))
    honeypot_idx = int(rng.choice(honeypot_candidates))

    # Assign target positions anywhere away from Entry (0) and Honeypot
    possible_target_slots = [i for i in range(1, n) if i != honeypot_idx]
    if len(possible_target_slots) < num_targets:
        num_targets = len(possible_target_slots)
        
    target_indices = [int(x) for x in rng.choice(possible_target_slots, size=num_targets, replace=False)]

    # Node attributes assignment
    fixed_node_names = {i: NODES_LARGE[i]["name"] for i in range(min(n, NUM_NODES_L))}
    for i in range(n):
        if i == 0:
            name, vuln, val, is_hp = "Entry Point", float(rng.uniform(0.85, 0.98)), 0, False
        elif i in target_indices:
            t_num = target_indices.index(i) + 1
            name, vuln, val, is_hp = f"Internal DB {t_num}", float(rng.uniform(0.15, 0.35)), 100, False
        elif i == honeypot_idx:
            name, vuln, val, is_hp = "Honeypot", float(rng.uniform(0.80, 0.95)), 0, True
        else:
            name = fixed_node_names.get(i, NODE_TYPE_NAMES[min(i, len(NODE_TYPE_NAMES)-2)])
            vuln, val, is_hp = float(rng.uniform(0.25, 0.85)), int(rng.integers(5, 35)), False
            
        G.add_node(i, name=name, vulnerability=round(vuln, 2), value=val, is_honeypot=is_hp)

    # Step 1: Inherit Backbone Edges from large graph template
    valid_fixed_edges = [
        (u, v, diff, noise) for (u, v, diff, noise) in EDGES_LARGE
        if u < n and v < n and u != honeypot_idx and v != honeypot_idx
    ]
    LARGE_TARGET_IDX = NUM_NODES_L - 1 # Node 8

    target_approach = [edge for edge in valid_fixed_edges if edge[1] == LARGE_TARGET_IDX]
    regular_edges   = [edge for edge in valid_fixed_edges if edge[1] != LARGE_TARGET_IDX]

    indices = list(range(len(regular_edges)))
    rng.shuffle(indices)
    n_inherit = max(int(len(regular_edges) * BACKBONE_INHERIT_RATIO), min(3, len(regular_edges)))
    inherited_regular = [regular_edges[i] for i in indices[:n_inherit]]
    
    for u, v, base_diff, base_noise in (target_approach + inherited_regular):
        actual_v = target_indices[0] if v == LARGE_TARGET_IDX else v
        if actual_v >= n:
            continue
        diff  = float(np.clip(base_diff  + rng.uniform(-0.10, 0.10), 0.05, 0.90))
        noise = float(np.clip(base_noise + rng.uniform(-0.08, 0.08), 0.03, 0.60))
        G.add_edge(u, actual_v, base_difficulty=round(diff, 2), difficulty=round(diff, 2), noise=round(noise, 2))

    # Step 2: Backbone path guarantee to targets
    for t_idx in target_indices:
        if not nx.has_path(G, 0, t_idx):
            backbone = build_backbone_path(n, t_idx, honeypot_idx, rng)
            for u, v in backbone:
                if not G.has_edge(u, v):
                    diff  = float(rng.uniform(0.15, 0.55))
                    noise = float(rng.uniform(0.08, 0.40))
                    G.add_edge(u, v, base_difficulty=round(diff, 2), difficulty=round(diff, 2), noise=round(noise, 2))

    # Step 3: Random Cross Edges
    n_cross  = int(rng.integers(n // 4, n // 2))
    attempts = 0
    while attempts < n_cross * 3:
        u = int(rng.integers(0, n - 1))
        v = int(rng.integers(u + 1, n))
        if not G.has_edge(u, v) and v != honeypot_idx and u != honeypot_idx:
            diff  = float(rng.uniform(0.20, 0.70))
            noise = float(rng.uniform(0.10, 0.50))
            G.add_edge(u, v, base_difficulty=round(diff, 2), difficulty=round(diff, 2), noise=round(noise, 2))
        attempts += 1

    # Step 4: Honeypot Trap Edges
    hp_sources = [0] + [i for i in range(1, min(4, n)) if i != honeypot_idx and G.has_edge(0, i)]
    for src in hp_sources[:2]:
        if not G.has_edge(src, honeypot_idx):
            G.add_edge(src, honeypot_idx, base_difficulty=0.05, difficulty=0.05, noise=0.03)

    # Step 5: Clean Isolated Nodes
    for i in range(1, n):
        if G.in_degree(i) == 0 and i != honeypot_idx:
            src  = int(rng.integers(0, i))
            diff  = float(rng.uniform(0.20, 0.60))
            noise = float(rng.uniform(0.10, 0.40))
            G.add_edge(src, i, base_difficulty=round(diff, 2), difficulty=round(diff, 2), noise=round(noise, 2))

    # Final hard reachability check
    for t_idx in target_indices:
        if not nx.has_path(G, 0, t_idx):
            G.add_edge(0, t_idx, base_difficulty=0.70, difficulty=0.70, noise=0.50)

    # Pack everything related to this concrete environment configuration
    return {
        "graph": G,
        "num_nodes": n,
        "target_nodes": target_indices,
        "honeypot_node": honeypot_idx
    }

def main():
    output_filename = "env/networks/network_pool.pkl"
    print("Generating persistent 20-graph network pool...")

    # Canonical base configuration generator
    pool_rng = random.Random(42)
    network_pool = []

    for i in range(20):
        pool_seed = 50000 + i
        pool_nodes = pool_rng.randint(MIN_NODES, MAX_NODES)
        
        # Generate the graph configuration details
        net_config = generate_single_network(seed=pool_seed, num_nodes=pool_nodes)
        network_pool.append(net_config)
        
        print(f"  -> Generated network graph {i+1}/20 [Nodes: {net_config['num_nodes']}, Targets: {len(net_config['target_nodes'])}]")

    # Write out serialized list of networks to disk
    with open(output_filename, "wb") as f:
        pickle.dump(network_pool, f)
        
    print(f"\nSuccessfully stored all 20 network topologies inside '{output_filename}'!")

if __name__ == "__main__":
    main()