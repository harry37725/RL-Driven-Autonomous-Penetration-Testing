
# 🛡️ Pentest DDQN — Autonomous Penetration Testing with Deep Reinforcement Learning

> A Dueling Double DQN agent that learns to navigate simulated corporate networks, find vulnerabilities, and reach critical targets — all while evading a live defender.

----------
<img width="1120" height="560" alt="episode_04" src="https://github.com/user-attachments/assets/8e3a9fdf-c492-4512-b201-2a362e4a2742" />


## Table of Contents

-   [Project Overview](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#project-overview)
-   [The Journey](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#the-journey)
    -   [Stage 1 — 10-Node Network (Foundation)](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#stage-1--10-node-network-foundation)
    -   [Stage 2 — Reward Hypertuning](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#stage-2--reward-hypertuning)
    -   [Stage 3 — Parameter Hypertuning](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#stage-3--parameter-hypertuning)
    -   [Stage 4 — Combined Grid Search](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#stage-4--combined-grid-search)
    -   [Stage 5 — 15-Node Network (Scale-Up)](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#stage-5--15-node-network-scale-up)
    -   [Stage 6 — Generalisation (Random Topologies)](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#stage-6--generalisation-random-topologies)
-   [Architecture](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#architecture)
-   [Results](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#results)
-   [Project Structure](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#project-structure)
-   [How to Run](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#how-to-run)
-   [Dependencies](https://claude.ai/chat/95be9904-e310-441a-adb7-7b428f2dcd83?onboarding=1#dependencies)

----------

## Project Overview

This project trains a **Dueling Double DQN (DDQN)** agent to autonomously perform penetration testing on simulated corporate networks. The agent starts at an internet-facing entry point and must compromise a chain of machines to reach a high-value target (the Internal Database) — without getting caught by an active network defender.

The problem is modelled as a **Markov Decision Process (MDP)**:


**State**  --- Which nodes are compromised, patched, current position, detection level, timestep

**Actions** --- Probe a node (gather intel) or Exploit a node (attempt compromise)

**Reward** --- +800 target reached, +200 new node, −200 caught, −4/step, −7 high alert  *(different values in different networks)*

**Done** --- Target reached / caught by defender / timeout

The agent faces a live **defender** that patches compromised nodes and raises detection levels — forcing the agent to learn stealthy, strategic paths rather than brute-force attacks.

----------

## The Journey

### Stage 1 — 10-Node Network (Foundation)

<img width="285" height="277" alt="image" src="https://github.com/user-attachments/assets/3f48737f-0de7-4d9f-bb30-c2d2334d8e43" />


The project began with a **10-node corporate network** — a carefully designed topology with two distinct paths to the target:

```
[0: Entry] ──► [1: WebServer] ──► [2: AppServer] ──► [8: InternalDB ★]
                     │                   │
                     ▼                   ▼
               [3: MailServer]     [4: DevBox] ──► [7: AdminPanel] ──► [8: InternalDB ★]
                     │                   │
                     ▼                   ▼
               [6: DBBackup]       [5: FileServer]
                     │
                     └──────────────────► [8: InternalDB ★]

```

**Two strategic paths exist:**

-   **Fast Path** — `0 → 1 → 2 → 8` — fewer hops but high noise and a hard final exploit
-   **Stealth Path** — `0 → 1 → 2 → 4 → 7 → 8` — lower noise, more hops, requires working through Admin Panel

The network environment defines 10 nodes with varying vulnerability scores (0.25–0.95), 15 directed edges with difficulty and noise parameters, and a defender that patches nodes with 15% probability per step.

**MDP dimensions for 10-node:**

-   State vector: **32 floats** (10 compromised + 10 patched + 10 position-one-hot + detection + timestep)
-   Action space: **20 actions** (probe 0–9, exploit 0–9)

The Dueling DQN architecture was established here:

```
Input (32) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(64, ReLU)
                                                          │
                                          ┌───────────────┴───────────────┐
                                    Value V(s)                   Advantage A(s,a)
                                    Dense(32) → 1             Dense(32) → 20
                                          └───────────────┬───────────────┘
                                               Q(s,a) = V(s) + A(s,a) − mean(A)

```

The **Double DQN** fix decouples action selection from evaluation — the online network picks the best action while the target network scores it. This prevents overestimation of risky moves.

----------

### Stage 2 — Reward Hypertuning

<img width="1460" height="1120" alt="summary_dashboard" src="https://github.com/user-attachments/assets/e5b97e51-ccc5-40c4-aecd-ccccafe90615" />

Before touching any network parameters, the **reward function itself was tuned** — because the agent can only learn what the reward signal teaches it.

Three reward dimensions were explored, each with 3 configurations, trained for 500 episodes each:

			Group 1 — R_TARGET vs R_NEW_NODE Ratio

| Configuration       | R_TARGET | R_NEW_NODE | Success% | Caught% |
|---------------------|----------|------------|----------|---------|
| Balanced (10:1)     | 500      | 50         | 26%      | 28%     |
| Target-heavy (27:1) | 800      | 30         | 16%      | 0%      |
| Node-heavy (4:1)    | 300      | 80         | 10%      | 0%      |

<img width="1607" height="982" alt="group1_target_ratio" src="https://github.com/user-attachments/assets/4b695108-4527-48da-9643-812148a53184" />

			Group 2 — Caught Penalty

| Configuration | R_CAUGHT | Success% | Caught% | Insight                                |
|---------------|----------|----------|---------|----------------------------------------|
| Current       | −200     | 32%      | 30%     | Acceptable risk                        |
| Fearful       | −350     | 2%       | 2%      | Agent paralysed — never attacks        |
| Reckless      | −100     | 28%      | 8%      | Lower caution, slightly lower success  |


<img width="1607" height="982" alt="group2_caught_penalty" src="https://github.com/user-attachments/assets/770f9aad-4097-4080-a8f2-c625aaeea8d2" />


			Group 3 — Distance Reward Shaping

Tested whether adding a potential-based distance-to-target reward helped guide the agent.
-->
| Configuration         | Reward Scope | Success% | 
|------------------------|--------------|----------|
| Current (≤2 hops)      | Limited      | 12%      |      
| Extended (≤3 hops)     | Broader      | 22%      |       
| No distance reward     | None         | **36%**  | 

<img width="1607" height="982" alt="group3_distance_shape" src="https://github.com/user-attachments/assets/166318c6-a2a1-4614-bd1b-2ea4a22b02f1" />
					Key finding:
A −350 caught penalty completely paralysed the agent — it stopped attacking entirely to avoid the risk. A balanced +500/+50 ratio with −200 caught penalty gave the best trade-off between exploration and success. The final reward values selected were:

```python
R_TARGET          = +500
R_NEW_NODE        = +50
R_FAILED_EXPLOIT  = -4
R_HIGH_ALERT      = -7
R_CAUGHT          = -200
R_STEP            = -4

```
----------
R_Target=800 and R_NEW_NODE=200 worked better latter

----------
### Stage 3 — Parameter Hypertuning

<img width="1631" height="1119" alt="summary_dashboard" src="https://github.com/user-attachments/assets/52b3c946-7d2a-460e-84ff-3988c8b1d796" />


With the reward function locked in, all five major agent hyperparameters were tuned **independently** (one group at a time, holding others fixed at baseline). Each configuration ran for **1000 episodes**.

			5 groups tuned in priority order:

**Group 1 — Learning Rate** (biggest impact on convergence)

Tested `[1e-4, 5e-4, 1e-3, 3e-3]`. Too high → unstable Q-values. Too low → too slow to learn within 1000 episodes. **Winner: 5e-4**

**Group 2 — Gamma (discount factor)**

Tested `[0.90, 0.95, 0.99, 0.999]`. The agent must plan multi-hop paths (up to 6 hops to the target), so a high gamma matters. **Winner: 0.99**

**Group 3 — Epsilon Decay**

Tested `[0.995, 0.999, 0.9995, 0.9999]`. Too fast → agent commits to bad local optima early. Too slow → never exploits what it learned. **Winner: 0.999**

**Group 4 — Batch Size**

Tested `[32, 64, 128, 256]`. Larger batches → more stable gradients for this environment. **Winner: 256**

**Group 5 — Target Update Frequency**

| Configuration       | Update Freq | Success% | Caught% |
|---------------------|-------------|----------|---------|
| Infrequent (500)    | 500         | **55%**  | 16%     |
| Very frequent (50)  | 50          | 22%      | 10%     |
| Frequent (100)      | 100         | 20%      | 6%      |
| Default (200)       | 200         | 19%      | 15%     |


**Winner: 500** — infrequent target updates kept Q-value targets stable long enough for the online network to learn meaningful patterns, preventing the oscillation that plagued more frequent updates.

**Confirmed best configuration (individual tuning):**

```python
CFG = {
    "lr":                 5e-4,
    "gamma":              0.99,
    "epsilon_decay":      0.999,
    "batch_size":         256,
    "target_update_freq": 500,
}

```

----------

### Stage 4 — Combined Grid Search

Individual tuning can miss interaction effects between parameters. A full **grid search across 243 combinations** ( 5 parameters) was run, **500 episodes each**, scored by:

```
True Score = Success% − Caught%

```
<img width="1805" height="1124" alt="top15_configs" src="https://github.com/user-attachments/assets/7de31801-b7c2-42c2-9f2d-e834750048b7" />


**Top 5 configurations from grid search:**

| Rank | LR    | Gamma | ε-decay | Batch | TU-Freq | Success% | Caught% | True Score |
|------|-------|-------|---------|-------|---------|----------|---------|------------|
| 1    | 5e-4  | 0.99  | 0.999   | 256   | 350     | 57%      | 12%     | **45**     |
| 2    | 5e-4  | 0.999 | 0.999   | 64    | 350     | 61%      | 26%     | 35         |
| 3    | 5e-4  | 0.999 | 0.999   | 256   | 500     | 54%      | 23%     | 31         |
| 4    | 5e-4  | 0.995 | 0.999   | 256   | 200     | 48%      | 18%     | 30         |
| 5    | 5e-4  | 0.995 | 0.9995  | 256   | 350     | 50%      | 23%     | 27         |


The top config (`lr=5e-4, gamma=0.99, ε-decay=0.999, batch=256, TU=350`) achieved **57% success with only 12% caught** — the best risk-adjusted performance across all 81 combinations.

**Final training config adopted:**

```python
CFG = {
    "lr":                 5e-4,
    "gamma":              0.99,
    "epsilon_decay":      0.999,
    "batch_size":         256,
    "target_update_freq": 350,
}

```

----------

### Stage 5 — 15-Node Network (Scale-Up)

<img width="827" height="553" alt="image" src="https://github.com/user-attachments/assets/f6c8be1c-d5fb-4af0-8638-54051b4d1d00" />


With a trained, well-tuned 10-node agent in hand, the environment was **scaled up to 15 nodes**. Rather than training from scratch, the shared layer weights were transferred — a deliberate design choice that respected what the agent had already learned.

**5 new nodes added:**

| Node | Type       | Vulnerability | Special Property                                   |
|------|------------|---------------|----------------------------------------------------|
| 10   | DMZ Server | 0.70          | New entry-point hop                                |
| 11   | VPN Gateway| 0.45          | Requires DMZ credential first                      |
| 12   | Cloud Instance | 0.85      | High vulnerability, alternate path                 |
| 13   | Trap (Honeypot) | 0.90   | Looks easy — triggers instant high alert (detection → 0.85) |
| 14   | Secondary DB | 0.40        | Extra path to InternalDB via Cloud                 |


**New mechanics in the 15-node environment:**

-   **Honeypot trap** — node 13 looks highly vulnerable (0.90) but exploiting it immediately pushes detection to 0.85, nearly triggering a catch
-   **VPN credential requirement** — node 11 cannot be exploited without first compromising node 10 (DMZ)
-   **Progressive difficulty** — the environment gets harder as the agent accumulates successes:
    -   Level 1 (0–50 successes): Base reactive defender
    -   Level 2 (50–150 successes): Faster patching, adaptive defender

**New MDP dimensions:**

-   State vector: **47 floats**
-   Action space: **30 actions** (probe 0–14, exploit 0–14)

**Weight transfer strategy:**

```
Transferred (shape unchanged):
  ✅  shared.2  — hidden Dense 128→128
  ✅  shared.4  — hidden Dense 128→64
  ✅  value_stream.0, value_stream.2  — V(s) head
  ✅  advantage_stream.0  — intermediate advantage layer

Partially transferred (with smart padding):
  ⚡  shared.0.weight  — input layer (32→128 adapted to 47→128)
                         old 32 columns copied, 15 new columns random-init

Skipped (output layer changed):
  ⚠️   advantage_stream.2  — action count changed (20 → 30)

```

The 15-node agent fine-tuned from this transferred base, reaching a **95% success rate** — a result that would have taken far longer training from scratch.

<img width="329" height="300" alt="image" src="https://github.com/user-attachments/assets/2c2c8b89-14b5-4e8d-aff7-fdbbae7a02c9" />


**15-node test results (5 episodes, greedy policy):**

| Metric          	  | Result                   			|
|-----------------|----------------------------	|
| ✅ Success rate | **95%**                  	|
| 🚨 Caught rate  | 0–5%                     		|
| ⏱ Timeout       | 0-5%                    			|
| GIF replays     | `results_large/replays/`|


The agent learned to:

-   **Avoid the honeypot** (node 13) despite its high apparent vulnerability
-   **Respect the VPN credential requirement** — always hitting DMZ before VPN Gateway
-   **Prefer the stealth path** (via Admin Panel) over the direct noisy route
-   **Back off when detection climbs** — using probe-heavy turns to let the detection level decay

----------

### Stage 6 — Generalisation (Random Topologies)

The final challenge: can the agent navigate **networks it has never seen before?**

A `RandomNetworkEnv` was built that generates new corporate network topologies at runtime with randomised:

-   Node vulnerability scores
-   Edge difficulties and noise levels
-   Network topology (within realistic corporate constraints)
-   Number of nodes (10–18)

**Guarantees** built into the generator: always one valid path to target, no isolated nodes, a honeypot always present at a random position, and a realistic DMZ/internal/target zone structure.

**New MDP dimensions (fixed regardless of actual network size):**

-   State vector: **57 floats** (padded/truncated to MAX_NODES=18)
-   Action space: **36 actions** (probe/exploit 0–17)

The 15-node model's weights were transferred again — shared layers carried over, input layer adapted from 47→57, output layer from 30→36.

**Generalisation evaluation: 100 episodes across 20 unseen networks (5 repeats each)**

<img width="421" height="385" alt="image" src="https://github.com/user-attachments/assets/7bb1ba3f-309d-4972-8984-4940b212cca5" />


The agent successfully transfers to networks with similar topology structure (small, 10–13 node networks resembling its training data) but times out on larger or structurally different topologies — a classic **distribution shift** problem.

**Why it struggles with full generalisation:**

-   The state representation pads to 18 nodes, but the agent's Q-values were learned on specific 15-node structural patterns
-   Larger networks (16–18 nodes) have fundamentally different path lengths and topology
-   The agent never learned to "explore unknown territory" — it exploits patterns it memorised

**This is an honest, documented limitation** — and a clear direction for future work (curriculum learning across diverse topologies, graph neural network encodings, or meta-RL).

----------

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     DuelingDQN Network               |
│                                                      | 
│  Input (state_size)                                  │
│       ↓                                              |       
│  Dense(128, ReLU)  ─┐                                |
│  Dense(128, ReLU)   │  Shared Feature Extractor      |      
│  Dense(64,  ReLU)  ─┘                                |
│       ↓                                              |
│  ┌────┴─────┐                                        |
│  │          │                                        |
│  V(s)       A(s,a)                                   |
│  Dense(32)  Dense(32)                                |
│  Linear(1)  Linear(action_size)                      |
│  │          │                                        |
│  └────┬─────┘                                        |
│       ↓                                              |
│  Q(s,a) = V(s) + A(s,a) − mean(A(s,a))               |
└──────────────────────────────────────────────────────┘

```

**DDQN fix:** Online net selects action `a* = argmax Q_online(s', a)`, target net evaluates it `Q_target(s', a*)`. Decoupling prevents overestimation of risky actions.

**Replay Buffer:** 50,000 transitions, uniform random sampling, NumPy-backed for speed.

**Gradient clipping:** `max_norm=10.0` on all parameters to prevent exploding gradients during early training.

----------

## Results

### 10-Node Network

<img width="503" height="374" alt="test_results" src="https://github.com/user-attachments/assets/acefcceb-7139-436b-b7cd-3d640d2d5a21" />

| Phase                | Episodes   | Best Success%             |
|-----------------------|------------|---------------------------|
| Initial training      | 5000       | ~57%                      |
| After reward tuning   | 500/config | 32% best single config    |
| After param tuning    | 1000/config| 55% (TU=500)              |
| After grid search     | 500/config | **57% with 12% caught**   |


<img width="497" height="542" alt="Screenshot 2026-05-28 121841" src="https://github.com/user-attachments/assets/5bade261-091d-407b-aff8-c9156765fe50" />


### 15-Node Network

<img width="1120" height="560" alt="episode_04" src="https://github.com/user-attachments/assets/3132a910-5e16-49b8-8b25-599b9d1263eb" />


| Metric               | Value                  |
|-----------------------|------------------------|
| ✅ **Success rate**   | **95%**                |
| 🚨 Caught rate        | <5%                    |
| ⏱ Timeout             | <5%                     |
| Training episodes     | 3000                   |
| Transfer from 10-node | Yes (shared layers)    |


## Project Structure

```
RL_Project/
├── agent/
│   ├── ddqn_agent.py          # Dueling DDQN + partial weight transfer
│   └── replay_buffer.py       # Experience replay buffer
│
├── env/
│   ├── network_env.py          # 10-node fixed corporate network
│   ├── network_env_large.py    # 15-node network + progressive difficulty
│   ├── network_env_20.py       # Pool of 20 stored random networks
│   ├── network_env_random.py   # Random network generator
│   └── network_generator.py   # Network topology builder utilities
│       networks/
│       └── network_pool.pkl   # 20 pre-generated unseen topologies
│
├── hypertuning/
│   ├── reward_hypertune.py    # Reward function tuning (3 groups)
│   ├── hyperparam_tune.py     # Parameter tuning (5 groups independently)
│   └── grid_search.py         # Combined grid search (81 combinations)
│
├── src/
│   ├── train.py               # Main training loop
│   ├── test.py                # Test 15-node agent, save GIFs
│   ├── test_generalisation.py # Evaluate across 20 unseen networks
│   └── transfer_weights.py    # 10-node → 15-node weight transfer
│
├── visualisation/
│   ├── visualiser.py          # Live training dashboard
│   ├── graph_renderer.py      # Network graph + GIF renderer
│   └── training_dashboard.py  # Training metrics plots
│
├── results/
│   ├── ddqn_final.pt          # Trained 10-node model
│   ├── checkpoints/           # Snapshots every 500 episodes
│   ├── plots/                 # Training curves
│   ├── replays/               # Episode GIFs (10-node)
│   ├── reward_tuning/         # Reward hypertuning results
│   ├── hyperparam_tuning/     # Parameter tuning results
│   └── grid_search/           # Grid search heatmaps + rankings
│
├── results_large/
│   ├── ddqn_transfer.pt       # Transfer-initialized 15-node model
│   ├── checkpoints/           # 15-node training snapshots
│   ├── plots/                 # 15-node training curves + test results
│   └── replays/               # Episode GIFs (15-node, best replays)
│
└── results_random/
    ├── checkpoints/           # Generalisation training snapshots
    └── eval/                  # Evaluation reports (JSON + TXT)

```

----------

## How to Run

**Train the 10-node agent (from scratch):**

```bash
python src/train.py

```

**Test the 15-node agent (watch 5 episodes):**

```bash
python src/test.py --model results_large/checkpoints/ddqn_ep00500.pt --episodes 5

```

**Run reward hypertuning:**

```bash
python hypertuning/reward_hypertune.py

```

**Run parameter hypertuning (specific groups):**

```bash
python hypertuning/hyperparam_tune.py --groups 1 2 3

```

**Run full grid search:**

```bash
python hypertuning/grid_search.py

```

**Transfer weights from 10-node to 15-node:**

```bash
python src/transfer_weights.py --source results/ddqn_final.pt --output results_large/ddqn_transfer.pt

```

**Evaluate generalisation on 20 unseen networks:**

```bash
python src/test_generalisation.py --model results_random/checkpoints/ddqn_ep02500.pt

```

----------

## Dependencies

```
torch
numpy
networkx
matplotlib
tqdm
Pillow       # GIF rendering

```

Install with:

```bash
pip install torch numpy networkx matplotlib tqdm Pillow

```

----------

## Key Takeaways

1.  **Reward shaping is as important as architecture** — a −350 caught penalty completely paralysed the agent; reward tuning came first for a reason.
    
2.  **Target update frequency is underappreciated** — updating the target network every 500 steps (vs 200) was the single biggest individual parameter win.
    
3.  **Transfer learning across different-sized networks works** — shared hidden layers transferred cleanly from 10→15→18 nodes, with only the input/output layers needing reinitialisation.
    
4.  **95% success on 15-node requires the honeypot lesson** — the agent learned to avoid node 13 even though it had the highest apparent vulnerability (0.90), purely through reward signal.
    
5.  **Generalisation to truly random topologies is hard** — 15% overall success on 20 unseen networks shows the gap between "learned a good policy" and "learned to reason about arbitrary graphs." Graph Neural Networks or meta-learning are the natural next steps.
    

----------

_Built with PyTorch · NetworkX · Matplotlib_
