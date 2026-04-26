---
title: Supply Chain RL Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - supply-chain
  - logistics
  - routing
  - long-horizon
  - sparse-rewards
---

#  Supply Chain RL Environment

A production-grade reinforcement learning environment built on [OpenEnv](https://github.com/meta-llm/openenv), targeting **long-horizon planning** under real-world logistics constraints. Agents must route parcels across a realistic Indian three-tier logistics network over episodes of up to **2000 steps**, with sparse delayed rewards, nested multi-parcel contract goals, and explicit cascade recovery mechanics.

## Important Links
[Google Colab](https://colab.research.google.com/drive/1uoYcg-PZTGSXboJj9t9CrQ_D6env9zYZ?usp=sharing),
[Hugging Face]()

---

## Training Results

![Training](training_curve_final(1).png)
![Training](training_curve_final(2).png)

---

## What Makes This Hard

This isn't a shortest-path problem. The agent must simultaneously:

- Route up to **60 active parcels** through a 21-hub network with dynamic congestion, stochastic delays, and edge blockages
- Track and complete **multi-parcel contracts** that only pay out when *all* K parcels arrive — potentially 300+ steps after the contract spawned
- Respond to **cascade crises** (hub failure + route block) within a 150-step recovery window or face a heavy penalty
- Do all of this under **sparse rewards** — no hand-holding signal every step, just delivery events and contract/cascade resolutions

| Property | Value |
|---|---|
| Episode length | 2000 steps (configurable 1000–5000) |
| Reward mode | Sparse by default |
| Active parcels | Up to 60 simultaneously |
| Simultaneous contracts | Up to 5 |
| Network size | 21 hubs, 80 directed edges |
| Concurrent RL workers | 4 (via WebSocket) |

---

## Network Topology

Real Indian logistics network across 3 tiers:

```
         [M0 Delhi]──────────[M1 Mumbai]──────────[M2 Bengaluru]
          /       \          /          \          /            \
    [R0 Hyd] [R1 Chennai] [R2 Kolkata] [R3 Pune] [R4 Ahmd] [R5 Jaipur]
      / \      / \          / \          / \        / \        / \
   [L0][L1] [L2][L3]    [L4][L5]     [L6][L7]  [L8][L9] [L10][L11]
```

| Tier | Hubs | Capacity |
|---|---|---|
| 0 — Mega | Delhi, Mumbai, Bengaluru | 80 parcels |
| 1 — Regional | Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur | 40 parcels |
| 2 — Local | Lucknow, Chandigarh, Nagpur, Bhubaneswar, Coimbatore, Kochi, Indore, Surat, Patna, Guwahati, Visakhapatnam, Vadodara | 20 parcels |

---

## Quick Start

### Docker (recommended)

```bash
docker build -t supply-chain-env:latest -f server/Dockerfile .
docker run -p 8000:8000 supply-chain-env:latest
```

### Local

```bash
uv sync
uvicorn server.app:app --reload --port 8000
```

### Basic RL Loop

```python
from supply_chain import SupplyChainEnv, SupplyChainAction, ParcelRouteDecision

with SupplyChainEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()

    for step in range(obs.metadata["max_steps"]):
        decisions = []
        for parcel in obs.active_parcels:
            if parcel.is_delivered or parcel.is_expired:
                continue
            # Pick a non-blocked neighbour — replace with your agent logic
            for edge in obs.edges:
                if edge.from_hub == parcel.current_hub and not edge.is_blocked:
                    decisions.append(ParcelRouteDecision(
                        parcel_id=parcel.parcel_id,
                        next_hub_id=edge.to_hub,
                    ))
                    break

        result = env.step(SupplyChainAction(routing_decisions=decisions))
        obs = result.observation

        if result.done:
            break
```

---

## Reward Structure

Nine reward components returned every step in `obs.reward_breakdown`:

| Component | Signal | When |
|---|---|---|
| `delivery_bonus` | `+` | Each parcel delivered on time |
| `lateness_penalty` | `−` | Each parcel that expires |
| `contract_bonus` | `+` large | All K parcels in a contract delivered (delayed) |
| `recovery_bonus` | `+`/`−` | Cascade recovery window expires (delayed) |
| `capacity_penalty` | `−` | Hub overloaded (every step) |
| `congestion_penalty` | `−` | Routing through congestion when a better path exists |
| `illegal_move_penalty` | `−` | Invalid routing decision |
| `hop_progress` | `+` tiny | Suppressed in sparse mode |
| `waiting_penalty` | `−` tiny | Suppressed in sparse mode |

Toggle dense rewards for curriculum training:

```python
env = SupplyChainEnvironment(sparse_rewards=False)  # dense — good for early training
env = SupplyChainEnvironment(sparse_rewards=True)   # sparse — default, harder
```

---

## Contracts

Contracts are multi-step, multi-parcel goals: deliver K parcels (3–8) from city A to city B within N steps. A large lump reward fires only on full completion — partial progress gives no contract bonus.

```python
for contract in obs.metadata["active_contracts"]:
    print(f"Contract {contract['contract_id']}: "
          f"{contract['origin_city']} → {contract['dest_city']}, "
          f"{contract['remaining_count']} parcels left, "
          f"{contract['steps_to_deadline']} steps to deadline")

# Find which parcel belongs to which contract
for parcel in obs.active_parcels:
    contract_id = obs.metadata["parcel_contract_map"].get(parcel.parcel_id)
    if contract_id:
        print(f"  {parcel.parcel_id} is serving contract {contract_id}")
```

---

## Cascade Recovery

A cascade crisis triggers when a hub fails and an adjacent route blocks simultaneously, stranding a group of parcels. The agent has **150 steps** to reroute them.

- Full recovery (all parcels rerouted): `+60` × priority
- Partial recovery: proportional bonus
- No recovery: `−40` × priority

```python
if obs.metadata["cascade_recovery_needed"]:
    for cascade in obs.metadata["active_cascades"]:
        print(f"CASCADE {cascade['event_id']}: "
              f"{cascade['failed_city']} hub down, "
              f"{cascade['total_affected']} parcels stuck, "
              f"{cascade['steps_to_deadline']} steps left")
        if cascade["urgent"]:
            print("  ⚠ URGENT — under 30 steps!")
```

---

## Observation Reference

### `obs.hubs` — list of `HubStatus`
`hub_id`, `tier`, `current_load`, `capacity`, `load_ratio`, `is_overloaded`

### `obs.edges` — list of `EdgeStatus`
`from_hub`, `to_hub`, `base_travel_time`, `current_travel_time`, `congestion_level`, `has_random_delay`, `is_blocked`

### `obs.active_parcels` — list of `ParcelStatus`
`parcel_id`, `origin_hub`, `destination_hub`, `current_hub`, `hops_taken`, `steps_in_transit`, `deadline`, `steps_remaining`, `priority`, `is_delivered`, `is_expired`

### `obs.metadata` — key fields

| Key | Type | Description |
|---|---|---|
| `max_steps` | int | Total episode length |
| `sparse_reward_mode` | bool | Whether dense signals are suppressed |
| `active_contracts` | list | Currently active contract dicts |
| `parcel_contract_map` | dict | `{parcel_id: contract_id}` |
| `total_contracts_completed` | int | Episode KPI |
| `total_contracts_failed` | int | Episode KPI |
| `active_cascades` | list | Active cascade recovery events |
| `cascade_recovery_needed` | bool | Shortcut flag |
| `delivered_last_100` | int | Rolling 100-step delivery count |
| `delivery_rate_100` | float | `delivered / (delivered + expired)` over last 100 steps |
| `in_rush_hour` | bool | True during peak demand windows |
| `demand_surge_active` | bool | True during demand surge events |
| `failed_hubs` | list | Hub IDs currently failed |
| `weather_zones` | list | Active weather events and affected hubs |

---

## Observation Vectorisation

```python
import numpy as np

MAX_PARCELS = 60
MAX_CONTRACTS = 5

def obs_to_vector(obs) -> np.ndarray:
    hub_feats = np.array([[h.load_ratio, float(h.is_overloaded)]
                          for h in obs.hubs]).flatten()

    edge_feats = np.array([[
        e.congestion_level, float(e.has_random_delay),
        float(e.is_blocked), e.current_travel_time / 10.0,
    ] for e in obs.edges]).flatten()

    parcel_map = obs.metadata.get("parcel_contract_map", {})
    parcel_feats = []
    for p in obs.active_parcels[:MAX_PARCELS]:
        parcel_feats.extend([
            max(0.0, p.steps_remaining / obs.metadata["max_steps"]),
            p.priority / 2.0,
            float(p.parcel_id in parcel_map),
            float(p.is_delivered),
            float(p.is_expired),
        ])
    while len(parcel_feats) < MAX_PARCELS * 5:
        parcel_feats.append(0.0)

    contract_feats = []
    for c in obs.metadata.get("active_contracts", [])[:MAX_CONTRACTS]:
        contract_feats.extend([
            c["completion_frac"],
            c["priority"] / 2.5,
            max(0.0, c["steps_to_deadline"] / obs.metadata["max_steps"]),
            c["remaining_count"] / 8.0,
        ])
    while len(contract_feats) < MAX_CONTRACTS * 4:
        contract_feats.append(0.0)

    cascade_feats = [0.0, 0.0]
    if obs.metadata.get("active_cascades"):
        c = obs.metadata["active_cascades"][0]
        cascade_feats = [
            max(0.0, c["steps_to_deadline"] / 150.0),
            float(c["urgent"]),
        ]

    global_feats = np.array([
        obs.current_step / obs.metadata["max_steps"],
        float(obs.metadata.get("in_rush_hour", False)),
        float(obs.metadata.get("demand_surge_active", False)),
        obs.metadata.get("delivery_rate_100", 0.5),
        float(obs.metadata.get("cascade_recovery_needed", False)),
    ])

    return np.concatenate([
        hub_feats, edge_feats, parcel_feats, contract_feats, cascade_feats, global_feats
    ]).astype(np.float32)
```

---

## Training Curriculum

```python
# Stage 1 — dense rewards, short episodes
env = SupplyChainEnvironment(max_steps=500,  sparse_rewards=False)

# Stage 2 — introduce contracts, still dense
env = SupplyChainEnvironment(max_steps=1000, sparse_rewards=False)

# Stage 3 — sparse rewards, full episode
env = SupplyChainEnvironment(max_steps=2000, sparse_rewards=True)

# Stage 4 — maximum difficulty
env = SupplyChainEnvironment(max_steps=5000, sparse_rewards=True)
```

---

## Environment Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_steps` | 2000 | Episode length |
| `sparse_rewards` | True | Suppress hop/waiting signals |
| `seed` | None | RNG seed for reproducibility |
| `MAX_ACTIVE_PARCELS` | 60 | Max simultaneous parcels |
| `BASE_PARCEL_SPAWN_RATE` | 1.2 | Poisson λ for new parcels/step |
| `MAX_ACTIVE_CONTRACTS` | 5 | Max simultaneous contracts |
| `CONTRACT_SPAWN_INTERVAL` | 80 | Steps between contract spawns |
| `CASCADE_TRIGGER_PROB` | 0.003 | Per-step cascade probability |
| `CASCADE_RECOVERY_WINDOW` | 150 | Steps to recover from cascade |
| `DELAY_PROB` | 0.08 | Per-edge stochastic delay probability |
| `BLOCK_PROB` | 0.02 | Per-edge full block probability |

---

## Project Structure

```
supply_chain/
├── __init__.py          # Package exports
├── client.py            # SupplyChainEnv — WebSocket client for RL agents
├── models.py            # Pydantic models: actions, observations, statuses
├── openenv.yaml         # OpenEnv spec
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── app.py                        # FastAPI server (HTTP + WebSocket)
    ├── supply_chain_environment.py   # Core environment logic
    ├── requirements.txt
    ├── Dockerfile
    └── data/
        ├── hubs.csv                  # 21 hub definitions with capacities
        ├── edges.csv                 # 80 directed routes with travel times
        ├── demand_patterns.csv       # 36-point demand curve (interpolated)
        └── weather_zones.csv         # 3 weather zones with delay probabilities
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset episode, returns initial observation |
| `POST` | `/step` | Submit action, returns next observation + reward |
| `GET` | `/state` | Current episode state |
| `GET` | `/schema` | Action/observation JSON schemas |
| `WS` | `/ws` | Persistent WebSocket session |
| `GET` | `/health` | Health check |
| `GET` | `/web` | Interactive web UI |

---
