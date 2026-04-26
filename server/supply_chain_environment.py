# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain RL Environment — Core Implementation (v1.0 — Long-Horizon Edition).

Key upgrades over v0.4 for Theme #2 (Super Long-Horizon Planning):

1. LONG HORIZON
   - Episodes now run 1000–5000 steps (configurable via MAX_STEPS).
   - Demand curve stretches across the full episode via interpolation.
   - Service windows, rush hours, and weather events all scaled to match.

2. SPARSE / DELAYED REWARDS
   - Per-step dense reward replaced by sparse signals:
       * delivery_bonus only fires on delivery (unchanged, but now rarer relative to horizon)
       * contract_bonus fires only when a full CONTRACT is completed (new — delayed)
       * recovery_bonus fires only when a cascade crisis is fully resolved (new — delayed)
   - Intermediate hop_progress and waiting_penalty are scaled down 10× so they don't
     swamp the sparse signals.
   - SPARSE_REWARD_MODE (default True) suppresses hop/waiting signals entirely,
     leaving only delivery, contract, and recovery rewards.

3. NESTED CONTRACT GOALS
   - Contracts are multi-parcel objectives: "deliver K parcels from city A to city B
     within N steps". The agent must decompose the contract into per-parcel routing
     decisions and track contract state across hundreds of steps.
   - Contracts have priorities and deadlines, rewarding early completion with a large
     bonus. Failed contracts give a heavy penalty.
   - Up to MAX_ACTIVE_CONTRACTS contracts active simultaneously.

4. EXPLICIT RECOVERY MECHANICS
   - CascadeEvent: a hub failure + route block combo that creates a backlog of stuck
     parcels. The agent must actively reroute them within RECOVERY_WINDOW steps.
   - Recovery is scored: full reroute = large bonus, partial = proportional, none = penalty.
   - CascadeEvent metadata exposed in observation so the agent knows recovery is needed.

5. EXTENDED STATE TRACKING
   - obs.metadata now includes contract states, cascade events, and long-horizon KPIs
     (delivery_rate_100, expiry_rate_100) computed over rolling windows of 100 steps.
   - Parcel IDs linked to contract IDs so agents can reason about which parcel serves
     which contract.

All v0.4 network topology, CSV loading, weather, demand surge, and hub failure logic
is preserved unchanged.
"""

import csv
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        EdgeStatus,
        HubStatus,
        ParcelRouteDecision,
        ParcelStatus,
        SupplyChainAction,
        SupplyChainObservation,
    )
except ImportError:
    from models import (  # type: ignore
        EdgeStatus,
        HubStatus,
        ParcelRouteDecision,
        ParcelStatus,
        SupplyChainAction,
        SupplyChainObservation,
    )


# ---------------------------------------------------------------------------
# Constants — Long-Horizon Edition
# ---------------------------------------------------------------------------

# --- Episode length ---
MAX_STEPS = 2000                    # Default: 2000 steps. Set to 1000–5000 as needed.

# --- Parcel management ---
MAX_ACTIVE_PARCELS = 60
BASE_PARCEL_SPAWN_RATE = 1.2
PARCEL_SPAWN_RATE = BASE_PARCEL_SPAWN_RATE
PARCEL_PRUNE_INTERVAL = 50

# --- Deadline tuning ---
DEADLINE_SLACK = 3.5

# --- Stochastic events ---
DELAY_PROB = 0.08
DELAY_MAGNITUDE = 2
BLOCK_PROB = 0.02
CONGESTION_THRESH = 5
CONGESTION_EXTRA_TIME = 1

# --- Rush hours: 5 windows spread across the episode ---
RUSH_HOUR_WINDOWS = [
    (100, 200),
    (400, 500),
    (700, 800),
    (1100, 1200),
    (1600, 1700),
]
RUSH_HOUR_BACKGROUND_PARCELS = 3

# --- Demand surge ---
DEMAND_SURGE_PROB = 0.008
DEMAND_SURGE_MIN_DURATION = 30
DEMAND_SURGE_MAX_DURATION = 60
DEMAND_SURGE_SPAWN_RATE = 4.0

# --- Hub failures ---
HUB_FAILURE_PROB = 0.005
HUB_FAILURE_MIN_DURATION = 20
HUB_FAILURE_MAX_DURATION = 50
HUB_FAILURE_CAPACITY_RATIO = 0.5

# --- Service windows: 5 off-peak windows across the episode ---
SERVICE_WINDOWS_OFF_PEAK = [
    (0, 50),
    (250, 300),
    (550, 600),
    (950, 1000),
    (1450, 1500),
]
OFF_PEAK_MULTIPLIER: Dict[str, float] = {
    "mega_to_mega": 1.2,
    "mega_to_regional": 1.5,
    "regional_to_regional": 1.5,
    "regional_to_local": 2.0,
    "local_to_local": 2.5,
}

# ---------------------------------------------------------------------------
# Sparse Reward Mode
# ---------------------------------------------------------------------------

SPARSE_REWARD_MODE = True

# Reward coefficients
R_DELIVERY_BASE = 20.0
R_DELIVERY_EARLY_BONUS = 5.0
R_LATENESS_PER_STEP = -1.5
R_EXPIRY_PENALTY = -15.0

# Hop/waiting: tiny in sparse mode, normal otherwise
R_HOP_PROGRESS = 0.05 if SPARSE_REWARD_MODE else 0.5
R_WAITING_PER_STEP = -0.03 if SPARSE_REWARD_MODE else -0.3

R_CAPACITY_PENALTY = -2.0
R_CONGESTION_PENALTY = -0.5
R_ILLEGAL_MOVE = -3.0

# --- Contract rewards ---
R_CONTRACT_COMPLETION_BASE = 80.0
R_CONTRACT_EARLY_BONUS = 20.0
R_CONTRACT_FAILURE_PENALTY = -50.0
R_CONTRACT_PARTIAL_CREDIT = 5.0

# --- Cascade recovery rewards ---
R_RECOVERY_FULL_BONUS = 60.0
R_RECOVERY_PARTIAL_RATIO = 0.6
R_RECOVERY_FAILURE_PENALTY = -40.0

_REWARD_KEYS = (
    "delivery_bonus",
    "lateness_penalty",
    "hop_progress",
    "waiting_penalty",
    "capacity_penalty",
    "congestion_penalty",
    "illegal_move_penalty",
    "contract_bonus",
    "recovery_bonus",
)

# ---------------------------------------------------------------------------
# Contract system
# ---------------------------------------------------------------------------

MAX_ACTIVE_CONTRACTS = 5
CONTRACT_SPAWN_INTERVAL = 80
CONTRACT_MIN_PARCELS = 3
CONTRACT_MAX_PARCELS = 8
CONTRACT_DEADLINE_SLACK = 5.0
CONTRACT_PRIORITY_CHOICES = [1.0, 1.5, 2.0, 2.5]

# ---------------------------------------------------------------------------
# Cascade recovery system
# ---------------------------------------------------------------------------

CASCADE_TRIGGER_PROB = 0.003
CASCADE_RECOVERY_WINDOW = 150
CASCADE_MIN_AFFECTED_PARCELS = 3
CASCADE_MAX_AFFECTED_PARCELS = 10

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")


# ---------------------------------------------------------------------------
# CSV Data Loader
# ---------------------------------------------------------------------------

class _DataLoader:
    """Loads and validates the four bundled CSV files at environment startup."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.hubs = self._load_hubs()
        self.edges = self._load_edges()
        self.demand_curve = self._load_demand_patterns()
        self.weather_zones = self._load_weather_zones()

    def _path(self, filename: str) -> str:
        p = os.path.join(self.data_dir, filename)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Supply chain data file not found: {p}\n"
                f"Expected data/ directory alongside supply_chain_environment.py"
            )
        return p

    def _load_hubs(self) -> List[Dict]:
        hubs = []
        with open(self._path("hubs.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hubs.append({
                    "hub_id":   row["hub_id"].strip(),
                    "city":     row["city"].strip(),
                    "state":    row["state"].strip(),
                    "tier":     int(row["tier"]),
                    "lat":      float(row["lat"]),
                    "lon":      float(row["lon"]),
                    "capacity": int(row["capacity"]),
                    "notes":    row["notes"].strip(),
                })
        assert len(hubs) == 21, f"Expected 21 hubs, got {len(hubs)}"
        return hubs

    def _load_edges(self) -> List[Dict]:
        edges = []
        with open(self._path("edges.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                edges.append({
                    "from_hub":         row["from_hub"].strip(),
                    "to_hub":           row["to_hub"].strip(),
                    "from_city":        row["from_city"].strip(),
                    "to_city":          row["to_city"].strip(),
                    "tier_type":        row["tier_type"].strip(),
                    "distance_km":      float(row["distance_km"]),
                    "base_travel_time": int(row["base_travel_time_steps"]),
                    "highway_notes":    row["highway_notes"].strip(),
                })
        return edges

    def _load_demand_patterns(self) -> List[float]:
        """
        Returns MAX_STEPS spawn-rate multipliers by interpolating the
        demand_patterns.csv curve across the full episode.
        """
        raw = []
        with open(self._path("demand_patterns.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw.append(float(row["spawn_rate_multiplier"]))

        n = len(raw)
        curve = []
        for step in range(MAX_STEPS):
            idx = (step / MAX_STEPS) * n
            lo = int(idx) % n
            hi = (lo + 1) % n
            frac = idx - int(idx)
            curve.append(raw[lo] * (1 - frac) + raw[hi] * frac)
        return curve

    def _load_weather_zones(self) -> Dict[int, Dict]:
        zones: Dict[int, Dict] = {}
        seen: Set[int] = set()
        with open(self._path("weather_zones.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                zid = int(row["zone_id"])
                if zid in seen:
                    continue
                seen.add(zid)
                hub_ids = [h.strip() for h in row["hub_ids"].split("-")]
                zones[zid] = {
                    "zone_id":          zid,
                    "zone_name":        row["zone_name"].strip(),
                    "imd_division":     row["imd_division"].strip(),
                    "hub_ids":          hub_ids,
                    "delay_prob_boost": float(row["delay_prob_boost"]),
                    "congestion_boost": int(row["congestion_boost"]),
                    "event_prob":       float(row["event_prob_per_step"]),
                    "min_duration":     int(row["min_duration_steps"]),
                    "max_duration":     int(row["max_duration_steps"]),
                    "notes":            row["notes"].strip(),
                }
        return zones


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------

class _Hub:
    def __init__(self, hub_id: str, tier: int, city: str, state: str,
                 lat: float, lon: float, capacity: int, notes: str):
        self.hub_id = hub_id
        self.tier = tier
        self.city = city
        self.state = state
        self.lat = lat
        self.lon = lon
        self._base_capacity = capacity
        self.capacity = capacity
        self.notes = notes
        self.parcel_ids: Set[str] = set()
        self.is_failed: bool = False
        self.failure_steps_remaining: int = 0

    @property
    def load(self) -> int:
        return len(self.parcel_ids)

    @property
    def is_overloaded(self) -> bool:
        return self.load > self.capacity

    def apply_failure(self):
        self.is_failed = True
        self.capacity = max(1, int(self._base_capacity * HUB_FAILURE_CAPACITY_RATIO))

    def restore(self):
        self.is_failed = False
        self.capacity = self._base_capacity


class _Edge:
    def __init__(self, from_hub: str, to_hub: str, from_city: str, to_city: str,
                 tier_type: str, distance_km: float, base_travel_time: int,
                 highway_notes: str):
        self.from_hub = from_hub
        self.to_hub = to_hub
        self.from_city = from_city
        self.to_city = to_city
        self.tier_type = tier_type
        self.distance_km = distance_km
        self.base_travel_time = base_travel_time
        self.highway_notes = highway_notes
        self.parcels_in_transit: int = 0
        self.background_load: int = 0
        self.has_delay: bool = False
        self.is_blocked: bool = False
        self.in_service_window: bool = True
        self.weather_zone: Optional[int] = None
        self.weather_delay_boost: float = 0.0
        self.weather_congestion_boost: int = 0

    @property
    def effective_load(self) -> int:
        return self.parcels_in_transit + self.background_load + self.weather_congestion_boost

    @property
    def congestion_level(self) -> float:
        return min(1.0, self.effective_load / CONGESTION_THRESH)

    @property
    def current_travel_time(self) -> int:
        t = self.base_travel_time
        if not self.in_service_window:
            t = int(math.ceil(t * OFF_PEAK_MULTIPLIER.get(self.tier_type, 1.5)))
        if self.effective_load >= CONGESTION_THRESH:
            t += CONGESTION_EXTRA_TIME
        if self.has_delay:
            t += DELAY_MAGNITUDE
        return t


class _Parcel:
    def __init__(self, parcel_id: str, origin: str, destination: str,
                 spawn_step: int, deadline: int, priority: float,
                 contract_id: Optional[str] = None):
        self.parcel_id = parcel_id
        self.origin = origin
        self.destination = destination
        self.current_hub = origin
        self.spawn_step = spawn_step
        self.deadline = deadline
        self.priority = priority
        self.contract_id = contract_id
        self.hops_taken = 0
        self.steps_in_transit = 0
        self.is_delivered = False
        self.is_expired = False
        self.steps_until_arrival: Optional[int] = None
        self.next_hub: Optional[str] = None


class _WeatherEvent:
    def __init__(self, zone: int, duration: int, zone_data: Dict):
        self.zone = zone
        self.steps_remaining = duration
        self.zone_name = zone_data["zone_name"]
        self.delay_boost = zone_data["delay_prob_boost"]
        self.congestion_boost = zone_data["congestion_boost"]


# ---------------------------------------------------------------------------
# Contract system
# ---------------------------------------------------------------------------

@dataclass
class _Contract:
    """
    A multi-parcel delivery contract: route `target_count` parcels from
    `origin_hub` to `dest_hub` before `deadline`. Rewarded only on full
    completion; failed contracts incur a heavy penalty.
    """
    contract_id: str
    origin_hub: str
    dest_hub: str
    origin_city: str
    dest_city: str
    target_count: int
    deadline: int
    priority: float
    spawn_step: int

    delivered_parcel_ids: Set[str] = field(default_factory=set)
    assigned_parcel_ids: Set[str] = field(default_factory=set)
    is_completed: bool = False
    is_failed: bool = False

    @property
    def delivered_count(self) -> int:
        return len(self.delivered_parcel_ids)

    @property
    def remaining_count(self) -> int:
        return max(0, self.target_count - self.delivered_count)

    @property
    def completion_fraction(self) -> float:
        return self.delivered_count / self.target_count if self.target_count > 0 else 0.0


# ---------------------------------------------------------------------------
# Cascade recovery system
# ---------------------------------------------------------------------------

@dataclass
class _CascadeEvent:
    """
    A cascade crisis: a hub fails and a key route blocks simultaneously,
    stranding a group of parcels. The agent must reroute them within
    CASCADE_RECOVERY_WINDOW steps.
    """
    event_id: str
    failed_hub_id: str
    blocked_edge: Tuple[str, str]
    affected_parcel_ids: Set[str]
    start_step: int
    recovery_deadline: int
    resolved: bool = False
    # FIX: track rerouted parcels in a set to avoid double-counting across steps
    rerouted_parcel_ids: Set[str] = field(default_factory=set)

    @property
    def rerouted_count(self) -> int:
        return len(self.rerouted_parcel_ids)

    @property
    def total_affected(self) -> int:
        return len(self.affected_parcel_ids)

    @property
    def recovery_fraction(self) -> float:
        return self.rerouted_count / self.total_affected if self.total_affected > 0 else 0.0


# ---------------------------------------------------------------------------
# Rolling window tracker (for long-horizon KPIs)
# ---------------------------------------------------------------------------

class _RollingWindow:
    """Tracks delivered and expired counts in a rolling window of `size` steps."""
    def __init__(self, size: int = 100):
        self.size = size
        self._delivered: deque = deque()
        self._expired: deque = deque()

    def record(self, step: int, delivered: int, expired: int):
        self._delivered.append((step, delivered))
        self._expired.append((step, expired))
        cutoff = step - self.size
        while self._delivered and self._delivered[0][0] < cutoff:
            self._delivered.popleft()
        while self._expired and self._expired[0][0] < cutoff:
            self._expired.popleft()

    @property
    def delivered_last_window(self) -> int:
        return sum(v for _, v in self._delivered)

    @property
    def expired_last_window(self) -> int:
        return sum(v for _, v in self._expired)


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class SupplyChainEnvironment(Environment):
    """
    Long-Horizon Indian Supply Chain RL Environment (v1.0).

    Parameters
    ----------
    seed : int, optional
        RNG seed for reproducible episodes.
    data_dir : str, optional
        Path to the data/ directory. Defaults to data/ alongside this file.
    max_steps : int, optional
        Override for MAX_STEPS constant. Default 2000.
    sparse_rewards : bool, optional
        Override for SPARSE_REWARD_MODE. Default True.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        seed: Optional[int] = None,
        data_dir: str = DATA_DIR,
        max_steps: int = MAX_STEPS,
        sparse_rewards: bool = SPARSE_REWARD_MODE,
    ):
        self._seed = seed
        self._max_steps = max_steps
        self._sparse_rewards = sparse_rewards
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = _PoissonRandom(seed)

        self._data = _DataLoader(data_dir)

        self._hubs: Dict[str, _Hub] = {}
        self._edges: Dict[Tuple[str, str], _Edge] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._sp_hops: Dict[Tuple[str, str], int] = {}
        self._sp_time: Dict[Tuple[str, str], int] = {}

        self._parcels: Dict[str, _Parcel] = {}
        self._current_step: int = 0
        self._total_delivered: int = 0
        self._total_expired: int = 0
        self._total_spawned: int = 0

        self._active_weather_events: List[_WeatherEvent] = []
        self._demand_surge_steps_remaining: int = 0
        self._failed_hub_ids: Set[str] = set()

        # Contract system
        self._contracts: Dict[str, _Contract] = {}
        self._total_contracts_completed: int = 0
        self._total_contracts_failed: int = 0
        self._next_contract_spawn_step: int = CONTRACT_SPAWN_INTERVAL

        # FIX: cascade_events must be a list, not a dict
        self._cascade_events: List[_CascadeEvent] = []

        # Rolling window KPIs
        self._rolling = _RollingWindow(size=100)

        self._build_network()
        self._precompute_shortest_paths()

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(self):
        self._hubs.clear()
        self._edges.clear()
        self._adjacency.clear()

        for h in self._data.hubs:
            self._hubs[h["hub_id"]] = _Hub(
                hub_id=h["hub_id"], tier=h["tier"], city=h["city"],
                state=h["state"], lat=h["lat"], lon=h["lon"],
                capacity=h["capacity"], notes=h["notes"],
            )

        for e in self._data.edges:
            key = (e["from_hub"], e["to_hub"])
            self._edges[key] = _Edge(
                from_hub=e["from_hub"], to_hub=e["to_hub"],
                from_city=e["from_city"], to_city=e["to_city"],
                tier_type=e["tier_type"], distance_km=e["distance_km"],
                base_travel_time=e["base_travel_time"],
                highway_notes=e["highway_notes"],
            )
            if e["to_hub"] not in self._adjacency[e["from_hub"]]:
                self._adjacency[e["from_hub"]].append(e["to_hub"])

        self._assign_weather_zones()

    def _assign_weather_zones(self):
        zone_membership: Dict[str, int] = {}
        for zid, zdata in self._data.weather_zones.items():
            for hub_id in zdata["hub_ids"]:
                zone_membership[hub_id] = zid
        for (a, b), edge in self._edges.items():
            za = zone_membership.get(a)
            zb = zone_membership.get(b)
            if za is not None and za == zb:
                edge.weather_zone = za

    # ------------------------------------------------------------------
    # Shortest paths
    # ------------------------------------------------------------------

    def _precompute_shortest_paths(self):
        self._sp_hops.clear()
        self._sp_time.clear()
        for src in self._hubs:
            dist_hops = {src: 0}
            q: deque = deque([src])
            while q:
                node = q.popleft()
                for nbr in self._adjacency[node]:
                    if nbr not in dist_hops:
                        dist_hops[nbr] = dist_hops[node] + 1
                        q.append(nbr)
            for dst, d in dist_hops.items():
                self._sp_hops[(src, dst)] = d

            dist_time: Dict[str, int] = {src: 0}
            frontier: List[Tuple[int, str]] = [(0, src)]
            while frontier:
                frontier.sort(key=lambda x: x[0])
                cost, node = frontier.pop(0)
                if cost > dist_time.get(node, 999999):
                    continue
                for nbr in self._adjacency[node]:
                    edge = self._edges.get((node, nbr))
                    if edge is None:
                        continue
                    new_cost = dist_time[node] + edge.base_travel_time
                    if new_cost < dist_time.get(nbr, 999999):
                        dist_time[nbr] = new_cost
                        frontier.append((new_cost, nbr))
            for dst, d in dist_time.items():
                self._sp_time[(src, dst)] = d

    def _sp_hops_dist(self, a: str, b: str) -> int:
        return self._sp_hops.get((a, b), 999)

    def _sp_time_dist(self, a: str, b: str) -> int:
        return self._sp_time.get((a, b), 999)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> SupplyChainObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_step = 0
        self._total_delivered = 0
        self._total_expired = 0
        self._total_spawned = 0
        self._parcels.clear()
        self._active_weather_events.clear()
        self._demand_surge_steps_remaining = 0
        self._failed_hub_ids.clear()

        # Reset contract system
        self._contracts.clear()
        self._total_contracts_completed = 0
        self._total_contracts_failed = 0
        self._next_contract_spawn_step = CONTRACT_SPAWN_INTERVAL

        # FIX: always reset as a list
        self._cascade_events = []
        self._rolling = _RollingWindow(size=100)

        if self._seed is not None:
            self._rng.seed(self._seed)

        for hub in self._hubs.values():
            hub.parcel_ids.clear()
            hub.restore()

        for edge in self._edges.values():
            edge.parcels_in_transit = 0
            edge.background_load = 0
            edge.has_delay = False
            edge.is_blocked = False
            edge.in_service_window = True
            edge.weather_delay_boost = 0.0
            edge.weather_congestion_boost = 0

        for _ in range(5):
            self._spawn_parcel()

        return self._make_observation(
            reward=0.0, done=False,
            reward_breakdown={k: 0.0 for k in _REWARD_KEYS},
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: SupplyChainAction) -> SupplyChainObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._current_step += 1

        rb: Dict[str, float] = {k: 0.0 for k in _REWARD_KEYS}

        self._advance_in_transit()

        illegal_moves = self._apply_action(action, rb)
        rb["illegal_move_penalty"] += illegal_moves * R_ILLEGAL_MOVE

        delivered_ids = self._process_deliveries(rb)
        expired_ids = self._process_expirations(rb)

        moved_parcel_ids: Set[str] = {d.parcel_id for d in action.routing_decisions}
        self._apply_waiting_penalty(moved_parcel_ids, rb)
        self._apply_capacity_penalties(rb)

        # Update contracts and cascade events
        self._update_contracts(delivered_ids, rb)
        self._update_cascade_events(moved_parcel_ids, rb)

        # Stochastic event updates
        self._update_edge_events()
        self._update_weather_events()
        self._update_hub_failures()
        self._update_demand_surge()
        self._update_service_windows()

        # Maybe spawn a cascade event
        self._maybe_spawn_cascade()

        # Parcel spawning with demand curve
        step_idx = min(self._current_step - 1, self._max_steps - 1)
        step_multiplier = self._data.demand_curve[step_idx]
        if self._demand_surge_steps_remaining > 0:
            effective_rate = DEMAND_SURGE_SPAWN_RATE
        else:
            effective_rate = BASE_PARCEL_SPAWN_RATE * step_multiplier

        n_new = self._rng.poisson_approx(effective_rate)
        active_count = sum(1 for p in self._parcels.values()
                           if not p.is_delivered and not p.is_expired)
        for _ in range(n_new):
            if active_count < MAX_ACTIVE_PARCELS:
                self._spawn_parcel()
                active_count += 1

        # Maybe spawn a new contract
        if self._current_step >= self._next_contract_spawn_step:
            self._maybe_spawn_contract()
            self._next_contract_spawn_step = self._current_step + CONTRACT_SPAWN_INTERVAL

        if self._current_step % PARCEL_PRUNE_INTERVAL == 0:
            self._prune_completed_parcels()

        # Update rolling KPIs
        self._rolling.record(
            self._current_step,
            len(delivered_ids),
            len(expired_ids),
        )

        done = self._current_step >= self._max_steps
        total_reward = sum(rb.values())

        return self._make_observation(
            reward=total_reward, done=done,
            reward_breakdown=rb,
            delivered_ids=delivered_ids,
            expired_ids=expired_ids,
        )

    # ------------------------------------------------------------------
    # Contract spawning and tracking
    # ------------------------------------------------------------------

    def _maybe_spawn_contract(self):
        """Spawn a new contract if below the cap."""
        active_contracts = sum(
            1 for c in self._contracts.values()
            if not c.is_completed and not c.is_failed
        )
        if active_contracts >= MAX_ACTIVE_CONTRACTS:
            return

        local_hubs = [hid for hid, h in self._hubs.items() if h.tier == 2]
        if len(local_hubs) < 2:
            return

        origin, dest = self._rng.sample(local_hubs, 2)
        travel_time = self._sp_time_dist(origin, dest)
        target_count = self._rng.randint(CONTRACT_MIN_PARCELS, CONTRACT_MAX_PARCELS)
        deadline = (
            self._current_step
            + max(50, int(travel_time * CONTRACT_DEADLINE_SLACK * target_count)
                  + self._rng.randint(0, 20))
        )
        priority = round(self._rng.choice(CONTRACT_PRIORITY_CHOICES), 1)

        cid = f"C{len(self._contracts):04d}"
        contract = _Contract(
            contract_id=cid,
            origin_hub=origin,
            dest_hub=dest,
            origin_city=self._hubs[origin].city,
            dest_city=self._hubs[dest].city,
            target_count=target_count,
            deadline=deadline,
            priority=priority,
            spawn_step=self._current_step,
        )
        self._contracts[cid] = contract

        # FIX: collect spawned parcel IDs directly from _spawn_parcel return value
        for _ in range(target_count):
            active_count = sum(1 for p in self._parcels.values()
                               if not p.is_delivered and not p.is_expired)
            if active_count >= MAX_ACTIVE_PARCELS:
                break
            pid = self._spawn_parcel(
                force_origin=origin,
                force_dest=dest,
                contract_id=cid,
                deadline=deadline,
            )
            if pid is not None:
                contract.assigned_parcel_ids.add(pid)

    def _update_contracts(self, delivered_ids: List[str], rb: Dict[str, float]):
        """Check if any delivered parcel satisfies a contract. Score completed/failed contracts."""
        for pid in delivered_ids:
            parcel = self._parcels.get(pid)
            if parcel is None or parcel.contract_id is None:
                continue
            contract = self._contracts.get(parcel.contract_id)
            if contract is None or contract.is_completed or contract.is_failed:
                continue
            contract.delivered_parcel_ids.add(pid)

            if contract.delivered_count >= contract.target_count:
                contract.is_completed = True
                self._total_contracts_completed += 1
                steps_early = max(0, contract.deadline - self._current_step)
                bonus = (
                    R_CONTRACT_COMPLETION_BASE * contract.priority
                    + R_CONTRACT_EARLY_BONUS * steps_early * contract.priority
                )
                rb["contract_bonus"] += bonus

        # Check for expired contracts
        for contract in self._contracts.values():
            if contract.is_completed or contract.is_failed:
                continue
            if self._current_step >= contract.deadline:
                contract.is_failed = True
                self._total_contracts_failed += 1
                partial = R_CONTRACT_PARTIAL_CREDIT * contract.delivered_count * contract.priority
                penalty = R_CONTRACT_FAILURE_PENALTY * contract.priority
                rb["contract_bonus"] += penalty + partial

    # ------------------------------------------------------------------
    # Cascade event spawning and recovery tracking
    # ------------------------------------------------------------------

    def _maybe_spawn_cascade(self):
        """Trigger a cascade crisis: fail a hub and block an adjacent edge."""
        if self._rng.random() > CASCADE_TRIGGER_PROB:
            return

        candidates = [
            hid for hid, h in self._hubs.items()
            if not h.is_failed
            and hid not in self._failed_hub_ids
            and h.load >= CASCADE_MIN_AFFECTED_PARCELS
        ]
        if not candidates:
            return

        failed_hub_id = self._rng.choice(candidates)
        hub = self._hubs[failed_hub_id]

        adj = self._adjacency.get(failed_hub_id, [])
        if not adj:
            return
        block_target = self._rng.choice(adj)
        block_edge = (failed_hub_id, block_target)

        # Parcels at the hub, not currently in transit
        affected = {
            pid for pid in hub.parcel_ids
            if pid in self._parcels
            and not self._parcels[pid].is_delivered
            and not self._parcels[pid].is_expired
            and self._parcels[pid].steps_until_arrival is None
        }
        if len(affected) < CASCADE_MIN_AFFECTED_PARCELS:
            return

        affected = set(list(affected)[:CASCADE_MAX_AFFECTED_PARCELS])

        duration = self._rng.randint(HUB_FAILURE_MIN_DURATION, HUB_FAILURE_MAX_DURATION)
        hub.apply_failure()
        hub.failure_steps_remaining = duration
        self._failed_hub_ids.add(failed_hub_id)

        edge_obj = self._edges.get(block_edge)
        if edge_obj:
            edge_obj.is_blocked = True

        event = _CascadeEvent(
            event_id=f"CAS{len(self._cascade_events):03d}",
            failed_hub_id=failed_hub_id,
            blocked_edge=block_edge,
            affected_parcel_ids=affected,
            start_step=self._current_step,
            recovery_deadline=self._current_step + CASCADE_RECOVERY_WINDOW,
        )
        self._cascade_events.append(event)

    def _update_cascade_events(self, moved_parcel_ids: Set[str], rb: Dict[str, float]):
        """
        Track how many affected parcels get rerouted.
        Score recovery when the window expires.
        FIX: use rerouted_parcel_ids set to count each parcel only once.
        FIX: only keep unresolved events after scoring.
        """
        resolved_this_step = []

        for event in self._cascade_events:
            if event.resolved:
                continue

            # Count each affected parcel that has been moved exactly once
            for pid in event.affected_parcel_ids:
                if pid in moved_parcel_ids and pid not in event.rerouted_parcel_ids:
                    parcel = self._parcels.get(pid)
                    if parcel and parcel.steps_until_arrival is not None:
                        event.rerouted_parcel_ids.add(pid)

            # Score when recovery window expires
            if self._current_step >= event.recovery_deadline:
                event.resolved = True
                resolved_this_step.append(event)
                frac = event.recovery_fraction
                if frac >= 1.0:
                    rb["recovery_bonus"] += R_RECOVERY_FULL_BONUS
                elif frac > 0:
                    rb["recovery_bonus"] += R_RECOVERY_FULL_BONUS * frac * R_RECOVERY_PARTIAL_RATIO
                else:
                    rb["recovery_bonus"] += R_RECOVERY_FAILURE_PENALTY

        # FIX: keep only unresolved events
        self._cascade_events = [e for e in self._cascade_events if not e.resolved]

    # ------------------------------------------------------------------
    # Stochastic event updates
    # ------------------------------------------------------------------

    def _update_edge_events(self):
        in_rush_hour = any(
            start <= self._current_step <= end
            for start, end in RUSH_HOUR_WINDOWS
        )
        for edge in self._edges.values():
            effective_delay_prob = DELAY_PROB + edge.weather_delay_boost
            edge.has_delay = self._rng.random() < effective_delay_prob
            edge.is_blocked = self._rng.random() < BLOCK_PROB
            edge.background_load = RUSH_HOUR_BACKGROUND_PARCELS if in_rush_hour else 0

    def _update_weather_events(self):
        still_active = []
        for event in self._active_weather_events:
            event.steps_remaining -= 1
            if event.steps_remaining > 0:
                still_active.append(event)
        self._active_weather_events = still_active

        active_zones = {e.zone for e in self._active_weather_events}
        for zid, zdata in self._data.weather_zones.items():
            if zid not in active_zones:
                if self._rng.random() < zdata["event_prob"]:
                    duration = self._rng.randint(zdata["min_duration"], zdata["max_duration"])
                    self._active_weather_events.append(
                        _WeatherEvent(zone=zid, duration=duration, zone_data=zdata)
                    )

        weather_zones_active = {e.zone for e in self._active_weather_events}
        for edge in self._edges.values():
            if edge.weather_zone is not None and edge.weather_zone in weather_zones_active:
                zdata = self._data.weather_zones[edge.weather_zone]
                edge.weather_delay_boost = zdata["delay_prob_boost"]
                edge.weather_congestion_boost = zdata["congestion_boost"]
            else:
                edge.weather_delay_boost = 0.0
                edge.weather_congestion_boost = 0

    def _update_hub_failures(self):
        recovered = []
        for hub_id in list(self._failed_hub_ids):
            hub = self._hubs[hub_id]
            hub.failure_steps_remaining -= 1
            if hub.failure_steps_remaining <= 0:
                hub.restore()
                recovered.append(hub_id)
        for hub_id in recovered:
            self._failed_hub_ids.discard(hub_id)

        for hub_id, hub in self._hubs.items():
            if hub_id in self._failed_hub_ids:
                continue
            if self._rng.random() < HUB_FAILURE_PROB:
                duration = self._rng.randint(
                    HUB_FAILURE_MIN_DURATION, HUB_FAILURE_MAX_DURATION
                )
                hub.apply_failure()
                hub.failure_steps_remaining = duration
                self._failed_hub_ids.add(hub_id)

    def _update_demand_surge(self):
        if self._demand_surge_steps_remaining > 0:
            self._demand_surge_steps_remaining -= 1
        else:
            if self._rng.random() < DEMAND_SURGE_PROB:
                self._demand_surge_steps_remaining = self._rng.randint(
                    DEMAND_SURGE_MIN_DURATION, DEMAND_SURGE_MAX_DURATION
                )

    def _update_service_windows(self):
        in_off_peak = any(
            start <= self._current_step <= end
            for start, end in SERVICE_WINDOWS_OFF_PEAK
        )
        for edge in self._edges.values():
            if edge.tier_type == "mega_to_mega":
                edge.in_service_window = True
            else:
                edge.in_service_window = not in_off_peak

    # ------------------------------------------------------------------
    # Transit, action, deliveries, penalties
    # ------------------------------------------------------------------

    def _advance_in_transit(self):
        for parcel in list(self._parcels.values()):
            if parcel.is_delivered or parcel.is_expired:
                continue
            parcel.steps_in_transit += 1
            if parcel.steps_until_arrival is not None:
                parcel.steps_until_arrival -= 1
                if parcel.steps_until_arrival <= 0:
                    old_hub = parcel.current_hub
                    new_hub = parcel.next_hub
                    edge = self._edges.get((old_hub, new_hub))
                    if edge:
                        edge.parcels_in_transit = max(0, edge.parcels_in_transit - 1)
                    self._hubs[old_hub].parcel_ids.discard(parcel.parcel_id)
                    self._hubs[new_hub].parcel_ids.add(parcel.parcel_id)
                    parcel.current_hub = new_hub
                    parcel.hops_taken += 1
                    parcel.steps_until_arrival = None
                    parcel.next_hub = None

    def _apply_action(self, action: SupplyChainAction, rb: Dict[str, float]) -> int:
        illegal = 0
        for decision in action.routing_decisions:
            pid = decision.parcel_id
            next_hub = decision.next_hub_id

            parcel = self._parcels.get(pid)
            if parcel is None or parcel.is_delivered or parcel.is_expired:
                illegal += 1
                continue
            if parcel.steps_until_arrival is not None:
                illegal += 1
                continue
            if next_hub not in self._adjacency.get(parcel.current_hub, []):
                illegal += 1
                continue

            edge = self._edges.get((parcel.current_hub, next_hub))
            if edge is None or edge.is_blocked:
                illegal += 1
                continue

            # Congestion penalty
            chosen_dist = self._sp_hops_dist(next_hub, parcel.destination)
            has_better_alternative = False
            for nb in self._adjacency.get(parcel.current_hub, []):
                if nb == next_hub:
                    continue
                alt_edge = self._edges.get((parcel.current_hub, nb))
                if alt_edge is None or alt_edge.is_blocked:
                    continue
                alt_dist = self._sp_hops_dist(nb, parcel.destination)
                if alt_dist <= chosen_dist and alt_edge.congestion_level < edge.congestion_level - 0.2:
                    has_better_alternative = True
                    break
            if edge.congestion_level > 0.7 and has_better_alternative:
                rb["congestion_penalty"] += R_CONGESTION_PENALTY

            # Hop progress (suppressed in sparse mode)
            if not self._sparse_rewards:
                old_dist = self._sp_hops_dist(parcel.current_hub, parcel.destination)
                new_dist = self._sp_hops_dist(next_hub, parcel.destination)
                if new_dist < old_dist:
                    rb["hop_progress"] += R_HOP_PROGRESS * parcel.priority

            edge.parcels_in_transit += 1
            parcel.steps_until_arrival = max(1, edge.current_travel_time)
            parcel.next_hub = next_hub

        return illegal

    def _process_deliveries(self, rb: Dict[str, float]) -> List[str]:
        delivered = []
        for parcel in list(self._parcels.values()):
            if parcel.is_delivered or parcel.is_expired:
                continue
            if (parcel.current_hub == parcel.destination
                    and parcel.steps_until_arrival is None):
                parcel.is_delivered = True
                self._hubs[parcel.current_hub].parcel_ids.discard(parcel.parcel_id)
                self._total_delivered += 1
                delivered.append(parcel.parcel_id)

                steps_remaining = parcel.deadline - self._current_step
                if steps_remaining >= 0:
                    delivery_reward = (
                        R_DELIVERY_BASE * parcel.priority
                        + R_DELIVERY_EARLY_BONUS * steps_remaining * parcel.priority
                    )
                else:
                    delivery_reward = R_LATENESS_PER_STEP * abs(steps_remaining) * parcel.priority
                rb["delivery_bonus"] += delivery_reward
        return delivered

    def _process_expirations(self, rb: Dict[str, float]) -> List[str]:
        expired = []
        for parcel in list(self._parcels.values()):
            if parcel.is_delivered or parcel.is_expired:
                continue
            if self._current_step >= parcel.deadline:
                parcel.is_expired = True
                hub = self._hubs.get(parcel.current_hub)
                if hub:
                    hub.parcel_ids.discard(parcel.parcel_id)
                self._total_expired += 1
                expired.append(parcel.parcel_id)
                rb["lateness_penalty"] += R_EXPIRY_PENALTY * parcel.priority
        return expired

    def _apply_waiting_penalty(self, moved_ids: Set[str], rb: Dict[str, float]):
        for parcel in self._parcels.values():
            if parcel.is_delivered or parcel.is_expired:
                continue
            if parcel.steps_until_arrival is not None:
                continue
            if parcel.parcel_id in moved_ids:
                continue
            has_valid_move = any(
                not self._edges[(parcel.current_hub, nb)].is_blocked
                for nb in self._adjacency.get(parcel.current_hub, [])
                if (parcel.current_hub, nb) in self._edges
            )
            if has_valid_move:
                rb["waiting_penalty"] += R_WAITING_PER_STEP

    def _apply_capacity_penalties(self, rb: Dict[str, float]):
        for hub in self._hubs.values():
            if hub.is_overloaded:
                overflow = hub.load - hub.capacity
                rb["capacity_penalty"] += R_CAPACITY_PENALTY * overflow

    # ------------------------------------------------------------------
    # Parcel spawning and pruning
    # ------------------------------------------------------------------

    def _spawn_parcel(
        self,
        force_origin: Optional[str] = None,
        force_dest: Optional[str] = None,
        contract_id: Optional[str] = None,
        deadline: Optional[int] = None,
    ) -> Optional[str]:
        """
        Spawn a parcel and return its parcel_id.
        FIX: returns the pid so callers can track it without fragile index arithmetic.
        """
        local_hubs = [hid for hid, h in self._hubs.items() if h.tier == 2]
        if force_origin and force_dest:
            origin, destination = force_origin, force_dest
        else:
            origin, destination = self._rng.sample(local_hubs, 2)

        travel_time_shortest = self._sp_time_dist(origin, destination)
        if deadline is None:
            deadline = (
                self._current_step
                + max(10, int(travel_time_shortest * DEADLINE_SLACK) + self._rng.randint(0, 4))
            )
        priority = round(self._rng.choice([0.5, 1.0, 1.5, 2.0]), 1)

        pid = f"P{self._total_spawned:04d}"
        self._total_spawned += 1

        parcel = _Parcel(
            parcel_id=pid,
            origin=origin,
            destination=destination,
            spawn_step=self._current_step,
            deadline=deadline,
            priority=priority,
            contract_id=contract_id,
        )
        self._parcels[pid] = parcel
        self._hubs[origin].parcel_ids.add(pid)
        return pid

    def _prune_completed_parcels(self):
        to_delete = [
            pid for pid, p in self._parcels.items()
            if p.is_delivered or p.is_expired
        ]
        for pid in to_delete:
            del self._parcels[pid]

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        reward: float,
        done: bool,
        reward_breakdown: Dict[str, float],
        delivered_ids: Optional[List[str]] = None,
        expired_ids: Optional[List[str]] = None,
    ) -> SupplyChainObservation:

        hub_statuses = [
            HubStatus(
                hub_id=h.hub_id,
                tier=h.tier,
                current_load=h.load,
                capacity=h.capacity,
                load_ratio=round(min(1.0, h.load / h.capacity), 3),
                is_overloaded=h.is_overloaded,
            )
            for h in self._hubs.values()
        ]

        edge_statuses = [
            EdgeStatus(
                from_hub=e.from_hub,
                to_hub=e.to_hub,
                base_travel_time=e.base_travel_time,
                current_travel_time=e.current_travel_time,
                congestion_level=round(e.congestion_level, 3),
                has_random_delay=e.has_delay,
                is_blocked=e.is_blocked,
            )
            for e in self._edges.values()
        ]

        active_parcel_statuses = [
            ParcelStatus(
                parcel_id=p.parcel_id,
                origin_hub=p.origin,
                destination_hub=p.destination,
                current_hub=p.current_hub,
                hops_taken=p.hops_taken,
                steps_in_transit=p.steps_in_transit,
                deadline=p.deadline,
                steps_remaining=p.deadline - self._current_step,
                priority=p.priority,
                is_delivered=p.is_delivered,
                is_expired=p.is_expired,
            )
            for p in self._parcels.values()
            if not p.is_delivered and not p.is_expired
        ]

        active_contracts = [
            {
                "contract_id":       c.contract_id,
                "origin_hub":        c.origin_hub,
                "dest_hub":          c.dest_hub,
                "origin_city":       c.origin_city,
                "dest_city":         c.dest_city,
                "target_count":      c.target_count,
                "delivered_count":   c.delivered_count,
                "remaining_count":   c.remaining_count,
                "deadline":          c.deadline,
                "steps_to_deadline": c.deadline - self._current_step,
                "priority":          c.priority,
                "completion_frac":   round(c.completion_fraction, 3),
                "assigned_parcels":  list(c.assigned_parcel_ids),
            }
            for c in self._contracts.values()
            if not c.is_completed and not c.is_failed
        ]

        active_cascades = [
            {
                "event_id":          e.event_id,
                "failed_hub":        e.failed_hub_id,
                "failed_city":       self._hubs[e.failed_hub_id].city,
                "blocked_edge":      list(e.blocked_edge),
                "affected_parcels":  list(e.affected_parcel_ids),
                "total_affected":    e.total_affected,
                "rerouted_count":    e.rerouted_count,
                "recovery_fraction": round(e.recovery_fraction, 3),
                "steps_to_deadline": e.recovery_deadline - self._current_step,
                "urgent":            (e.recovery_deadline - self._current_step) < 30,
            }
            for e in self._cascade_events
            if not e.resolved
        ]

        weather_zone_info = [
            {
                "zone":            ev.zone,
                "zone_name":       ev.zone_name,
                "steps_remaining": ev.steps_remaining,
                "affected_hubs":   self._data.weather_zones[ev.zone]["hub_ids"],
                "imd_division":    self._data.weather_zones[ev.zone]["imd_division"],
            }
            for ev in self._active_weather_events
        ]

        step_idx = min(self._current_step, self._max_steps - 1)
        demand_multiplier = self._data.demand_curve[step_idx]

        parcel_contract_map = {
            p.parcel_id: p.contract_id
            for p in self._parcels.values()
            if p.contract_id is not None and not p.is_delivered and not p.is_expired
        }

        return SupplyChainObservation(
            hubs=hub_statuses,
            edges=edge_statuses,
            active_parcels=active_parcel_statuses,
            delivered_this_step=delivered_ids or [],
            expired_this_step=expired_ids or [],
            current_step=self._current_step,
            total_delivered=self._total_delivered,
            total_expired=self._total_expired,
            total_parcels_spawned=self._total_spawned,
            reward=reward,
            reward_breakdown=reward_breakdown,
            done=done,
            metadata={
                "episode_id":              self._state.episode_id,
                "step_count":              self._state.step_count,
                "max_steps":               self._max_steps,
                "active_parcel_count":     len(active_parcel_statuses),
                "sparse_reward_mode":      self._sparse_rewards,

                "in_rush_hour": any(
                    s <= self._current_step <= e
                    for s, e in RUSH_HOUR_WINDOWS
                ),

                "demand_multiplier":              round(demand_multiplier, 3),
                "demand_surge_active":            self._demand_surge_steps_remaining > 0,
                "demand_surge_steps_remaining":   self._demand_surge_steps_remaining,

                "weather_zones":         weather_zone_info,
                "weather_zones_active":  [ev.zone for ev in self._active_weather_events],

                "failed_hubs":       list(self._failed_hub_ids),
                "failed_hub_cities": [self._hubs[h].city for h in self._failed_hub_ids],

                "in_off_peak": any(
                    s <= self._current_step <= e
                    for s, e in SERVICE_WINDOWS_OFF_PEAK
                ),

                "active_contracts":          active_contracts,
                "total_contracts_completed": self._total_contracts_completed,
                "total_contracts_failed":    self._total_contracts_failed,
                "parcel_contract_map":       parcel_contract_map,

                "active_cascades":        active_cascades,
                "cascade_recovery_needed": len(active_cascades) > 0,

                "delivered_last_100": self._rolling.delivered_last_window,
                "expired_last_100":   self._rolling.expired_last_window,
                "delivery_rate_100": round(
                    self._rolling.delivered_last_window /
                    max(1, self._rolling.delivered_last_window + self._rolling.expired_last_window),
                    3,
                ),
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_hub_info(self, hub_id: str) -> Optional[Dict]:
        hub = self._hubs.get(hub_id)
        if hub is None:
            return None
        return {
            "hub_id": hub.hub_id, "city": hub.city, "state": hub.state,
            "tier": hub.tier, "lat": hub.lat, "lon": hub.lon, "notes": hub.notes,
        }

    def get_edge_info(self, from_hub: str, to_hub: str) -> Optional[Dict]:
        edge = self._edges.get((from_hub, to_hub))
        if edge is None:
            return None
        return {
            "from_hub": edge.from_hub, "to_hub": edge.to_hub,
            "from_city": edge.from_city, "to_city": edge.to_city,
            "distance_km": edge.distance_km, "highway_notes": edge.highway_notes,
            "tier_type": edge.tier_type,
        }


# ---------------------------------------------------------------------------
# Poisson sampler
# ---------------------------------------------------------------------------

class _PoissonRandom(random.Random):
    def poisson_approx(self, lam: float) -> int:
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self.random()
        return k - 1


# ---------------------------------------------------------------------------
# Smoke test — python supply_chain_environment.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Supply Chain v1.0 — Long-Horizon Smoke Test ===")
    print(f"MAX_STEPS={MAX_STEPS}, SPARSE_REWARD_MODE={SPARSE_REWARD_MODE}")

    env = SupplyChainEnvironment(seed=42)
    obs = env.reset()
    print(f"\nReset: {len(obs.active_parcels)} parcels, {len(obs.hubs)} hubs, {len(obs.edges)} edges")

    total_reward = 0.0
    log_interval = 100

    for step_i in range(MAX_STEPS):
        decisions = []
        for p in obs.active_parcels:
            if p.is_delivered or p.is_expired:
                continue
            best_nb, best_dist = None, 999999
            for e in obs.edges:
                if e.from_hub != p.current_hub or e.is_blocked:
                    continue
                d = env._sp_time_dist(e.to_hub, p.destination_hub)
                if d < best_dist:
                    best_dist = d
                    best_nb = e.to_hub
            if best_nb:
                decisions.append(
                    ParcelRouteDecision(parcel_id=p.parcel_id, next_hub_id=best_nb)
                )

        obs = env.step(SupplyChainAction(routing_decisions=decisions))
        total_reward += obs.reward

        if (step_i + 1) % log_interval == 0 or obs.done:
            flags = []
            if obs.metadata["in_rush_hour"]:             flags.append("RUSH")
            if obs.metadata["demand_surge_active"]:      flags.append("SURGE")
            if obs.metadata["cascade_recovery_needed"]:  flags.append("CASCADE!")
            if obs.metadata["active_contracts"]:         flags.append(f"CONTRACTS:{len(obs.metadata['active_contracts'])}")

            print(
                f"Step {step_i+1:5d}/{MAX_STEPS} | "
                f"reward={obs.reward:+.2f} | "
                f"total={total_reward:+.1f} | "
                f"del={obs.total_delivered} | "
                f"exp={obs.total_expired} | "
                f"contracts: done={obs.metadata['total_contracts_completed']} "
                f"fail={obs.metadata['total_contracts_failed']} | "
                f"rate_100={obs.metadata['delivery_rate_100']:.2f} | "
                + (" ".join(flags) if flags else "normal")
            )

        if obs.done:
            break

    print(f"\n=== Episode complete ===")
    print(f"Total reward:       {total_reward:.2f}")
    print(f"Delivered:          {obs.total_delivered}")
    print(f"Expired:            {obs.total_expired}")
    print(f"Contracts done:     {obs.metadata['total_contracts_completed']}")
    print(f"Contracts failed:   {obs.metadata['total_contracts_failed']}")
    print(f"Final delivery rate (last 100): {obs.metadata['delivery_rate_100']:.2%}")
