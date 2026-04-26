# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Supply Chain RL Environment — v1.0 Long-Horizon Edition.

Adds two new reward keys to _REWARD_KEYS and updates metadata documentation
to reflect contract and cascade recovery fields introduced in v1.0.
All v0.4 fields are unchanged for backward compatibility.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ParcelRouteDecision(Action):
    """
    Routing decision for a single parcel.

    The agent selects, for each active parcel, which next-hop hub to send it
    to. Only neighbours of the parcel's current hub are valid choices; the
    server validates this and applies a penalty for illegal moves.
    """
    parcel_id: str = Field(..., description="Unique identifier of the parcel being routed")
    next_hub_id: str = Field(..., description="Hub ID to route the parcel to next")


class SupplyChainAction(Action):
    """
    Composite action: one routing decision per active parcel.

    The agent may route any subset of currently active parcels. Parcels not
    included in the action stay at their current hub for this time-step.
    """
    routing_decisions: List[ParcelRouteDecision] = Field(
        default_factory=list,
        description=(
            "List of routing decisions, one per parcel the agent wants to move. "
            "Omitting a parcel keeps it waiting at its current hub."
        ),
    )


# ---------------------------------------------------------------------------
# Observation sub-objects
# ---------------------------------------------------------------------------

class HubStatus(Observation):
    """Snapshot of a single hub at the current time-step."""
    hub_id: str = Field(..., description="Unique hub identifier")
    tier: int = Field(..., description="Tier level: 0=Mega, 1=Regional, 2=Local")
    current_load: int = Field(..., description="Number of parcels currently at this hub")
    capacity: int = Field(..., description="Maximum parcels this hub can hold")
    load_ratio: float = Field(..., description="current_load / capacity ∈ [0, 1]")
    is_overloaded: bool = Field(..., description="True if current_load > capacity")


class EdgeStatus(Observation):
    """Snapshot of a single directed edge (route between hubs)."""
    from_hub: str = Field(..., description="Origin hub ID")
    to_hub: str = Field(..., description="Destination hub ID")
    base_travel_time: int = Field(..., description="Steps without delay/congestion")
    current_travel_time: int = Field(..., description="Effective travel time this step")
    congestion_level: float = Field(..., description="Traffic load ∈ [0, 1]")
    has_random_delay: bool = Field(..., description="Stochastic delay active this step")
    is_blocked: bool = Field(..., description="Edge completely impassable this step")


class ParcelStatus(Observation):
    """Snapshot of a single parcel in the network."""
    parcel_id: str = Field(..., description="Unique parcel identifier")
    origin_hub: str = Field(..., description="Where the parcel started")
    destination_hub: str = Field(..., description="Where it needs to reach")
    current_hub: str = Field(..., description="Where the parcel is right now")
    hops_taken: int = Field(..., description="Number of hubs visited so far")
    steps_in_transit: int = Field(..., description="Total env steps since parcel spawned")
    deadline: int = Field(..., description="Absolute step by which parcel must arrive")
    steps_remaining: int = Field(..., description="deadline - current_step")
    priority: float = Field(..., description="Priority weight ∈ [0.5, 2.0]; higher = more important")
    is_delivered: bool = Field(default=False, description="True once parcel reached destination")
    is_expired: bool = Field(default=False, description="True if deadline passed without delivery")


# ---------------------------------------------------------------------------
# Top-level Observation
# ---------------------------------------------------------------------------

class SupplyChainObservation(Observation):
    """
    Full environment observation returned to the agent each step.

    v1.0 additions (all in metadata dict):
      - active_contracts: list of active multi-parcel contract dicts
      - total_contracts_completed / total_contracts_failed: episode KPIs
      - parcel_contract_map: {parcel_id: contract_id} for active contract parcels
      - active_cascades: list of active cascade recovery event dicts
      - cascade_recovery_needed: bool shortcut
      - delivered_last_100 / expired_last_100 / delivery_rate_100: rolling KPIs
      - max_steps: episode length for normalisation
      - sparse_reward_mode: whether dense intermediate rewards are suppressed

    reward_breakdown now has 9 keys (added contract_bonus, recovery_bonus):
      delivery_bonus, lateness_penalty, hop_progress, waiting_penalty,
      capacity_penalty, congestion_penalty, illegal_move_penalty,
      contract_bonus, recovery_bonus
    """

    # Network state
    hubs: List[HubStatus] = Field(default_factory=list, description="Status of every hub")
    edges: List[EdgeStatus] = Field(default_factory=list, description="Status of every edge")

    # Parcel state
    active_parcels: List[ParcelStatus] = Field(
        default_factory=list, description="All parcels currently in the network"
    )
    delivered_this_step: List[str] = Field(
        default_factory=list, description="Parcel IDs delivered during this step"
    )
    expired_this_step: List[str] = Field(
        default_factory=list, description="Parcel IDs that expired (missed deadline) this step"
    )

    # Episode bookkeeping
    current_step: int = Field(default=0, description="Current environment step index")
    total_delivered: int = Field(default=0, description="Cumulative deliveries this episode")
    total_expired: int = Field(default=0, description="Cumulative expirations this episode")
    total_parcels_spawned: int = Field(default=0, description="Total parcels ever spawned")

    # Reward breakdown — v1.0: 9 components (was 7)
    reward: float = Field(default=0.0, description="Total scalar reward this step")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Component rewards: delivery_bonus, lateness_penalty, "
            "capacity_penalty, congestion_penalty, hop_progress, waiting_penalty, "
            "illegal_move_penalty, contract_bonus, recovery_bonus"
        ),
    )

    # Episode terminal flag
    done: bool = Field(default=False, description="True when episode ends")
    metadata: Dict = Field(
        default_factory=dict,
        description=(
            "Diagnostic and planning info. v1.0 keys include: "
            "active_contracts, total_contracts_completed, total_contracts_failed, "
            "parcel_contract_map, active_cascades, cascade_recovery_needed, "
            "delivered_last_100, expired_last_100, delivery_rate_100, "
            "max_steps, sparse_reward_mode, in_rush_hour, demand_multiplier, "
            "demand_surge_active, weather_zones, failed_hubs, in_off_peak."
        ),
    )
