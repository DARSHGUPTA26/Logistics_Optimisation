# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supply Chain RL Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    EdgeStatus,
    HubStatus,
    ParcelRouteDecision,
    ParcelStatus,
    SupplyChainAction,
    SupplyChainObservation,
)


class SupplyChainEnv(EnvClient[SupplyChainAction, SupplyChainObservation, State]):
    """
    Client for the Supply Chain RL Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with low latency.

    Example:
        >>> with SupplyChainEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset()
        ...     print(f"Active parcels: {len(obs.active_parcels)}")
        ...
        ...     decisions = [
        ...         ParcelRouteDecision(parcel_id=obs.active_parcels[0].parcel_id,
        ...                             next_hub_id="R0")
        ...     ]
        ...     result = env.step(SupplyChainAction(routing_decisions=decisions))
        ...     print(f"Reward: {result.reward}")

    Docker Example:
        >>> env = SupplyChainEnv.from_docker_image("supply_chain_env:latest")
        >>> try:
        ...     result = env.reset()
        ...     # run your RL loop
        ... finally:
        ...     env.close()
    """

    def _step_payload(self, action: SupplyChainAction) -> Dict:
        return {
            "routing_decisions": [
                {"parcel_id": d.parcel_id, "next_hub_id": d.next_hub_id}
                for d in action.routing_decisions
            ]
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupplyChainObservation]:
        obs_data = payload.get("observation", {})
        observation = self._parse_observation(obs_data, payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_observation(self, obs_data: Dict, payload: Dict) -> SupplyChainObservation:
        hubs = [
            HubStatus(
                hub_id=h["hub_id"],
                tier=h["tier"],
                current_load=h["current_load"],
                capacity=h["capacity"],
                load_ratio=h["load_ratio"],
                is_overloaded=h["is_overloaded"],
            )
            for h in obs_data.get("hubs", [])
        ]

        edges = [
            EdgeStatus(
                from_hub=e["from_hub"],
                to_hub=e["to_hub"],
                base_travel_time=e["base_travel_time"],
                current_travel_time=e["current_travel_time"],
                congestion_level=e["congestion_level"],
                has_random_delay=e["has_random_delay"],
                is_blocked=e["is_blocked"],
            )
            for e in obs_data.get("edges", [])
        ]

        active_parcels = [
            ParcelStatus(
                parcel_id=p["parcel_id"],
                origin_hub=p["origin_hub"],
                destination_hub=p["destination_hub"],
                current_hub=p["current_hub"],
                hops_taken=p["hops_taken"],
                steps_in_transit=p["steps_in_transit"],
                deadline=p["deadline"],
                steps_remaining=p["steps_remaining"],
                priority=p["priority"],
                is_delivered=p.get("is_delivered", False),
                is_expired=p.get("is_expired", False),
            )
            for p in obs_data.get("active_parcels", [])
        ]

        return SupplyChainObservation(
            hubs=hubs,
            edges=edges,
            active_parcels=active_parcels,
            delivered_this_step=obs_data.get("delivered_this_step", []),
            expired_this_step=obs_data.get("expired_this_step", []),
            current_step=obs_data.get("current_step", 0),
            total_delivered=obs_data.get("total_delivered", 0),
            total_expired=obs_data.get("total_expired", 0),
            total_parcels_spawned=obs_data.get("total_parcels_spawned", 0),
            reward=payload.get("reward", 0.0),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
