"""Flat (peer-to-peer) MAS topology — control group.

All agents are instantiated as Level-5 ``AnalystPrefab`` entities, so no
agent has rank-based override authority. The topology exposes a turn-taking
policy that the Game Master can map to Concordia sequential/simultaneous
engines.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal

from src.agents.analyst_prefab import AnalystPrefab
from src.config import ExperimentConfig


@dataclasses.dataclass(frozen=True)
class TurnTakingPolicy:
    """Flat-topology turn-taking configuration consumed by the Game Master."""

    mode: Literal["round_robin", "simultaneous"]
    order: list[str]
    override_authority: bool = False


class FlatTopology:
    """Constructs a flat peer-to-peer collective with uniform rank behavior."""

    def build(
        self,
        config: ExperimentConfig,
        names: list[str] | None = None,
    ) -> list[AnalystPrefab]:
        """Instantiate and return ``num_agents`` Level-5 analysts.

        Args:
            config: Experiment configuration driving agent count.
            names: Optional explicit name list. Must match ``num_agents``.

        Returns:
            A list of ``AnalystPrefab`` instances.

        Raises:
            ValueError: If ``names`` length does not match ``num_agents``.
        """
        num_agents = config.topology.num_agents
        if num_agents <= 0:
            raise ValueError("topology.num_agents must be positive")

        if names is not None and len(names) != num_agents:
            raise ValueError(
                "names length must equal config.topology.num_agents "
                f"({num_agents}), got {len(names)}"
            )

        agent_names = names or [f"Agent_{i:02d}" for i in range(1, num_agents + 1)]
        if len(agent_names) != len(set(agent_names)):
            raise ValueError("agent names must be unique")

        return [AnalystPrefab(params={"name": name}) for name in agent_names]

    def build_turn_taking_policy(
        self,
        agent_names: Sequence[str],
        simultaneous: bool = False,
    ) -> TurnTakingPolicy:
        """Return turn-taking policy metadata for flat coordination.

        Args:
            agent_names: Ordered list of participating agent names.
            simultaneous: If ``True``, all agents act each turn. If ``False``,
                turn-taking is round-robin fixed order.

        Raises:
            ValueError: If ``agent_names`` is empty or contains duplicates.
        """
        if not agent_names:
            raise ValueError("agent_names must be non-empty")

        order = list(agent_names)
        if len(order) != len(set(order)):
            raise ValueError("agent_names must be unique")

        return TurnTakingPolicy(
            mode="simultaneous" if simultaneous else "round_robin",
            order=order,
            override_authority=False,
        )

    def build_turn_taking_component(
        self,
        agent_names: Sequence[str],
        simultaneous: bool = False,
    ) -> TurnTakingPolicy:
        """Backward-compatible alias for :meth:`build_turn_taking_policy`."""
        return self.build_turn_taking_policy(
            agent_names=agent_names,
            simultaneous=simultaneous,
        )
