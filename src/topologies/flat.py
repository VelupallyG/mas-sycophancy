"""Flat MAS topology definition (control group).

The flat topology is a peer-to-peer structure:
  - 21 agents, all with rank="PEER".
  - Communication: global shared forum — every agent sees every other agent's
    previous-turn output.
  - Decision mechanism: majority vote on prediction_direction after T turns.
  - No single agent has structural authority.

This module provides helper types used by SimulationRunner. The actual
turn execution logic lives in src/game_master/simulation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


N_AGENTS = 21


@dataclass
class FlatTopologySpec:
    """Describes the flat topology for a single trial.

    Attributes:
        n_agents: Total number of peer agents.
        agent_names: Ordered list of agent IDs.
        injector_idx: Index of the agent receiving the hallucination injection
            (None in the flat-baseline condition).
    """

    n_agents: int = N_AGENTS
    agent_names: list[str] = field(
        default_factory=lambda: [f"peer_{i:02d}" for i in range(N_AGENTS)]
    )
    injector_idx: int | None = None

    def injector_name(self) -> str | None:
        """Return the name of the hallucination injector peer, or None."""
        if self.injector_idx is None:
            return None
        return self.agent_names[self.injector_idx]


def majority_vote(direction_counts: dict[str, int]) -> str:
    """Return the modal prediction_direction from a vote tally.

    Args:
        direction_counts: Mapping of direction → count of agents predicting it.

    Returns:
        The direction with the highest count. Ties broken alphabetically
        (deterministic, but arbitrary — meaningful enough for prototype).
    """
    if not direction_counts:
        return "NEUTRAL"
    return max(direction_counts, key=lambda d: (direction_counts[d], d))
