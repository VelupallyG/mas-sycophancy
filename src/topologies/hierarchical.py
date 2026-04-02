"""Hierarchical MAS topology definition (experimental group).

The hierarchical topology is a balanced 3-level tree:

  Level 1:  1 Orchestrator (CEO/CSO)
                │
       ┌────┬───┴───┬────┐
  Level 2:  4 Managers
                │
     (each manages exactly 4 analysts)
                │
  Level 3:  16 Entry Analysts

Total: 21 agents. Fan-out is uniform at 4 throughout.

Communication is strictly vertical. The GM enforces this by only routing
observations to the appropriate level (not laterally).

Turn execution order:
  Turn 1 (top-down):   L1 → L2 → L3
  Turns 2–N (bottom-up): L3 → L2 → L1, then L1+L2 feed back down.

This module provides the topology structure. Turn execution logic lives in
src/game_master/simulation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

N_L1 = 1
N_L2_MANAGERS = 4
N_L3_ANALYSTS_PER_MANAGER = 4
N_L3_ANALYSTS = N_L2_MANAGERS * N_L3_ANALYSTS_PER_MANAGER  # 16
N_TOTAL = N_L1 + N_L2_MANAGERS + N_L3_ANALYSTS  # 21


@dataclass
class HierarchicalTopologySpec:
    """Describes the hierarchical topology for a single trial.

    Attributes:
        orchestrator_name: Name of the L1 agent.
        manager_names: Names of the 4 L2 managers, in order.
        analyst_groups: Each element is the list of L3 analyst names
            reporting to the manager at the same index.
    """

    orchestrator_name: str = "orchestrator"
    manager_names: list[str] = field(
        default_factory=lambda: [f"manager_{m:02d}" for m in range(N_L2_MANAGERS)]
    )
    analyst_groups: list[list[str]] = field(
        default_factory=lambda: [
            [
                f"analyst_{m * N_L3_ANALYSTS_PER_MANAGER + a:02d}"
                for a in range(N_L3_ANALYSTS_PER_MANAGER)
            ]
            for m in range(N_L2_MANAGERS)
        ]
    )

    def all_analyst_names(self) -> list[str]:
        """Return the flat list of all L3 analyst names."""
        return [name for group in self.analyst_groups for name in group]

    def all_agent_names(self) -> list[str]:
        """Return all 21 agent names: orchestrator + managers + analysts."""
        return (
            [self.orchestrator_name]
            + self.manager_names
            + self.all_analyst_names()
        )

    def manager_for_analyst(self, analyst_name: str) -> str | None:
        """Return the manager name responsible for a given analyst."""
        for m_idx, group in enumerate(self.analyst_groups):
            if analyst_name in group:
                return self.manager_names[m_idx]
        return None
