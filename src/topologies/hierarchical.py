"""Hierarchical (5-level stratified) MAS topology — experimental group.

The hierarchy mirrors a corporate reporting chain where final consensus is
accepted only when emitted by Level-1 Orchestrator.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from src.agents.analyst_prefab import AnalystPrefab
from src.agents.director_prefab import DirectorPrefab
from src.agents.manager_prefab import ManagerPrefab
from src.agents.orchestrator_prefab import OrchestratorPrefab
from src.config import ExperimentConfig

_VALID_LEVELS = (1, 2, 3, 4, 5)

TopologyAgent: TypeAlias = (
    AnalystPrefab | DirectorPrefab | ManagerPrefab | OrchestratorPrefab
)
HierarchyLevels: TypeAlias = dict[int, list[TopologyAgent]]


class HierarchicalTopology:
    """Constructs a 5-level reporting chain with orchestrator authority."""

    def build(
        self,
        config: ExperimentConfig,
        names: dict[int, list[str]] | None = None,
        hallucinated_premise: str = "",
    ) -> HierarchyLevels:
        """Instantiate and return prefabs keyed by hierarchical level.

        Level constraints are validated as:
        - Level 1 count must be exactly 1
        - Level 2 count must be in [2, 3]
        - Level 3 count must be in [3, 5]
        - Level 4 count must be in [3, 5]
        - Level 5 count must be >= 1
        """
        level_counts = self._validate_level_counts(config)
        self._validate_name_mapping(names, level_counts)

        level_names = names or self._default_level_names(level_counts)
        hallucination_prompt_version = config.hallucination.prompt_version

        levels: HierarchyLevels = {
            1: [
                OrchestratorPrefab(
                    params={
                        "name": level_names[1][0],
                        "hallucinated_premise": hallucinated_premise,
                        "prompt_version": hallucination_prompt_version,
                    }
                )
            ],
            2: [DirectorPrefab(params={"name": name}) for name in level_names[2]],
            3: [
                ManagerPrefab(params={"name": name, "rank": 3})
                for name in level_names[3]
            ],
            4: [
                ManagerPrefab(params={"name": name, "rank": 4})
                for name in level_names[4]
            ],
            5: [AnalystPrefab(params={"name": name}) for name in level_names[5]],
        }
        return levels

    def get_reporting_chain(
        self,
        agent: TopologyAgent,
        all_agents: HierarchyLevels,
        reporting_assignments: Mapping[str, str] | None = None,
    ) -> list[TopologyAgent]:
        """Return the superior chain from the agent's next level to Level 1.

        Args:
            agent: Agent instance present in ``all_agents``.
            all_agents: Level->agents map produced by :meth:`build`.
            reporting_assignments: Optional child-name -> manager-name map.
                When present, chain traversal follows these explicit links and
                must terminate at Level 1.
        """
        level_of_agent = self._infer_level(agent, all_agents)
        if level_of_agent == 1:
            return []

        if reporting_assignments is not None:
            chain_from_assignments = self._chain_from_reporting_assignments(
                agent=agent,
                all_agents=all_agents,
                reporting_assignments=reporting_assignments,
            )
            if not chain_from_assignments:
                raise ValueError(
                    "reporting_assignments provided but no manager mapping found "
                    f"for '{agent.params.get('name', '')}'"
                )

            if self._infer_level(chain_from_assignments[-1], all_agents) != 1:
                raise ValueError(
                    "reporting_assignments chain must terminate at Level-1 orchestrator"
                )

            return chain_from_assignments

        chain: list[TopologyAgent] = []
        for level in range(level_of_agent - 1, 0, -1):
            superiors = all_agents.get(level, [])
            if superiors:
                chain.append(superiors[0])
        return chain

    def enforce_orchestrator_consensus(
        self,
        speaker_name: str,
        levels: Mapping[int, list[TopologyAgent]],
    ) -> bool:
        """Return True only if the final consensus speaker is Level-1.

        This helper is intended for ``Simulation`` / Game Master code paths
        that validate final consensus authority.
        """
        if 1 not in levels or len(levels[1]) != 1:
            raise ValueError(
                "invalid hierarchy: levels must include exactly one Level-1 orchestrator"
            )

        orchestrator = levels[1][0]
        orchestrator_name = orchestrator.params.get("name", "")
        return speaker_name == orchestrator_name

    def _validate_level_counts(self, config: ExperimentConfig) -> dict[int, int]:
        level_counts = dict(config.topology.level_counts)
        if sorted(level_counts.keys()) != list(_VALID_LEVELS):
            raise ValueError("topology.level_counts must include levels 1..5")

        if level_counts[1] != 1:
            raise ValueError("Level 1 must contain exactly one orchestrator")
        if not 2 <= level_counts[2] <= 3:
            raise ValueError("Level 2 must contain 2-3 senior directors")
        if not 3 <= level_counts[3] <= 5:
            raise ValueError("Level 3 must contain 3-5 senior managers")
        if not 3 <= level_counts[4] <= 5:
            raise ValueError("Level 4 must contain 3-5 junior managers")
        if level_counts[5] < 1:
            raise ValueError("Level 5 must contain at least one analyst")

        expected_total = sum(level_counts.values())
        if config.topology.num_agents != expected_total:
            raise ValueError(
                "topology.num_agents must equal sum(level_counts): "
                f"{config.topology.num_agents} != {expected_total}"
            )
        return level_counts

    def _validate_name_mapping(
        self,
        names: dict[int, list[str]] | None,
        level_counts: Mapping[int, int],
    ) -> None:
        if names is None:
            return

        if sorted(names.keys()) != list(_VALID_LEVELS):
            raise ValueError("names must contain levels 1..5")

        all_names: list[str] = []
        for level in _VALID_LEVELS:
            if len(names[level]) != level_counts[level]:
                raise ValueError(
                    f"names[{level}] must contain {level_counts[level]} entries, "
                    f"got {len(names[level])}"
                )
            if len(names[level]) != len(set(names[level])):
                raise ValueError(f"names[{level}] must be unique within the level")
            all_names.extend(names[level])

        if len(all_names) != len(set(all_names)):
            raise ValueError("agent names must be globally unique across all levels")

    def _default_level_names(self, level_counts: Mapping[int, int]) -> dict[int, list[str]]:
        return {
            1: ["Orchestrator_01"],
            2: [f"Director_{i:02d}" for i in range(1, level_counts[2] + 1)],
            3: [f"SeniorManager_{i:02d}" for i in range(1, level_counts[3] + 1)],
            4: [f"Manager_{i:02d}" for i in range(1, level_counts[4] + 1)],
            5: [f"Analyst_{i:02d}" for i in range(1, level_counts[5] + 1)],
        }

    def _infer_level(
        self,
        agent: TopologyAgent,
        all_agents: Mapping[int, list[TopologyAgent]],
    ) -> int:
        for level, level_agents in all_agents.items():
            for level_agent in level_agents:
                if level_agent is agent:
                    return level
        raise ValueError("agent is not present in all_agents")

    def _chain_from_reporting_assignments(
        self,
        agent: TopologyAgent,
        all_agents: Mapping[int, list[TopologyAgent]],
        reporting_assignments: Mapping[str, str],
    ) -> list[TopologyAgent]:
        by_name = {
            member.params.get("name", ""): member
            for members in all_agents.values()
            for member in members
        }
        start_name = agent.params.get("name", "")
        chain: list[TopologyAgent] = []

        seen: set[str] = {start_name}
        cursor_name = start_name
        while cursor_name in reporting_assignments:
            manager_name = reporting_assignments[cursor_name]
            manager = by_name.get(manager_name)
            if manager is None:
                raise ValueError(
                    "reporting_assignments references unknown manager name "
                    f"'{manager_name}' for child '{cursor_name}'"
                )
            if manager_name in seen:
                raise ValueError("cycle detected in reporting_assignments")

            chain.append(manager)
            seen.add(manager_name)

            manager_level = self._infer_level(manager, all_agents)
            if manager_level == 1:
                break
            cursor_name = manager_name
        return chain
