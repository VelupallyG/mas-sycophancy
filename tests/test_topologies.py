"""Tests for topology construction and scheduling helpers."""
from __future__ import annotations

import pytest

from src.agents.analyst_prefab import AnalystPrefab
from src.agents.director_prefab import DirectorPrefab
from src.agents.manager_prefab import ManagerPrefab
from src.agents.orchestrator_prefab import OrchestratorPrefab
from src.config import ExperimentConfig
from src.topologies.flat import FlatTopology
from src.topologies.hierarchical import HierarchicalTopology


def _make_valid_hierarchical_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.topology.level_counts = {1: 1, 2: 2, 3: 4, 4: 4, 5: 9}
    config.topology.num_agents = sum(config.topology.level_counts.values())
    return config


def test_flat_topology_agent_count() -> None:
    config = ExperimentConfig()
    config.topology.num_agents = 10
    agents = FlatTopology().build(config)
    assert len(agents) == 10


def test_flat_topology_all_level_5_analysts() -> None:
    config = ExperimentConfig()
    agents = FlatTopology().build(config)
    assert all(isinstance(agent, AnalystPrefab) for agent in agents)


def test_flat_topology_custom_names_applied() -> None:
    config = ExperimentConfig()
    config.topology.num_agents = 3
    agents = FlatTopology().build(config, names=["A", "B", "C"])
    assert [agent.params["name"] for agent in agents] == ["A", "B", "C"]


def test_flat_topology_invalid_names_raises() -> None:
    config = ExperimentConfig()
    config.topology.num_agents = 3
    with pytest.raises(ValueError):
        FlatTopology().build(config, names=["A", "B"])


def test_flat_topology_duplicate_names_raise() -> None:
    config = ExperimentConfig()
    config.topology.num_agents = 3
    with pytest.raises(ValueError):
        FlatTopology().build(config, names=["A", "A", "B"])


def test_flat_topology_round_robin_component() -> None:
    component = FlatTopology().build_turn_taking_policy(["A", "B", "C"])
    assert component.mode == "round_robin"
    assert component.order == ["A", "B", "C"]
    assert component.override_authority is False


def test_flat_topology_simultaneous_component() -> None:
    component = FlatTopology().build_turn_taking_policy(
        ["A", "B", "C"], simultaneous=True
    )
    assert component.mode == "simultaneous"


def test_flat_topology_turn_taking_rejects_empty() -> None:
    with pytest.raises(ValueError):
        FlatTopology().build_turn_taking_policy([])


def test_flat_topology_turn_taking_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        FlatTopology().build_turn_taking_policy(["A", "A"])


def test_hierarchical_topology_level_counts() -> None:
    config = _make_valid_hierarchical_config()
    levels = HierarchicalTopology().build(config)
    for level, expected_count in config.topology.level_counts.items():
        assert len(levels[level]) == expected_count


def test_hierarchical_level_1_contains_orchestrator() -> None:
    config = _make_valid_hierarchical_config()
    levels = HierarchicalTopology().build(config)
    assert len(levels[1]) == 1
    assert isinstance(levels[1][0], OrchestratorPrefab)


def test_hierarchical_passes_hallucinated_premise_to_orchestrator() -> None:
    config = _make_valid_hierarchical_config()
    claim = "Fabricated market signal"
    levels = HierarchicalTopology().build(config, hallucinated_premise=claim)
    assert levels[1][0].params["hallucinated_premise"] == claim


def test_hierarchical_level_types() -> None:
    config = _make_valid_hierarchical_config()
    levels = HierarchicalTopology().build(config)
    assert all(isinstance(agent, DirectorPrefab) for agent in levels[2])
    assert all(isinstance(agent, ManagerPrefab) for agent in levels[3])
    assert all(isinstance(agent, ManagerPrefab) for agent in levels[4])
    assert all(isinstance(agent, AnalystPrefab) for agent in levels[5])
    assert all(agent.params["rank"] == 3 for agent in levels[3])
    assert all(agent.params["rank"] == 4 for agent in levels[4])


def test_hierarchical_custom_names_applied() -> None:
    config = _make_valid_hierarchical_config()
    names = {
        1: ["CSO"],
        2: ["D1", "D2"],
        3: ["SM1", "SM2", "SM3", "SM4"],
        4: ["M1", "M2", "M3", "M4"],
        5: ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    }
    levels = HierarchicalTopology().build(config, names=names)
    assert levels[1][0].params["name"] == "CSO"
    assert [agent.params["name"] for agent in levels[2]] == ["D1", "D2"]


def test_hierarchical_invalid_name_lengths_raise() -> None:
    config = _make_valid_hierarchical_config()
    names = {
        1: ["CSO"],
        2: ["D1"],
        3: ["SM1", "SM2", "SM3", "SM4"],
        4: ["M1", "M2", "M3", "M4"],
        5: ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    }
    with pytest.raises(ValueError):
        HierarchicalTopology().build(config, names=names)


def test_hierarchical_duplicate_names_in_level_raise() -> None:
    config = _make_valid_hierarchical_config()
    names = {
        1: ["CSO"],
        2: ["D1", "D1"],
        3: ["SM1", "SM2", "SM3", "SM4"],
        4: ["M1", "M2", "M3", "M4"],
        5: ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
    }
    with pytest.raises(ValueError):
        HierarchicalTopology().build(config, names=names)


def test_hierarchical_duplicate_names_across_levels_raise() -> None:
    config = _make_valid_hierarchical_config()
    names = {
        1: ["CSO"],
        2: ["D1", "D2"],
        3: ["SM1", "SM2", "SM3", "SM4"],
        4: ["M1", "M2", "M3", "M4"],
        5: ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "D1"],
    }
    with pytest.raises(ValueError):
        HierarchicalTopology().build(config, names=names)


def test_hierarchical_reporting_chain_to_orchestrator() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)
    analyst = levels[5][0]
    chain = topology.get_reporting_chain(analyst, levels)
    assert len(chain) == 4
    assert isinstance(chain[-1], OrchestratorPrefab)


def test_hierarchical_enforces_orchestrator_consensus() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)
    orchestrator_name = levels[1][0].params["name"]
    assert topology.enforce_orchestrator_consensus(orchestrator_name, levels) is True
    assert topology.enforce_orchestrator_consensus("SomeoneElse", levels) is False


def test_hierarchical_consensus_validation_raises_on_invalid_levels() -> None:
    topology = HierarchicalTopology()
    with pytest.raises(ValueError):
        topology.enforce_orchestrator_consensus("CSO", levels={1: []})


def test_hierarchical_reporting_chain_uses_assignments() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)

    analyst = levels[5][0]
    manager = levels[4][1]
    senior_manager = levels[3][2]
    director = levels[2][1]
    orchestrator = levels[1][0]

    assignments = {
        analyst.params["name"]: manager.params["name"],
        manager.params["name"]: senior_manager.params["name"],
        senior_manager.params["name"]: director.params["name"],
        director.params["name"]: orchestrator.params["name"],
    }

    chain = topology.get_reporting_chain(
        agent=analyst,
        all_agents=levels,
        reporting_assignments=assignments,
    )
    assert [member.params["name"] for member in chain] == [
        manager.params["name"],
        senior_manager.params["name"],
        director.params["name"],
        orchestrator.params["name"],
    ]


def test_hierarchical_reporting_assignments_unknown_manager_raises() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)
    analyst = levels[5][0]

    with pytest.raises(ValueError):
        topology.get_reporting_chain(
            agent=analyst,
            all_agents=levels,
            reporting_assignments={analyst.params["name"]: "DOES_NOT_EXIST"},
        )


def test_hierarchical_reporting_assignments_cycle_raises() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)

    analyst = levels[5][0]
    manager = levels[4][0]

    with pytest.raises(ValueError, match="cycle detected"):
        topology.get_reporting_chain(
            agent=analyst,
            all_agents=levels,
            reporting_assignments={
                analyst.params["name"]: manager.params["name"],
                manager.params["name"]: analyst.params["name"],
            },
        )


def test_hierarchical_reporting_assignments_must_terminate_at_orchestrator() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)

    analyst = levels[5][0]
    manager = levels[4][0]

    with pytest.raises(ValueError, match="must terminate at Level-1 orchestrator"):
        topology.get_reporting_chain(
            agent=analyst,
            all_agents=levels,
            reporting_assignments={
                analyst.params["name"]: manager.params["name"],
            },
        )


def test_hierarchical_reporting_assignments_missing_root_mapping_raises() -> None:
    config = _make_valid_hierarchical_config()
    topology = HierarchicalTopology()
    levels = topology.build(config)

    analyst = levels[5][0]

    with pytest.raises(ValueError, match="no manager mapping found"):
        topology.get_reporting_chain(
            agent=analyst,
            all_agents=levels,
            reporting_assignments={},
        )


def test_hierarchical_invalid_counts_raise() -> None:
    config = _make_valid_hierarchical_config()
    config.topology.level_counts[2] = 1
    config.topology.num_agents = sum(config.topology.level_counts.values())
    with pytest.raises(ValueError):
        HierarchicalTopology().build(config)
