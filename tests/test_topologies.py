"""Tests for topology construction.

Implement in Session 3.  Tests cover:
- FlatTopology.build returns correct agent count
- All flat agents are at Level 5
- HierarchicalTopology.build returns correct level counts
- Level 1 contains exactly one OrchestratorAgent
- Custom name lists are applied correctly
- Invalid name list lengths raise ValueError
"""
import pytest

pytestmark = pytest.mark.skip(reason="Not implemented yet — Session 3")

from src.config import ExperimentConfig


def test_flat_topology_agent_count() -> None:
    from src.topologies.flat import FlatTopology

    config = ExperimentConfig()
    config.topology.num_agents = 10
    agents = FlatTopology().build(config)
    assert len(agents) == 10


def test_flat_topology_all_level_5() -> None:
    from src.topologies.flat import FlatTopology

    config = ExperimentConfig()
    agents = FlatTopology().build(config)
    assert all(a.rank.level == 5 for a in agents)


def test_hierarchical_topology_level_counts() -> None:
    from src.topologies.hierarchical import HierarchicalTopology

    config = ExperimentConfig()
    levels = HierarchicalTopology().build(config)
    for lvl, expected_count in config.topology.level_counts.items():
        assert len(levels[lvl]) == expected_count


def test_hierarchical_topology_level_1_is_orchestrator() -> None:
    from src.topologies.hierarchical import HierarchicalTopology
    from src.agents.orchestrator import OrchestratorAgent

    config = ExperimentConfig()
    levels = HierarchicalTopology().build(config)
    assert len(levels[1]) == 1
    assert isinstance(levels[1][0], OrchestratorAgent)


def test_flat_topology_invalid_names_raises() -> None:
    from src.topologies.flat import FlatTopology

    config = ExperimentConfig()
    config.topology.num_agents = 3
    with pytest.raises(ValueError):
        FlatTopology().build(config, names=["A", "B"])  # wrong length
