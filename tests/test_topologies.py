"""Tests for topology spec classes."""

from __future__ import annotations

from src.topologies.flat import FlatTopologySpec, majority_vote
from src.topologies.hierarchical import (
    HierarchicalTopologySpec,
    N_L2_MANAGERS,
    N_L3_ANALYSTS,
    N_TOTAL,
)


def test_flat_topology_agent_count():
    spec = FlatTopologySpec()
    assert len(spec.agent_names) == 21


def test_flat_topology_injector_name():
    spec = FlatTopologySpec(injector_idx=5)
    assert spec.injector_name() == "peer_05"


def test_flat_topology_no_injector():
    spec = FlatTopologySpec()
    assert spec.injector_name() is None


def test_majority_vote_clear_winner():
    counts = {"NEGATIVE": 14, "POSITIVE": 5, "NEUTRAL": 2}
    assert majority_vote(counts) == "NEGATIVE"


def test_majority_vote_empty():
    assert majority_vote({}) == "NEUTRAL"


def test_hierarchical_total_agents():
    spec = HierarchicalTopologySpec()
    assert len(spec.all_agent_names()) == N_TOTAL == 21


def test_hierarchical_analyst_groups():
    spec = HierarchicalTopologySpec()
    assert len(spec.analyst_groups) == N_L2_MANAGERS
    total_analysts = sum(len(g) for g in spec.analyst_groups)
    assert total_analysts == N_L3_ANALYSTS


def test_hierarchical_manager_for_analyst():
    spec = HierarchicalTopologySpec()
    # analyst_00 is in manager_00's group.
    assert spec.manager_for_analyst("analyst_00") == "manager_00"
    # analyst_04 is in manager_01's group.
    assert spec.manager_for_analyst("analyst_04") == "manager_01"


def test_hierarchical_manager_for_unknown():
    spec = HierarchicalTopologySpec()
    assert spec.manager_for_analyst("nonexistent") is None
