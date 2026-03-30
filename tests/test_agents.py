"""Tests for Concordia Prefab agent instantiation and component wiring.

Each prefab's build() method is exercised with:
  - a NoLanguageModel (returns empty strings / choice 0 — no GCP call)
  - a minimal AssociativeMemoryBank backed by a fixed-vector embedder

Tests verify:
  - build() returns an EntityAgentWithLogging with the correct agent_name
  - HierarchicalRank component is present and has the expected rank integer
  - StanceTracker component is present
  - OrchestratorPrefab injects / omits ConfidentialBriefing correctly
  - ManagerPrefab rejects ranks outside {3, 4}
  - WhistleblowerPrefab accepts ranks 1–5 and rejects out-of-range values
  - WhistleblowerPrefab has the dissent-focused self-perception question
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import no_language_model

from src.agents.analyst_prefab import AnalystPrefab
from src.agents.components import (
    HIERARCHICAL_RANK_KEY,
    STANCE_TRACKER_KEY,
    HierarchicalRank,
    StanceTracker,
)
from src.agents.director_prefab import DirectorPrefab
from src.agents.manager_prefab import ManagerPrefab
from src.agents.orchestrator_prefab import OrchestratorPrefab
from src.agents.whistleblower_prefab import WhistleblowerPrefab

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMBED_DIM = 16


def _embedder(text: str) -> np.ndarray:
    """Fixed-vector embedder — avoids any real model calls."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    return rng.random(_EMBED_DIM).astype(np.float32)


@pytest.fixture()
def model():
    return no_language_model.NoLanguageModel()


@pytest.fixture()
def memory_bank():
    return basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _components(agent) -> dict:
    return dict(agent.get_all_context_components())


# ---------------------------------------------------------------------------
# AnalystPrefab
# ---------------------------------------------------------------------------

class TestAnalystPrefab:
    def test_build_returns_correct_name(self, model, memory_bank):
        prefab = AnalystPrefab(params={"name": "Alex"})
        agent = prefab.build(model, memory_bank)
        assert agent.name == "Alex"

    def test_hierarchical_rank_is_5(self, model, memory_bank):
        prefab = AnalystPrefab(params={"name": "Alex"})
        agent = prefab.build(model, memory_bank)
        comp = _components(agent)[HIERARCHICAL_RANK_KEY]
        assert isinstance(comp, HierarchicalRank)
        assert comp._rank == 5

    def test_stance_tracker_present(self, model, memory_bank):
        prefab = AnalystPrefab(params={"name": "Alex"})
        agent = prefab.build(model, memory_bank)
        comp = _components(agent)[STANCE_TRACKER_KEY]
        assert isinstance(comp, StanceTracker)
        assert comp._agent_name == "Alex"

    def test_persona_component_present(self, model, memory_bank):
        prefab = AnalystPrefab(params={"name": "Alex"})
        agent = prefab.build(model, memory_bank)
        assert "Persona" in _components(agent)

    def test_person_by_situation_present(self, model, memory_bank):
        """Independent-assessment reasoning component must be wired in."""
        prefab = AnalystPrefab(params={"name": "Alex"})
        agent = prefab.build(model, memory_bank)
        assert "PersonBySituation" in _components(agent)

    def test_goal_component_absent_when_not_provided(self, model, memory_bank):
        prefab = AnalystPrefab(params={"name": "Alex", "goal": ""})
        agent = prefab.build(model, memory_bank)
        assert "Goal" not in _components(agent)

    def test_goal_component_present_when_provided(self, model, memory_bank):
        prefab = AnalystPrefab(params={"name": "Alex", "goal": "Win."})
        agent = prefab.build(model, memory_bank)
        assert "Goal" in _components(agent)


# ---------------------------------------------------------------------------
# ManagerPrefab
# ---------------------------------------------------------------------------

class TestManagerPrefab:
    @pytest.mark.parametrize("rank", [3, 4])
    def test_build_valid_ranks(self, model, memory_bank, rank):
        prefab = ManagerPrefab(params={"name": "Morgan", "rank": rank})
        agent = prefab.build(model, memory_bank)
        assert agent.name == "Morgan"
        comp = _components(agent)[HIERARCHICAL_RANK_KEY]
        assert comp._rank == rank

    def test_stance_tracker_present(self, model, memory_bank):
        prefab = ManagerPrefab(params={"name": "Morgan", "rank": 4})
        agent = prefab.build(model, memory_bank)
        assert STANCE_TRACKER_KEY in _components(agent)

    @pytest.mark.parametrize("bad_rank", [1, 2, 5, 0, 6])
    def test_invalid_rank_raises(self, model, memory_bank, bad_rank):
        prefab = ManagerPrefab(params={"name": "Morgan", "rank": bad_rank})
        with pytest.raises(ValueError):
            prefab.build(model, memory_bank)


# ---------------------------------------------------------------------------
# DirectorPrefab
# ---------------------------------------------------------------------------

class TestDirectorPrefab:
    def test_build_returns_correct_name(self, model, memory_bank):
        prefab = DirectorPrefab(params={"name": "Dana"})
        agent = prefab.build(model, memory_bank)
        assert agent.name == "Dana"

    def test_hierarchical_rank_is_2(self, model, memory_bank):
        prefab = DirectorPrefab(params={"name": "Dana"})
        agent = prefab.build(model, memory_bank)
        comp = _components(agent)[HIERARCHICAL_RANK_KEY]
        assert isinstance(comp, HierarchicalRank)
        assert comp._rank == 2

    def test_stance_tracker_present(self, model, memory_bank):
        prefab = DirectorPrefab(params={"name": "Dana"})
        agent = prefab.build(model, memory_bank)
        assert STANCE_TRACKER_KEY in _components(agent)


# ---------------------------------------------------------------------------
# OrchestratorPrefab
# ---------------------------------------------------------------------------

class TestOrchestratorPrefab:
    def test_build_returns_correct_name(self, model, memory_bank):
        prefab = OrchestratorPrefab(params={"name": "Orion"})
        agent = prefab.build(model, memory_bank)
        assert agent.name == "Orion"

    def test_hierarchical_rank_is_1(self, model, memory_bank):
        prefab = OrchestratorPrefab(params={"name": "Orion"})
        agent = prefab.build(model, memory_bank)
        comp = _components(agent)[HIERARCHICAL_RANK_KEY]
        assert isinstance(comp, HierarchicalRank)
        assert comp._rank == 1

    def test_no_briefing_in_control_condition(self, model, memory_bank):
        prefab = OrchestratorPrefab(
            params={"name": "Orion", "hallucinated_premise": ""}
        )
        agent = prefab.build(model, memory_bank)
        assert "ConfidentialBriefing" not in _components(agent)

    def test_briefing_injected_in_experimental_condition(self, model, memory_bank):
        signal = "Stock will drop 20% on earnings miss."
        prefab = OrchestratorPrefab(
            params={"name": "Orion", "hallucinated_premise": signal}
        )
        agent = prefab.build(model, memory_bank)
        assert "ConfidentialBriefing" in _components(agent)

    def test_briefing_contains_hallucinated_claim(self, model, memory_bank):
        signal = "Regulatory approval denied — shares to plummet."
        prefab = OrchestratorPrefab(
            params={"name": "Orion", "hallucinated_premise": signal}
        )
        agent = prefab.build(model, memory_bank)
        briefing_comp = _components(agent)["ConfidentialBriefing"]
        assert signal in briefing_comp._state

    def test_stance_tracker_present(self, model, memory_bank):
        prefab = OrchestratorPrefab(params={"name": "Orion"})
        agent = prefab.build(model, memory_bank)
        assert STANCE_TRACKER_KEY in _components(agent)


# ---------------------------------------------------------------------------
# WhistleblowerPrefab
# ---------------------------------------------------------------------------

class TestWhistleblowerPrefab:
    @pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
    def test_build_valid_ranks(self, model, memory_bank, rank):
        prefab = WhistleblowerPrefab(params={"name": "Quinn", "rank": rank})
        agent = prefab.build(model, memory_bank)
        assert agent.name == "Quinn"
        comp = _components(agent)[HIERARCHICAL_RANK_KEY]
        assert comp._rank == rank

    def test_stance_tracker_present(self, model, memory_bank):
        prefab = WhistleblowerPrefab(params={"name": "Quinn", "rank": 5})
        agent = prefab.build(model, memory_bank)
        assert STANCE_TRACKER_KEY in _components(agent)

    @pytest.mark.parametrize("bad_rank", [0, 6, -1, 99])
    def test_invalid_rank_raises(self, model, memory_bank, bad_rank):
        prefab = WhistleblowerPrefab(params={"name": "Quinn", "rank": bad_rank})
        with pytest.raises(ValueError):
            prefab.build(model, memory_bank)

    def test_persona_contains_challenge_directive(self, model, memory_bank):
        """Persona text must contain the anti-deference mandate."""
        prefab = WhistleblowerPrefab(params={"name": "Quinn", "rank": 5})
        agent = prefab.build(model, memory_bank)
        persona_state = _components(agent)["Persona"]._state
        # The prompt file must contain language about challenging unsupported claims
        assert "challenge" in persona_state.lower() or "contradict" in persona_state.lower()

    def test_whistleblower_self_perception_is_dissent_focused(self, model, memory_bank):
        """SelfPerception question must reference truth/accuracy, not generic."""
        prefab = WhistleblowerPrefab(params={"name": "Quinn", "rank": 2})
        agent = prefab.build(model, memory_bank)
        self_comp = _components(agent)["SelfPerception"]
        # The question text is set on _question attribute
        question_text = getattr(self_comp, "_question", "")
        assert "accuracy" in question_text.lower() or "factual" in question_text.lower()

    def test_l5_and_l2_whistleblowers_differ_only_in_rank(self, model, memory_bank):
        """Both variants should build; rank component is the only difference."""
        l5 = WhistleblowerPrefab(params={"name": "W5", "rank": 5}).build(model, memory_bank)
        l2 = WhistleblowerPrefab(params={"name": "W2", "rank": 2}).build(model, memory_bank)
        assert _components(l5)[HIERARCHICAL_RANK_KEY]._rank == 5
        assert _components(l2)[HIERARCHICAL_RANK_KEY]._rank == 2
        # Both have the same set of component keys
        assert set(_components(l5).keys()) == set(_components(l2).keys())


# ---------------------------------------------------------------------------
# StanceTracker unit tests
# ---------------------------------------------------------------------------

class TestStanceTracker:
    def test_record_and_retrieve(self):
        tracker = StanceTracker(agent_name="TestAgent")
        tracker.record(0, "Bullish: +5%")
        tracker.record(1, "Bearish: -3%")
        history = tracker.history
        assert len(history) == 2
        assert history[0] == {"turn": 0, "agent": "TestAgent", "stance": "Bullish: +5%"}
        assert history[1]["turn"] == 1

    def test_pre_act_emits_no_text(self):
        tracker = StanceTracker(agent_name="TestAgent")
        # _make_pre_act_value is the ActionSpecIgnored contract
        assert tracker._make_pre_act_value() == ""

    def test_history_is_a_copy(self):
        tracker = StanceTracker(agent_name="TestAgent")
        tracker.record(0, "initial")
        h = tracker.history
        h.append({"turn": 99, "agent": "Attacker", "stance": "injected"})
        assert len(tracker.history) == 1  # original unaffected


# ---------------------------------------------------------------------------
# HierarchicalRank unit tests
# ---------------------------------------------------------------------------

class TestHierarchicalRank:
    @pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
    def test_valid_ranks(self, rank):
        comp = HierarchicalRank(rank=rank)
        assert comp._rank == rank

    @pytest.mark.parametrize("bad_rank", [0, 6, -1])
    def test_invalid_rank_raises(self, bad_rank):
        with pytest.raises(ValueError):
            HierarchicalRank(rank=bad_rank)

    def test_pre_act_value_contains_rank_description(self):
        comp = HierarchicalRank(rank=1)
        value = comp._make_pre_act_value()
        assert "Chief Strategy Officer" in value or "Level 1" in value
