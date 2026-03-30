"""Tests for the Game Master and Simulation runner.

Tests cover:
- GameMasterConfig default values are correct
- SimulationResult fields are accessible
- Simulation builds real agents from prefabs and calls act()
- Stance extraction from natural language
- Turn records contain real LLM (mock) output
- Trace file is written to output directory after run
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.agents.analyst_prefab import AnalystPrefab
from src.agents.orchestrator_prefab import OrchestratorPrefab
from src.config import ExperimentConfig, TopologyConfig
from src.game_master.simulation import (
    GameMasterConfig,
    Simulation,
    SimulationResult,
    extract_stance,
)
from src.model import MockLanguageModel
from src.tasks.predictive_intel import PredictiveIntelTask


# ---------------------------------------------------------------------------
# extract_stance unit tests
# ---------------------------------------------------------------------------


class TestExtractStance:
    def test_negative_keywords(self):
        assert extract_stance("I expect a sharp drop in price") == "negative"

    def test_positive_keywords(self):
        assert extract_stance("Strong rally expected, buy signal") == "positive"

    def test_neutral_keywords(self):
        assert extract_stance("Market will remain flat and stable") == "neutral"

    def test_fallback_when_no_keywords(self):
        assert extract_stance("I have no opinion", fallback="neutral") == "neutral"
        assert extract_stance("I have no opinion", fallback="negative") == "negative"

    def test_mixed_resolves_to_majority(self):
        # "crash" and "drop" = 2 negative, "rally" = 1 positive
        assert extract_stance("crash after rally then drop") == "negative"

    def test_catastrophic_is_negative(self):
        text = "catastrophic crash, stock halted, sector contagion"
        assert extract_stance(text) == "negative"


# ---------------------------------------------------------------------------
# GameMasterConfig
# ---------------------------------------------------------------------------


class TestGameMasterConfig:
    def test_defaults(self):
        config = GameMasterConfig()
        assert config.enforce_approval_chain is False
        assert config.verbose is False
        assert config.log_dir == "data/"


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


class TestSimulationResult:
    def test_fields_accessible(self):
        result = SimulationResult(
            experiment_id="test_001",
            consensus_prediction="negative moderate",
            accuracy=0.8,
            trace_path="data/test_001.json",
        )
        assert result.experiment_id == "test_001"
        assert result.accuracy == pytest.approx(0.8)
        assert result.agent_turn_records == []


# ---------------------------------------------------------------------------
# Simulation integration tests with MockLanguageModel
# ---------------------------------------------------------------------------


class TestSimulationWithMock:
    """Run the full pipeline with a mock LLM to verify structure."""

    @pytest.fixture()
    def tmp_output(self, tmp_path):
        return str(tmp_path)

    @pytest.fixture()
    def mock_model(self):
        """Mock that returns bearish predictions (matching tech_earnings ground truth)."""
        return MockLanguageModel(
            responses=[
                "Based on the revenue miss, I predict a negative moderate "
                "reaction. The stock will decline 8-12% due to weak guidance "
                "and margin compression.",
            ]
        )

    def test_flat_simulation_produces_result(self, mock_model, tmp_output):
        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=2,
            topology=TopologyConfig(num_agents=3),
        )
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=False,
            log_dir=tmp_output,
        )
        prefabs = [
            AnalystPrefab(params={"name": f"Agent_{i}"})
            for i in range(3)
        ]
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        result = sim.run(topology_agents=prefabs, task=task)

        assert result.experiment_id == config.experiment_id
        assert isinstance(result.accuracy, float)
        assert 0.0 <= result.accuracy <= 1.0
        assert len(result.agent_turn_records) > 0
        assert result.metadata["topology"] == "flat"

    def test_flat_consensus_is_majority_synthesis(self, tmp_output):
        """Flat consensus should reflect majority direction, not last speaker."""
        # 2 agents say negative, 1 says positive → consensus should be negative
        call_idx = 0

        def _rotating_responses():
            nonlocal call_idx
            responses = [
                "I expect a negative moderate reaction with declining revenue.",
                "Strong rally expected, very positive outlook for growth.",
                "Bearish signal, negative impact due to margin compression.",
            ]
            resp = responses[call_idx % len(responses)]
            call_idx += 1
            return resp

        class RotatingMock(MockLanguageModel):
            def sample_text(self, prompt, **kwargs) -> str:
                return _rotating_responses()

        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=1,
            topology=TopologyConfig(num_agents=3),
        )
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=False,
            log_dir=tmp_output,
        )
        prefabs = [
            AnalystPrefab(params={"name": f"Agent_{i}"})
            for i in range(3)
        ]
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=RotatingMock())
        result = sim.run(topology_agents=prefabs, task=task)

        # Consensus should reflect the 2-to-1 negative majority,
        # NOT the last speaker's positive text
        consensus_stance = extract_stance(result.consensus_prediction)
        assert consensus_stance == "negative"
        # The positive agent's text should NOT be in the consensus
        assert "rally" not in result.consensus_prediction.lower()

    def test_flat_consensus_preserves_magnitude_keywords(self, mock_model, tmp_output):
        """Majority-aligned statements should be concatenated to preserve scoring keywords."""
        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=1,
            topology=TopologyConfig(num_agents=2),
        )
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=False,
            log_dir=tmp_output,
        )
        prefabs = [
            AnalystPrefab(params={"name": f"Agent_{i}"})
            for i in range(2)
        ]
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        result = sim.run(topology_agents=prefabs, task=task)

        # Mock always returns text with "negative moderate" + "decline" + "weak guidance"
        # Consensus should preserve these keywords for evaluate() scoring
        assert "negative" in result.consensus_prediction.lower() or \
               "decline" in result.consensus_prediction.lower()

    def test_flat_agents_all_act_each_turn(self, mock_model, tmp_output):
        """All agents should produce a response each turn in flat mode."""
        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=2,
            topology=TopologyConfig(num_agents=3),
        )
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=False,
            log_dir=tmp_output,
        )
        prefabs = [
            AnalystPrefab(params={"name": f"Agent_{i}"})
            for i in range(3)
        ]
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        result = sim.run(topology_agents=prefabs, task=task)

        # 3 agents * 2 turns = 6 records
        assert len(result.agent_turn_records) == 6

    def test_turn_records_contain_real_text(self, mock_model, tmp_output):
        """Turn records must contain the LLM's actual response, not fabricated text."""
        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=1,
            topology=TopologyConfig(num_agents=2),
        )
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=False,
            log_dir=tmp_output,
        )
        prefabs = [
            AnalystPrefab(params={"name": f"Agent_{i}"})
            for i in range(2)
        ]
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        result = sim.run(topology_agents=prefabs, task=task)

        for record in result.agent_turn_records:
            # Text should be from the mock model, not fabricated
            assert "revenue miss" in record["text"] or "decline" in record["text"]
            assert record["stance"] in {"positive", "negative", "neutral"}
            assert record["expected_stance"] == "negative"

    def test_hierarchical_simulation_produces_result(self, mock_model, tmp_output):
        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=2,
            topology=TopologyConfig(
                num_agents=4,
                level_counts={1: 1, 2: 2, 3: 0, 4: 0, 5: 1},
            ),
        )
        # Relax level count validation for minimal test
        from src.topologies.hierarchical import HierarchicalTopology
        from src.agents.director_prefab import DirectorPrefab

        levels = {
            1: [OrchestratorPrefab(params={"name": "CSO"})],
            2: [DirectorPrefab(params={"name": f"Dir_{i}"}) for i in range(2)],
            5: [AnalystPrefab(params={"name": "Analyst_1"})],
        }
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=True,
            log_dir=tmp_output,
        )
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        result = sim.run(topology_agents=levels, task=task)

        assert result.metadata["topology"] == "hierarchical"
        # Should have a consensus record from the orchestrator
        consensus_records = [
            r for r in result.agent_turn_records if r["agent_name"] == "CSO"
        ]
        assert len(consensus_records) > 0

    def test_trace_file_written(self, mock_model, tmp_output):
        config = ExperimentConfig(
            seed_doc="tech_earnings",
            max_turns=1,
            topology=TopologyConfig(num_agents=2),
        )
        gm_config = GameMasterConfig(
            experiment=config,
            enforce_approval_chain=False,
            log_dir=tmp_output,
        )
        prefabs = [
            AnalystPrefab(params={"name": f"Agent_{i}"})
            for i in range(2)
        ]
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        result = sim.run(topology_agents=prefabs, task=task)

        trace_path = Path(result.trace_path)
        assert trace_path.exists()
        with trace_path.open() as f:
            spans = json.load(f)
        assert isinstance(spans, list)
        assert len(spans) > 0

    def test_empty_topology_raises(self, mock_model, tmp_output):
        config = ExperimentConfig(seed_doc="tech_earnings", max_turns=1)
        gm_config = GameMasterConfig(
            experiment=config, log_dir=tmp_output,
        )
        task = PredictiveIntelTask()
        sim = Simulation(gm_config, model=mock_model)
        with pytest.raises(ValueError, match="non-empty"):
            sim.run(topology_agents=[], task=task)

    def test_wrong_task_type_raises(self, mock_model, tmp_output):
        config = ExperimentConfig(seed_doc="tech_earnings", max_turns=1)
        gm_config = GameMasterConfig(
            experiment=config, log_dir=tmp_output,
        )
        sim = Simulation(gm_config, model=mock_model)
        with pytest.raises(TypeError, match="PredictiveIntelTask"):
            sim.run(topology_agents=[AnalystPrefab()], task="not a task")
