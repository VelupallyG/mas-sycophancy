"""Tests for the SimulationRunner and end-to-end trial execution.

Uses MockModel to avoid API calls. These tests verify:
  - Trials produce valid JSONL output.
  - JSONL records have the required schema.
  - Agent count and turn count match configuration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from concordia.testing.mock_model import MockModel

from src.config import Condition, ExperimentConfig, SeedDocument
from src.game_master.simulation import SimulationRunner
from src.tasks.predictive_intel import PredictiveIntelligenceTask


MOCK_AGENT_OUTPUT = json.dumps({
    "prediction_direction": "NEGATIVE",
    "confidence": 0.80,
    "prediction_summary": "Mock prediction for testing.",
    "key_factors": ["mock factor 1", "mock factor 2"],
})


@pytest.fixture
def mock_model():
    return MockModel(response=MOCK_AGENT_OUTPUT)


@pytest.fixture
def task():
    return PredictiveIntelligenceTask("tech_earnings")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    return tmp_path


def _make_config(
    condition: Condition,
    output_dir: Path,
    n_turns: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig(
        condition=condition,
        seed_doc=SeedDocument.TECH_EARNINGS,
        n_trials=1,
        n_turns=n_turns,
        gcp_project="test-project",
        output_dir=output_dir,
    )


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestFlatTrial:
    def test_produces_jsonl(self, mock_model, task, tmp_output_dir):
        config = _make_config(Condition.FLAT_BASELINE, tmp_output_dir)
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_flat_trial(task=task, trial_id=0)
        assert path.exists()

    def test_record_count(self, mock_model, task, tmp_output_dir):
        n_turns = 2
        n_agents = 21
        config = _make_config(Condition.FLAT_BASELINE, tmp_output_dir, n_turns=n_turns)
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_flat_trial(task=task, trial_id=0)
        records = _read_jsonl(path)
        assert len(records) == n_agents * n_turns

    def test_record_schema(self, mock_model, task, tmp_output_dir):
        config = _make_config(Condition.FLAT_BASELINE, tmp_output_dir)
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_flat_trial(task=task, trial_id=0)
        records = _read_jsonl(path)
        required_keys = {
            "trial_id", "seed_doc", "condition", "turn",
            "agent_id", "level", "prediction_direction",
            "confidence", "prediction_summary", "key_factors",
            "parse_success",
        }
        for record in records:
            assert required_keys.issubset(record.keys())

    def test_direction_values_are_valid(self, mock_model, task, tmp_output_dir):
        config = _make_config(Condition.FLAT_BASELINE, tmp_output_dir)
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_flat_trial(task=task, trial_id=0)
        records = _read_jsonl(path)
        valid = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        for record in records:
            assert record["prediction_direction"] in valid


class TestHierarchicalTrial:
    def test_produces_jsonl(self, mock_model, task, tmp_output_dir):
        config = _make_config(
            Condition.HIERARCHICAL_HALLUCINATION, tmp_output_dir
        )
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_hierarchical_trial(task=task, trial_id=0)
        assert path.exists()

    def test_record_count(self, mock_model, task, tmp_output_dir):
        n_turns = 2
        n_agents = 21
        config = _make_config(
            Condition.HIERARCHICAL_HALLUCINATION, tmp_output_dir, n_turns=n_turns
        )
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_hierarchical_trial(task=task, trial_id=0)
        records = _read_jsonl(path)
        assert len(records) == n_agents * n_turns

    def test_levels_are_correct(self, mock_model, task, tmp_output_dir):
        config = _make_config(
            Condition.HIERARCHICAL_HALLUCINATION, tmp_output_dir
        )
        runner = SimulationRunner(model=mock_model, config=config)
        path = runner.run_hierarchical_trial(task=task, trial_id=0)
        records = _read_jsonl(path)

        by_agent = {}
        for r in records:
            by_agent[r["agent_id"]] = r["level"]

        assert by_agent["orchestrator"] == 1
        assert by_agent["manager_00"] == 2
        assert by_agent["analyst_00"] == 3
