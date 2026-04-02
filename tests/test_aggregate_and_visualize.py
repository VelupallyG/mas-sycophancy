"""Tests for the analysis pipeline: aggregate_results and visualize."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

from analysis.aggregate_results import aggregate
from analysis.visualize import (
    plot_delta_squared,
    plot_deference_trajectory,
    plot_tof_nof,
    plot_trail_breakdown,
)


# --- Fixtures ---

BASELINE_RESULT = {
    "experiment_id": "flat_tech_earnings_test",
    "condition": "baseline",
    "seed_doc": "tech_earnings",
    "accuracy": 0.8,
    "consensus_prediction": "positive moderate",
    "trace_path": "/tmp/trace.json",
    "agent_turn_records": [
        {"agent_name": "A", "turn": 1, "stance": "positive", "text": "A t1 positive", "expected_stance": "positive", "hierarchy_level": None},
        {"agent_name": "A", "turn": 2, "stance": "positive", "text": "A t2 positive", "expected_stance": "positive", "hierarchy_level": None},
        {"agent_name": "B", "turn": 1, "stance": "positive", "text": "B t1 positive", "expected_stance": "positive", "hierarchy_level": None},
        {"agent_name": "B", "turn": 2, "stance": "positive", "text": "B t2 positive", "expected_stance": "positive", "hierarchy_level": None},
    ],
    "metadata": {
        "seed_id": "tech_001",
        "seed_doc": "tech_earnings",
        "ground_truth_direction": "positive",
        "ground_truth_magnitude": "moderate",
        "topology": "flat",
        "max_turns": 10,
    },
}

INFLUENCED_RESULT = {
    "experiment_id": "hier_tech_earnings_test",
    "condition": "influenced",
    "seed_doc": "tech_earnings",
    "accuracy": 0.3,
    "consensus_prediction": "negative moderate",
    "trace_path": "/tmp/trace2.json",
    "agent_turn_records": [
        {"agent_name": "A", "turn": 1, "stance": "positive", "text": "A t1 positive", "expected_stance": "positive", "hierarchy_level": 5},
        {"agent_name": "A", "turn": 2, "stance": "negative", "text": "A t2 i defer to negative", "expected_stance": "positive", "hierarchy_level": 5},
        {"agent_name": "B", "turn": 1, "stance": "negative", "text": "B t1 negative", "expected_stance": "positive", "hierarchy_level": 4},
        {"agent_name": "B", "turn": 2, "stance": "negative", "text": "B t2 negative", "expected_stance": "positive", "hierarchy_level": 4},
    ],
    "metadata": {
        "seed_id": "tech_001",
        "seed_doc": "tech_earnings",
        "ground_truth_direction": "positive",
        "ground_truth_magnitude": "moderate",
        "topology": "hierarchical",
        "max_turns": 10,
    },
}


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Write sample result JSONs to a temp directory."""
    baseline_path = tmp_path / "flat_tech_earnings_test_result.json"
    influenced_path = tmp_path / "hier_tech_earnings_test_result.json"
    baseline_path.write_text(json.dumps(BASELINE_RESULT), encoding="utf-8")
    influenced_path.write_text(json.dumps(INFLUENCED_RESULT), encoding="utf-8")
    return tmp_path


# --- Aggregate Tests ---

class TestAggregate:
    def test_returns_dataframe(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_columns_present(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        expected_cols = {
            "experiment_id", "topology", "condition", "seed_doc",
            "accuracy", "score", "grade", "delta_squared",
            "mean_tof", "total_nof",
            "trail_reasoning_pct", "trail_planning_pct", "trail_system_pct",
            "mean_deference_count",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_delta_squared_computed(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        influenced = df[df["condition"] == "influenced"].iloc[0]
        # delta_squared = baseline_acc (0.8) - influenced_acc (0.3) = 0.5
        assert influenced["delta_squared"] == pytest.approx(0.5)

    def test_baseline_has_no_delta_squared(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        baseline = df[df["condition"] == "baseline"].iloc[0]
        assert baseline["delta_squared"] is None or pd.isna(baseline["delta_squared"])

    def test_score_and_grade(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        baseline = df[df["condition"] == "baseline"].iloc[0]
        # All agents correct => score 100, grade A+
        assert baseline["score"] == 100.0
        assert baseline["grade"] == "A+"

    def test_empty_dir(self, tmp_path: Path) -> None:
        df = aggregate(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_result.json"
        bad.write_text("not json", encoding="utf-8")
        df = aggregate(tmp_path)
        assert len(df) == 0


# --- Visualization Tests (smoke tests: just verify they don't crash) ---

class TestVisualize:
    def test_plot_delta_squared(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        output = data_dir / "figs_ds"
        output.mkdir()
        plot_delta_squared(df, output)
        assert (output / "delta_squared.png").exists()

    def test_plot_tof_nof(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        output = data_dir / "figs_tof"
        output.mkdir()
        plot_tof_nof(df, output)
        assert (output / "tof_nof.png").exists()

    def test_plot_trail_breakdown(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        output = data_dir / "figs_trail"
        output.mkdir()
        plot_trail_breakdown(df, output)
        assert (output / "trail_breakdown.png").exists()

    def test_plot_deference_trajectory(self, data_dir: Path) -> None:
        df = aggregate(data_dir)
        output = data_dir / "figs_def"
        output.mkdir()
        plot_deference_trajectory(df, output)
        assert (output / "deference_trajectory.png").exists()

    def test_empty_df_no_crash(self, data_dir: Path) -> None:
        df = pd.DataFrame()
        output = data_dir / "figs_empty"
        output.mkdir()
        # Should not raise
        plot_delta_squared(df, output)
        plot_tof_nof(df, output)
        plot_trail_breakdown(df, output)
        plot_deference_trajectory(df, output)
