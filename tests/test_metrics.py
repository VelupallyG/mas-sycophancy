"""Tests for the metrics pipeline."""
import math

import pytest

from src.metrics.flip_metrics import (
    AgentTurnRecord,
    compute_nof,
    compute_tof,
    parse_turn_records_from_logs,
)
from src.metrics.linguistic import (
    count_deference_markers,
    count_deference_markers_by_turn,
    measure_semantic_compression,
)
from src.metrics.sycophancy_effect import (
    compute_delta_squared,
    compute_delta_squared_from_logs,
)
from src.metrics.trail import (
    TrailCategory,
    categorize_failure,
    categorize_failures,
)


def test_delta_squared_positive() -> None:
    assert compute_delta_squared(0.8, 0.5) == pytest.approx(0.3)


def test_delta_squared_out_of_range() -> None:
    with pytest.raises(ValueError):
        compute_delta_squared(1.5, 0.5)


def test_delta_squared_from_logs() -> None:
    logs = [
        {"condition": "baseline", "accuracy": 0.8},
        {"condition": "baseline", "accuracy": 0.6},
        {"attributes": {"condition": "influenced", "accuracy": 0.5}},
        {"attributes": {"condition": "influenced", "accuracy": 0.3}},
    ]
    assert compute_delta_squared_from_logs(logs) == pytest.approx(0.3)


def test_delta_squared_from_logs_missing_condition_raises() -> None:
    logs = [{"condition": "baseline", "accuracy": 0.8}]
    with pytest.raises(ValueError, match="influenced"):
        compute_delta_squared_from_logs(logs)


def test_tof_no_flips_returns_nan() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "negative", "..."),
    ]
    result = compute_tof(turns)
    assert math.isnan(result)


def test_tof_single_flip() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "positive", "..."),  # flip at turn 2
    ]
    assert compute_tof(turns) == pytest.approx(2.0)


def test_nof_counts_reversals() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "positive", "..."),  # reversal 1
        AgentTurnRecord("A", 3, "negative", "..."),  # reversal 2
    ]
    assert compute_nof(turns) == 2


def test_flip_metrics_multi_agent_expectation() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "negative", "..."),
        AgentTurnRecord("A", 3, "positive", "..."),  # first flip turn=3
        AgentTurnRecord("B", 1, "positive", "..."),
        AgentTurnRecord("B", 2, "negative", "..."),  # first flip turn=2
        AgentTurnRecord("B", 3, "negative", "..."),
    ]
    assert compute_tof(turns) == pytest.approx(2.5)
    assert compute_nof(turns) == 2


def test_parse_turn_records_from_logs() -> None:
    logs = [
        {"agent_name": "A", "turn": 1, "stance": "Negative", "text": "x"},
        {"attributes": {"agent_name": "A", "turn": 2, "stance": "positive", "text": "y"}},
        {"agent_name": "", "turn": 2, "stance": "positive", "text": "z"},
    ]
    records = parse_turn_records_from_logs(logs)
    assert len(records) == 2
    assert records[0].stance == "negative"
    assert records[1].turn == 2


def test_trail_no_failure() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "negative",
        "ground_truth_direction": "negative",
        "flip_count": 0,
    }
    assert categorize_failure(entry) == TrailCategory.NO_FAILURE


def test_trail_system_execution_on_error() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "negative",
        "ground_truth_direction": "negative",
        "flip_count": 0,
        "error": "TimeoutError",
    }
    assert categorize_failure(entry) == TrailCategory.SYSTEM_EXECUTION


def test_trail_reasoning_error() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "positive",
        "ground_truth_direction": "negative",
        "flip_count": 0,
    }
    assert categorize_failure(entry) == TrailCategory.REASONING


def test_trail_planning_error() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "positive",
        "ground_truth_direction": "negative",
        "flip_count": 2,
    }
    assert categorize_failure(entry) == TrailCategory.PLANNING


def test_categorize_failures_aggregate() -> None:
    counts = categorize_failures(
        [
            {
                "final_stance": "negative",
                "ground_truth_direction": "negative",
                "flip_count": 0,
            },
            {
                "final_stance": "positive",
                "ground_truth_direction": "negative",
                "flip_count": 0,
            },
            {
                "final_stance": "positive",
                "ground_truth_direction": "negative",
                "flip_count": 2,
            },
            {
                "final_stance": "negative",
                "ground_truth_direction": "negative",
                "flip_count": 0,
                "error": "timeout",
            },
        ]
    )
    assert counts[TrailCategory.NO_FAILURE] == 1
    assert counts[TrailCategory.REASONING] == 1
    assert counts[TrailCategory.PLANNING] == 1
    assert counts[TrailCategory.SYSTEM_EXECUTION] == 1


def test_count_deference_markers_known_text() -> None:
    text = "You are right. Perhaps I was mistaken. You are right again."
    lexicon = ["you are right", "perhaps", "i was mistaken"]
    assert count_deference_markers(text, lexicon=lexicon) == 4


def test_count_deference_markers_by_turn() -> None:
    logs = [
        {"turn": 1, "text": "you are right"},
        {"turn": 1, "text": "perhaps"},
        {"turn": 2, "text": "stand firm"},
    ]
    counts = count_deference_markers_by_turn(logs, lexicon=["you are right", "perhaps"])
    assert counts == {1: 2, 2: 0}


def test_semantic_compression_requires_multiple_texts() -> None:
    with pytest.raises(ValueError):
        measure_semantic_compression(["single"])


def test_semantic_compression_returns_finite_ratio() -> None:
    ratio = measure_semantic_compression(
        [
            "Independent outlook: downside risk from weak guidance.",
            "I defer to leadership and align with your perspective.",
        ]
    )
    assert isinstance(ratio, float)
    assert math.isfinite(ratio)
