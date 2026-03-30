"""Tests for the metrics pipeline.

Implement in Session 5.  Tests cover:
- compute_delta_squared returns correct value and rejects out-of-range input
- compute_tof returns NaN when no agent flips
- compute_tof returns correct first-flip turn
- compute_nof counts reversals correctly
- categorize_failure correctly classifies TRAIL categories
- count_deference_markers counts correctly on known text
"""
import math

import pytest

from src.metrics.flip_metrics import AgentTurnRecord, compute_nof, compute_tof
from src.metrics.sycophancy_effect import compute_delta_squared
from src.metrics.trail import TrailCategory, categorize_failure


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_delta_squared_positive() -> None:
    assert compute_delta_squared(0.8, 0.5) == pytest.approx(0.3)


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_delta_squared_out_of_range() -> None:
    with pytest.raises(ValueError):
        compute_delta_squared(1.5, 0.5)


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_tof_no_flips_returns_nan() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "negative", "..."),
    ]
    result = compute_tof(turns)
    assert math.isnan(result)


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_tof_single_flip() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "positive", "..."),  # flip at turn 2
    ]
    assert compute_tof(turns) == pytest.approx(2.0)


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_nof_counts_reversals() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "positive", "..."),  # reversal 1
        AgentTurnRecord("A", 3, "negative", "..."),  # reversal 2
    ]
    assert compute_nof(turns) == 2


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_trail_no_failure() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "negative",
        "ground_truth_direction": "negative",
        "flip_count": 0,
    }
    assert categorize_failure(entry) == TrailCategory.NO_FAILURE


@pytest.mark.skip(reason="Not implemented yet — Session 5")
def test_trail_system_execution_on_error() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "negative",
        "ground_truth_direction": "negative",
        "flip_count": 0,
        "error": "TimeoutError",
    }
    assert categorize_failure(entry) == TrailCategory.SYSTEM_EXECUTION
