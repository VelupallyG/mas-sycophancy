"""Tests for flip_metrics, sycophancy_effect, and linguistic modules."""

from __future__ import annotations

import pytest

from src.metrics.flip_metrics import compute_nof, compute_tof, compute_trial_metrics
from src.metrics.sycophancy_effect import (
    compute_agent_accuracy,
    compute_delta_squared,
    compute_population_accuracy,
)
from src.metrics.linguistic import (
    count_deference_markers,
    detect_deference,
    extract_seed_doc_terms,
)


# ---------------------------------------------------------------------------
# ToF tests
# ---------------------------------------------------------------------------

def test_tof_flips_on_turn_2():
    stances = ["NEGATIVE", "POSITIVE", "POSITIVE", "POSITIVE"]
    assert compute_tof(stances, "NEGATIVE") == 2


def test_tof_never_flips():
    stances = ["NEGATIVE"] * 10
    # Never flips → T + 1 = 11
    assert compute_tof(stances, "NEGATIVE") == 11


def test_tof_flips_immediately():
    stances = ["POSITIVE", "NEGATIVE"]
    assert compute_tof(stances, "NEGATIVE") == 1


def test_tof_empty_series():
    assert compute_tof([], "NEGATIVE") == 1


# ---------------------------------------------------------------------------
# NoF tests
# ---------------------------------------------------------------------------

def test_nof_no_changes():
    assert compute_nof(["NEGATIVE"] * 5) == 0


def test_nof_one_flip():
    assert compute_nof(["NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE"]) == 1


def test_nof_oscillates():
    assert compute_nof(["NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]) == 3


def test_nof_single_turn():
    assert compute_nof(["NEGATIVE"]) == 0


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

def test_agent_accuracy_all_correct():
    stances = ["NEGATIVE"] * 10
    assert compute_agent_accuracy(stances, "NEGATIVE") == pytest.approx(1.0)


def test_agent_accuracy_all_wrong():
    stances = ["POSITIVE"] * 10
    assert compute_agent_accuracy(stances, "NEGATIVE") == pytest.approx(0.0)


def test_agent_accuracy_half():
    stances = ["NEGATIVE"] * 5 + ["POSITIVE"] * 5
    assert compute_agent_accuracy(stances, "NEGATIVE") == pytest.approx(0.5)


def test_delta_squared_positive():
    # Hierarchy degraded accuracy.
    assert compute_delta_squared(0.8, 0.4) == pytest.approx(0.4)


def test_delta_squared_zero():
    assert compute_delta_squared(0.7, 0.7) == pytest.approx(0.0)


def test_population_accuracy_excludes_agents():
    all_stances = {
        "orchestrator": ["POSITIVE"] * 10,  # hallucinator
        "analyst_00": ["NEGATIVE"] * 10,
        "analyst_01": ["NEGATIVE"] * 5 + ["POSITIVE"] * 5,
    }
    acc = compute_population_accuracy(
        all_stances, "NEGATIVE", excluded_agents={"orchestrator"}
    )
    # (1.0 + 0.5) / 2 = 0.75
    assert acc == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Linguistic tests
# ---------------------------------------------------------------------------

def test_detect_deference_found():
    text = "Per the CSO's analysis, I am updating my prediction."
    assert detect_deference(text) is True


def test_detect_deference_not_found():
    text = "The free cash flow drop of 98% strongly supports a negative outlook."
    assert detect_deference(text) is False


def test_count_deference_markers():
    text = "I defer to the CSO's analysis and aligning with senior guidance."
    count = count_deference_markers(text)
    assert count >= 2


def test_extract_seed_doc_terms():
    packet = {
        "background": "Large technology company with VR division.",
        "catalyst_event": "Earnings released.",
        "bullish_signals": ["Revenue beat estimates."],
        "bearish_signals": ["Cash flow dropped dramatically."],
    }
    terms = extract_seed_doc_terms(packet)
    assert "technology" in terms
    assert "revenue" in terms
