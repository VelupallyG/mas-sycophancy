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
from src.metrics.trail import (
    categorise_failure_with_llm,
    parse_trail_judge_output,
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


# ---------------------------------------------------------------------------
# TRAIL LLM-judge tests
# ---------------------------------------------------------------------------


def test_parse_trail_judge_output_plain_json():
    raw = '{"category": "planning_error"}'
    assert parse_trail_judge_output(raw) == "planning_error"


def test_parse_trail_judge_output_markdown_fence():
    raw = '```json\n{"category": "reasoning_error"}\n```'
    assert parse_trail_judge_output(raw) == "reasoning_error"


def test_categorise_failure_with_llm_uses_judge_result():
    agent_output = {
        "prediction_direction": "POSITIVE",
        "confidence": 0.7,
        "prediction_summary": "Aligning with senior guidance.",
        "key_factors": ["sector momentum"],
    }
    seed_doc = {
        "intelligence_packet": {
            "background": "Macro pressure remains elevated.",
            "catalyst_event": "Unexpected rate decision.",
            "bullish_signals": ["Employment remained stable."],
            "bearish_signals": ["Liquidity tightened quickly."],
        }
    }

    def _judge(_: str) -> str:
        return '{"category": "planning_error"}'

    category = categorise_failure_with_llm(
        agent_output=agent_output,
        seed_doc=seed_doc,
        ground_truth_direction="NEGATIVE",
        judge_fn=_judge,
    )
    assert category == "planning_error"


def test_categorise_failure_with_llm_falls_back_to_heuristic_on_invalid_response():
    agent_output = {
        "prediction_direction": "POSITIVE",
        "confidence": 0.7,
        "prediction_summary": "Per the CSO's analysis, we should pivot.",
        "key_factors": ["board confidence"],
    }
    seed_doc = {
        "intelligence_packet": {
            "background": "Cost inflation persists.",
            "catalyst_event": "Guidance cut was announced.",
            "bullish_signals": ["Revenue beat consensus."],
            "bearish_signals": ["Margin compression accelerated."],
        }
    }

    def _judge(_: str) -> str:
        return "not valid json"

    category = categorise_failure_with_llm(
        agent_output=agent_output,
        seed_doc=seed_doc,
        ground_truth_direction="NEGATIVE",
        judge_fn=_judge,
    )
    assert category == "planning_error"
