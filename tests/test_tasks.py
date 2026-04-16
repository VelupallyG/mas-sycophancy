"""Tests for seed document loading and schema compatibility."""

from __future__ import annotations

from src.tasks.predictive_intel import (
    PredictiveIntelligenceTask,
    extract_ground_truth_direction,
)


def test_loads_nested_ground_truth_seed():
    task = PredictiveIntelligenceTask(
        "geopolitics_sanctions_oil_supplyshock_2025_v1"
    )
    context = task.get_context()

    assert context.ground_truth == "POSITIVE"
    assert context.seed_doc_id == "geopolitical_oil_sanctions_2025_detailed"
    assert context.domain == "geopolitics"
    assert "ground_truth" not in context.formatted_prompt


def test_loads_current_finance_seed():
    task = PredictiveIntelligenceTask(
        "finance_earnings_alphabet_ai_capex_2026_v1"
    )
    context = task.get_context()

    assert context.ground_truth == "NEGATIVE"
    assert context.seed_doc_id == "tech_earnings_google_2026_detailed"
    assert context.domain == "finance"


def test_extract_ground_truth_direction_supports_legacy_and_nested():
    assert (
        extract_ground_truth_direction({"ground_truth_direction": "NEGATIVE"})
        == "NEGATIVE"
    )
    assert (
        extract_ground_truth_direction({"ground_truth": {"direction": "POSITIVE"}})
        == "POSITIVE"
    )
    assert extract_ground_truth_direction({}) is None
