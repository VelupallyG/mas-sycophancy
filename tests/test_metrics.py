"""Tests for the metrics pipeline."""
import math
from pathlib import Path

import pytest

from src.metrics.flip_metrics import (
    AgentTurnRecord,
    compute_nof,
    compute_nof_total,
    compute_tof,
    parse_turn_records_from_logs,
)
from src.metrics.linguistic import (
    count_deference_markers,
    count_deference_markers_by_category,
    count_deference_markers_by_turn,
    count_deference_markers_by_turn_and_category,
    measure_semantic_compression,
)
import src.metrics.linguistic as linguistic
from src.metrics.sycophancy_effect import (
    AgentDeltaSquared,
    compute_delta_squared,
    compute_delta_squared_by_level,
    compute_delta_squared_from_logs,
    compute_mean_delta_squared,
    compute_per_agent_delta_squared,
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


def test_delta_squared_from_logs_ignores_malformed_non_dict_entries() -> None:
    logs = [
        {"condition": "baseline", "accuracy": 0.8},
        "bad-entry",
        {"attributes": {"condition": "influenced", "accuracy": 0.3}},
    ]
    assert compute_delta_squared_from_logs(logs) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Per-agent delta-squared (ACL proposal Section 4.1)
# ---------------------------------------------------------------------------


def test_per_agent_delta_squared_basic() -> None:
    baseline = [
        {"agent_name": "A1", "accuracy": 0.9},
        {"agent_name": "A2", "accuracy": 0.8},
    ]
    influenced = [
        {"agent_name": "A1", "accuracy": 0.4, "hierarchy_level": 5},
        {"agent_name": "A2", "accuracy": 0.6, "hierarchy_level": 5},
    ]
    results = compute_per_agent_delta_squared(baseline, influenced)
    assert len(results) == 2
    a1 = next(r for r in results if r.agent_name == "A1")
    assert a1.delta_squared == pytest.approx(0.5)
    a2 = next(r for r in results if r.agent_name == "A2")
    assert a2.delta_squared == pytest.approx(0.2)


def test_per_agent_excludes_orchestrator_by_default() -> None:
    baseline = [
        {"agent_name": "CSO", "accuracy": 0.9},
        {"agent_name": "A1", "accuracy": 0.8},
    ]
    influenced = [
        {"agent_name": "CSO", "accuracy": 0.1, "hierarchy_level": 1},
        {"agent_name": "A1", "accuracy": 0.5, "hierarchy_level": 5},
    ]
    results = compute_per_agent_delta_squared(baseline, influenced)
    assert len(results) == 1
    assert results[0].agent_name == "A1"


def test_per_agent_includes_orchestrator_when_flag_off() -> None:
    baseline = [{"agent_name": "CSO", "accuracy": 0.9}]
    influenced = [{"agent_name": "CSO", "accuracy": 0.1, "hierarchy_level": 1}]
    results = compute_per_agent_delta_squared(
        baseline, influenced, exclude_orchestrator=False,
    )
    assert len(results) == 1


def test_per_agent_no_match_raises() -> None:
    baseline = [{"agent_name": "A1", "accuracy": 0.9}]
    influenced = [{"agent_name": "B1", "accuracy": 0.5, "hierarchy_level": 5}]
    with pytest.raises(ValueError, match="no agents"):
        compute_per_agent_delta_squared(baseline, influenced)


def test_mean_delta_squared() -> None:
    per_agent = [
        AgentDeltaSquared("A1", 0.9, 0.4, 0.5, 5),
        AgentDeltaSquared("A2", 0.8, 0.6, 0.2, 5),
    ]
    assert compute_mean_delta_squared(per_agent) == pytest.approx(0.35)


def test_delta_squared_by_level() -> None:
    per_agent = [
        AgentDeltaSquared("A1", 0.9, 0.4, 0.5, 5),
        AgentDeltaSquared("A2", 0.8, 0.6, 0.2, 5),
        AgentDeltaSquared("D1", 0.85, 0.7, 0.15, 2),
    ]
    by_level = compute_delta_squared_by_level(per_agent)
    assert by_level[5] == pytest.approx(0.35)
    assert by_level[2] == pytest.approx(0.15)


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
    assert compute_nof(turns) == pytest.approx(2.0)
    assert compute_nof_total(turns) == 2


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
    assert compute_nof(turns) == pytest.approx(1.0)
    assert compute_nof_total(turns) == 2


def test_nof_stratified_mean_by_hierarchy_level() -> None:
    turns = [
        AgentTurnRecord("L2_A", 1, "negative", "...", hierarchy_level=2),
        AgentTurnRecord("L2_A", 2, "positive", "...", hierarchy_level=2),
        AgentTurnRecord("L2_A", 3, "negative", "...", hierarchy_level=2),
        AgentTurnRecord("L2_B", 1, "negative", "...", hierarchy_level=2),
        AgentTurnRecord("L2_B", 2, "negative", "...", hierarchy_level=2),
        AgentTurnRecord("L5_A", 1, "negative", "...", hierarchy_level=5),
        AgentTurnRecord("L5_A", 2, "positive", "...", hierarchy_level=5),
    ]
    # Level 2 mean = (2 + 0) / 2 = 1.0; Level 5 mean = 1.0; stratified mean = 1.0
    assert compute_nof(turns) == pytest.approx(1.0)


def test_nof_falls_back_to_unstratified_mean_without_levels() -> None:
    turns = [
        AgentTurnRecord("A", 1, "negative", "..."),
        AgentTurnRecord("A", 2, "positive", "..."),
        AgentTurnRecord("B", 1, "negative", "..."),
        AgentTurnRecord("B", 2, "negative", "..."),
    ]
    # Per-agent reversals are [1, 0] => mean 0.5
    assert compute_nof(turns) == pytest.approx(0.5)


def test_parse_turn_records_from_logs() -> None:
    logs = [
        {
            "agent_name": "A",
            "turn": 1,
            "stance": "Negative",
            "text": "x",
            "expected_stance": "negative",
            "hierarchy_level": 5,
        },
        {
            "attributes": {
                "agent_name": "A",
                "turn": 2,
                "stance": "positive",
                "text": "y",
            }
        },
        {"agent_name": "", "turn": 2, "stance": "positive", "text": "z"},
    ]
    records = parse_turn_records_from_logs(logs)
    assert len(records) == 2
    assert records[0].stance == "negative"
    assert records[1].turn == 2
    assert records[0].expected_stance == "negative"
    assert records[0].hierarchy_level == 5


def test_parse_turn_records_from_logs_strict_raises() -> None:
    logs = [{"agent_name": "A", "turn": "1", "stance": "negative", "text": "x"}]
    with pytest.raises(ValueError, match="invalid turn"):
        parse_turn_records_from_logs(logs, strict=True)


def test_parse_turn_records_from_logs_non_dict_entry_strict_raises() -> None:
    with pytest.raises(ValueError, match="dictionary"):
        parse_turn_records_from_logs(["bad-entry"], strict=True)


def test_parse_turn_records_from_logs_non_dict_entry_non_strict_skips() -> None:
    records = parse_turn_records_from_logs([
        "bad-entry",
        {"agent_name": "A", "turn": 1, "stance": "negative", "text": "x"},
    ])
    assert len(records) == 1


def test_tof_uses_expected_stance_when_present() -> None:
    turns = [
        AgentTurnRecord("A", 1, "positive", "...", expected_stance="negative"),
        AgentTurnRecord("A", 2, "negative", "...", expected_stance="negative"),
    ]
    assert compute_tof(turns) == pytest.approx(1.0)


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
        "planning_signal": True,
    }
    assert categorize_failure(entry) == TrailCategory.PLANNING


def test_trail_wrong_final_without_planning_signal_is_reasoning() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "positive",
        "ground_truth_direction": "negative",
        "flip_count": 3,
    }
    assert categorize_failure(entry) == TrailCategory.REASONING


def test_trail_string_false_planning_signal_is_not_truthy() -> None:
    entry = {
        "agent_name": "A",
        "final_stance": "positive",
        "ground_truth_direction": "negative",
        "flip_count": 1,
        "planning_signal": "false",
    }
    assert categorize_failure(entry) == TrailCategory.REASONING


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
                "planning_signal": True,
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


def test_count_deference_markers_avoids_substring_false_positives() -> None:
    text = "perhapsing should not match, but perhaps should."
    assert count_deference_markers(text, lexicon=["perhaps"]) == 1


def test_count_deference_markers_by_turn() -> None:
    logs = [
        {"turn": 1, "text": "you are right"},
        {"turn": 1, "text": "perhaps"},
        {"turn": 2, "text": "stand firm"},
    ]
    counts = count_deference_markers_by_turn(logs, lexicon=["you are right", "perhaps"])
    assert counts == {1: 2, 2: 0}


def test_count_deference_markers_by_category() -> None:
    text = "You are right, perhaps I may be mistaken."
    categories = {
        "fawning": ["you are right"],
        "hedging": ["perhaps", "i may be mistaken"],
    }
    assert count_deference_markers_by_category(text, categories) == {
        "fawning": 1,
        "hedging": 2,
    }


def test_count_deference_markers_by_turn_and_category() -> None:
    logs = [
        {"turn": 1, "text": "you are right"},
        {"turn": 1, "text": "perhaps"},
        {"turn": 2, "text": "i may be mistaken"},
    ]
    categories = {
        "fawning": ["you are right"],
        "hedging": ["perhaps", "i may be mistaken"],
    }
    assert count_deference_markers_by_turn_and_category(
        logs,
        lexicon_by_category=categories,
    ) == {
        1: {"fawning": 1, "hedging": 1},
        2: {"fawning": 0, "hedging": 1},
    }


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


def test_count_deference_markers_invalid_lexicon_json_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_lexicon = tmp_path / "bad_lexicon.json"
    bad_lexicon.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(linguistic, "_DEFAULT_LEXICON_PATH", bad_lexicon)

    with pytest.raises(ValueError, match="invalid JSON"):
        count_deference_markers("you are right")


def test_count_deference_markers_lexicon_read_error_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "blocked_lexicon.json"
    blocked.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(linguistic, "_DEFAULT_LEXICON_PATH", blocked)

    original_open = Path.open

    def _raise_open(self: Path, *args: object, **kwargs: object):
        if self == blocked:
            raise OSError("permission denied")
        return original_open(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(Path, "open", _raise_open)
        with pytest.raises(RuntimeError, match="failed to read deference lexicon"):
            count_deference_markers("you are right")
