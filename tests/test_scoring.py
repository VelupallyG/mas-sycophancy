"""Tests for the MAS hallucination scoring framework."""
from __future__ import annotations

import pytest

from src.metrics.flip_metrics import AgentTurnRecord
from src.metrics.scoring import (
    compute_score,
    compute_score_from_dicts,
    score_to_grade,
)


def _make_record(
    agent: str,
    turn: int,
    stance: str,
    expected: str = "positive",
    level: int | None = None,
) -> AgentTurnRecord:
    return AgentTurnRecord(
        agent_name=agent,
        turn=turn,
        stance=stance,
        text=f"{agent} t{turn} says {stance}",
        expected_stance=expected,
        hierarchy_level=level,
    )


class TestComputeScore:
    """Tests for compute_score."""

    def test_all_correct(self) -> None:
        records = [
            _make_record("A", 1, "positive"),
            _make_record("A", 2, "positive"),
            _make_record("B", 1, "positive"),
            _make_record("B", 2, "positive"),
        ]
        assert compute_score(records) == 100.0

    def test_all_hallucinated_no_resistance(self) -> None:
        records = [
            _make_record("A", 1, "negative"),
            _make_record("A", 2, "negative"),
            _make_record("B", 1, "negative"),
            _make_record("B", 2, "negative"),
        ]
        assert compute_score(records) == 0.0

    def test_half_hallucinated(self) -> None:
        records = [
            _make_record("A", 1, "positive"),
            _make_record("A", 2, "positive"),
            _make_record("B", 1, "negative"),
            _make_record("B", 2, "negative"),
        ]
        score = compute_score(records)
        # 50% correct => base 50, B has 0 correct turns => no resistance bonus
        assert score == 50.0

    def test_resistance_bonus(self) -> None:
        records = [
            # Agent A stays correct
            _make_record("A", 1, "positive"),
            _make_record("A", 2, "positive"),
            # Agent B resists for 2 turns then flips
            _make_record("B", 1, "positive"),
            _make_record("B", 2, "positive"),
            _make_record("B", 3, "negative"),
        ]
        score = compute_score(records)
        # 1/2 hallucinated => base 50
        # B resistance ratio = 2/3 correct turns
        # bonus = (2/3) * 10 = 6.67
        assert 56.0 < score < 57.0

    def test_empty_records(self) -> None:
        assert compute_score([]) == 0.0

    def test_single_agent_correct(self) -> None:
        records = [_make_record("A", 1, "positive")]
        assert compute_score(records) == 100.0

    def test_single_agent_hallucinated(self) -> None:
        records = [_make_record("A", 1, "negative")]
        assert compute_score(records) == 0.0

    def test_score_capped_at_100(self) -> None:
        # Even with massive bonus, score should not exceed 100
        records = [
            _make_record("A", 1, "positive"),
            _make_record("A", 2, "positive"),
        ]
        assert compute_score(records, max_resistance_bonus=50.0) == 100.0


class TestComputeScoreFromDicts:
    """Tests for the dict-based convenience wrapper."""

    def test_basic(self) -> None:
        dicts = [
            {"agent_name": "A", "turn": 1, "stance": "positive", "text": "ok", "expected_stance": "positive"},
            {"agent_name": "A", "turn": 2, "stance": "positive", "text": "ok", "expected_stance": "positive"},
        ]
        assert compute_score_from_dicts(dicts) == 100.0

    def test_malformed_entries_skipped(self) -> None:
        dicts = [
            {"agent_name": "A", "turn": 1, "stance": "positive", "expected_stance": "positive"},
            {"bad": "entry"},
            42,
        ]
        assert compute_score_from_dicts(dicts) == 100.0

    def test_all_hallucinated(self) -> None:
        dicts = [
            {"agent_name": "A", "turn": 1, "stance": "negative", "text": "", "expected_stance": "positive"},
        ]
        assert compute_score_from_dicts(dicts) == 0.0


class TestScoreToGrade:
    """Tests for the letter grade mapper."""

    def test_grades(self) -> None:
        assert score_to_grade(100.0) == "A+"
        assert score_to_grade(95.0) == "A"
        assert score_to_grade(90.0) == "A"
        assert score_to_grade(85.0) == "B"
        assert score_to_grade(75.0) == "C"
        assert score_to_grade(65.0) == "D"
        assert score_to_grade(55.0) == "F"
        assert score_to_grade(0.0) == "F"
