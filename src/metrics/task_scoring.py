"""Task-specific scoring metrics.

Complements the generalized metrics (Δ², ToF, NoF, TRAIL) with per-seed-document
evaluation that assesses reasoning quality, not just directional accuracy.

Each seed document has a companion scoring guideline JSON in
src/tasks/scoring_guidelines/{seed_doc_name}.json containing:
  - critical_factors: data points a correct prediction should cite
  - trap_signals: misleading signals that weak predictions anchor on
  - confidence_range: expected confidence band for well-calibrated predictions
  - reasoning_quality_indicators: strong vs weak reasoning patterns

Usage:
    guidelines = load_scoring_guidelines("tech_earnings")
    scores = score_agent_output(agent_output, guidelines)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_GUIDELINES_DIR = Path(__file__).parent.parent / "tasks" / "scoring_guidelines"


@dataclass(frozen=True)
class TaskScoringGuidelines:
    """Loaded scoring guidelines for a single seed document."""

    seed_doc_name: str
    primary_driver: str
    critical_factors: list[str]
    supporting_factors: list[str]
    trap_signals: list[dict[str, str]]
    confidence_min: float
    confidence_max: float
    strong_indicators: list[str]
    weak_indicators: list[str]
    common_failure_modes: list[str]


@dataclass(frozen=True)
class TaskScore:
    """Result of scoring a single agent output against task-specific guidelines."""

    critical_factor_hits: int
    critical_factor_total: int
    critical_factor_ratio: float
    trap_signal_count: int
    confidence_calibrated: bool
    confidence_delta: float
    reasoning_strength_hits: int
    reasoning_weakness_hits: int


def load_scoring_guidelines(seed_doc_name: str) -> TaskScoringGuidelines | None:
    """Load task-specific scoring guidelines for a seed document.

    Args:
        seed_doc_name: Stem name matching the seed document
            (e.g., "tech_earnings", "policy_draft").

    Returns:
        TaskScoringGuidelines if found, None if no guidelines exist.
    """
    path = _GUIDELINES_DIR / f"{seed_doc_name}.json"
    if not path.exists():
        logger.debug("No scoring guidelines found for %s", seed_doc_name)
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    conf = data.get("confidence_range", {})
    return TaskScoringGuidelines(
        seed_doc_name=seed_doc_name,
        primary_driver=data.get("primary_driver", ""),
        critical_factors=data.get("critical_factors", []),
        supporting_factors=data.get("supporting_factors", []),
        trap_signals=data.get("trap_signals", []),
        confidence_min=conf.get("well_calibrated_min", 0.0),
        confidence_max=conf.get("well_calibrated_max", 1.0),
        strong_indicators=data.get("reasoning_quality_indicators", {}).get("strong", []),
        weak_indicators=data.get("reasoning_quality_indicators", {}).get("weak", []),
        common_failure_modes=data.get("common_failure_modes", []),
    )


def score_agent_output(
    agent_output: dict,
    guidelines: TaskScoringGuidelines,
) -> TaskScore:
    """Score a parsed agent output against task-specific guidelines.

    Evaluates three dimensions:
      1. Critical factor coverage — how many must-cite data points appear
      2. Trap signal exposure — how many misleading signals the agent cited
      3. Confidence calibration — whether confidence falls in the expected range

    All matching is case-insensitive substring matching against the agent's
    prediction_summary and key_factors fields.

    Args:
        agent_output: Parsed agent output dict with prediction_summary,
            key_factors, and confidence.
        guidelines: Loaded TaskScoringGuidelines for the relevant seed doc.

    Returns:
        TaskScore with the evaluation results.
    """
    summary = (agent_output.get("prediction_summary") or "").lower()
    key_factors = agent_output.get("key_factors", [])
    confidence = float(agent_output.get("confidence", 0.0))

    # Combine summary and key_factors into a single searchable text.
    combined = summary + " " + " ".join(
        f.lower() for f in key_factors
    )

    # 1. Critical factor coverage.
    critical_hits = sum(
        1 for factor in guidelines.critical_factors
        if _fuzzy_match(factor, combined)
    )
    critical_total = len(guidelines.critical_factors)
    critical_ratio = critical_hits / critical_total if critical_total > 0 else 0.0

    # 2. Trap signal exposure.
    trap_count = sum(
        1 for trap in guidelines.trap_signals
        if _fuzzy_match(trap["signal"], combined)
    )

    # 3. Confidence calibration.
    calibrated = guidelines.confidence_min <= confidence <= guidelines.confidence_max
    if confidence < guidelines.confidence_min:
        delta = guidelines.confidence_min - confidence
    elif confidence > guidelines.confidence_max:
        delta = confidence - guidelines.confidence_max
    else:
        delta = 0.0

    # 4. Reasoning quality pattern matching.
    strength_hits = sum(
        1 for indicator in guidelines.strong_indicators
        if _indicator_match(indicator, combined)
    )
    weakness_hits = sum(
        1 for indicator in guidelines.weak_indicators
        if _indicator_match(indicator, combined)
    )

    return TaskScore(
        critical_factor_hits=critical_hits,
        critical_factor_total=critical_total,
        critical_factor_ratio=critical_ratio,
        trap_signal_count=trap_count,
        confidence_calibrated=calibrated,
        confidence_delta=round(delta, 3),
        reasoning_strength_hits=strength_hits,
        reasoning_weakness_hits=weakness_hits,
    )


def score_trial_agents(
    agent_outputs: dict[str, list[dict]],
    guidelines: TaskScoringGuidelines,
) -> dict[str, list[TaskScore]]:
    """Score all agents across all turns in a trial.

    Args:
        agent_outputs: Mapping of agent_id to list of parsed outputs (one per turn).
        guidelines: Scoring guidelines for the seed document.

    Returns:
        Mapping of agent_id to list of TaskScore (one per turn).
    """
    results: dict[str, list[TaskScore]] = {}
    for agent_id, outputs in agent_outputs.items():
        results[agent_id] = [
            score_agent_output(out, guidelines) for out in outputs
        ]
    return results


def summarise_task_scores(
    scores: list[TaskScore],
) -> dict[str, float]:
    """Aggregate a list of TaskScores into summary statistics.

    Useful for summarising an agent's performance across turns or
    a population's performance in a trial.

    Returns:
        Dict with:
          mean_critical_factor_ratio: average coverage of critical factors
          mean_trap_signal_count: average number of trap signals cited
          calibration_rate: fraction of turns with well-calibrated confidence
          mean_confidence_delta: average deviation from calibrated range
          mean_strength_hits: average strong reasoning indicators
          mean_weakness_hits: average weak reasoning indicators
    """
    if not scores:
        return {
            "mean_critical_factor_ratio": 0.0,
            "mean_trap_signal_count": 0.0,
            "calibration_rate": 0.0,
            "mean_confidence_delta": 0.0,
            "mean_strength_hits": 0.0,
            "mean_weakness_hits": 0.0,
        }

    n = len(scores)
    return {
        "mean_critical_factor_ratio": sum(s.critical_factor_ratio for s in scores) / n,
        "mean_trap_signal_count": sum(s.trap_signal_count for s in scores) / n,
        "calibration_rate": sum(1 for s in scores if s.confidence_calibrated) / n,
        "mean_confidence_delta": sum(s.confidence_delta for s in scores) / n,
        "mean_strength_hits": sum(s.reasoning_strength_hits for s in scores) / n,
        "mean_weakness_hits": sum(s.reasoning_weakness_hits for s in scores) / n,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(reference: str, text: str) -> bool:
    """Check if the key terms from a reference phrase appear in the text.

    Splits the reference into significant words (length > 3) and requires
    at least 60% of them to appear in the text. This handles paraphrasing
    better than exact substring matching.
    """
    ref_words = [
        w.strip(".,;:\"'()[]{}").lower()
        for w in reference.split()
        if len(w.strip(".,;:\"'()[]{}")) > 3
    ]
    if not ref_words:
        return reference.lower() in text

    matches = sum(1 for w in ref_words if w in text)
    return matches / len(ref_words) >= 0.6


def _indicator_match(indicator: str, text: str) -> bool:
    """Check if a reasoning quality indicator is reflected in the text.

    Uses a looser threshold (40%) since indicators describe reasoning
    patterns rather than specific data points.
    """
    words = [
        w.strip(".,;:\"'()[]{}").lower()
        for w in indicator.split()
        if len(w.strip(".,;:\"'()[]{}")) > 3
    ]
    if not words:
        return False

    matches = sum(1 for w in words if w in text)
    return matches / len(words) >= 0.4
