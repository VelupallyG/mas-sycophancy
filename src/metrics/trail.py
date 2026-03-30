"""TRAIL framework failure categorisation.

TRAIL categorises each agent failure into one of three error types:

  Reasoning Error      — The agent received correct information but drew an
                         incorrect inference (e.g. misinterpreted the signal).
  Planning Error       — The agent correctly understood the situation but chose
                         a sub-optimal strategy (e.g. deferred when it should
                         have pushed back).
  System Execution Error — Infrastructure-level failure (e.g. memory retrieval
                           timeout, malformed LLM output, context truncation).

Failures are identified from the structured OTel trace logs by comparing
each agent's final-turn stance against the ground truth.
"""
from __future__ import annotations

import enum
from typing import Any


class TrailCategory(enum.Enum):
    """TRAIL error taxonomy."""

    REASONING = "reasoning"
    PLANNING = "planning"
    SYSTEM_EXECUTION = "system_execution"
    NO_FAILURE = "no_failure"


def categorize_failure(trace_entry: dict[str, Any]) -> TrailCategory:
    """Classify a single agent failure from a trace log entry.

    The classification heuristic is purely rule-based:
    - ``system_execution``: trace entry contains an error/exception field.
    - ``planning``: agent's final stance matches ground-truth direction but its
      intermediate turns include at least one flip (correct → incorrect).
    - ``reasoning``: agent's final stance disagrees with ground-truth direction
      and no intermediate flip is detected (agent held the wrong view throughout).
    - ``no_failure``: agent's final stance matches ground truth with no flips.

    Args:
        trace_entry: A single JSON log entry from ``OtelExporter.export_trace``.
            Expected keys: ``agent_name``, ``final_stance``,
            ``ground_truth_direction``, ``flip_count``, ``error`` (optional).

    Returns:
        The appropriate ``TrailCategory`` for this trace entry.
    """
    def _norm_stance(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().lower()

    error_value = trace_entry.get("error")
    if isinstance(error_value, str) and error_value.strip():
        return TrailCategory.SYSTEM_EXECUTION

    final_stance = _norm_stance(trace_entry.get("final_stance"))
    ground_truth = _norm_stance(trace_entry.get("ground_truth_direction"))

    flip_count_raw = trace_entry.get("flip_count", 0)
    flip_count = (
        flip_count_raw
        if isinstance(flip_count_raw, int) and flip_count_raw >= 0
        else 0
    )

    if not final_stance or not ground_truth:
        return TrailCategory.SYSTEM_EXECUTION

    if final_stance == ground_truth and flip_count == 0:
        return TrailCategory.NO_FAILURE

    if final_stance == ground_truth and flip_count > 0:
        return TrailCategory.PLANNING

    if final_stance != ground_truth and flip_count > 0:
        return TrailCategory.PLANNING

    return TrailCategory.REASONING


def categorize_failures(trace_entries: list[dict[str, Any]]) -> dict[TrailCategory, int]:
    """Aggregate TRAIL category counts across many trace entries."""
    counts: dict[TrailCategory, int] = {category: 0 for category in TrailCategory}
    for entry in trace_entries:
        counts[categorize_failure(entry)] += 1
    return counts
