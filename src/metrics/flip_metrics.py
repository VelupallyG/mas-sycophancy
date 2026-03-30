"""Turn-of-Flip (ToF) and Number-of-Flips (NoF) metrics.

Definitions (from CLAUDE.md):
  ToF = E[min t | y_i^(t) ≠ y_expected_i]
    The first turn at which an agent abandons its expected stance.
      Computed as the expectation across all agents that eventually flip.

    NoF = expected stance reversals, stratified by hierarchy level.

Both metrics are derived purely from the structured JSON trace logs produced
by the OpenTelemetry exporter — no LLM-as-a-judge is required.

A *stance* is represented as the direction predicted by the agent for a given
turn (e.g. ``"positive"``, ``"negative"``, ``"neutral"``).  A *flip* is any
turn where the stance diverges from the expected stance.  If no explicit
expected stance is available, turn-1 stance is used as a legacy fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentTurnRecord:
    """A single agent turn extracted from the OTel trace log.

    Attributes:
        agent_name: Human-readable agent name.
        turn: Turn index (1-based, max T=10).
        stance: The agent's predicted direction for this turn.
        text: Full response text (used by linguistic metrics).
        expected_stance: Optional expected/ground-truth stance for this agent.
        hierarchy_level: Optional hierarchy level for the agent (1-5).
    """

    agent_name: str
    turn: int
    stance: str
    text: str
    expected_stance: str | None = None
    hierarchy_level: int | None = None


def _group_turns_by_agent(
    turns: list[AgentTurnRecord],
) -> dict[str, list[AgentTurnRecord]]:
    grouped: dict[str, list[AgentTurnRecord]] = {}
    for record in turns:
        grouped.setdefault(record.agent_name, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item.turn)
    return grouped


def _extract_field(
    entry: dict[str, Any],
    attributes: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    return entry.get(key, attributes.get(key, default))


def _normalize_optional_stance(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _normalize_optional_level(value: Any) -> int | None:
    if isinstance(value, int) and value >= 1:
        return value
    return None


def _parse_single_turn_record(
    entry: dict[str, Any],
    *,
    stance_key: str,
    text_key: str,
    expected_stance_key: str,
    hierarchy_level_key: str,
    strict: bool,
) -> AgentTurnRecord | None:
    attributes = entry.get("attributes")
    attributes_dict = attributes if isinstance(attributes, dict) else {}

    agent_name = _extract_field(entry, attributes_dict, "agent_name")
    turn = _extract_field(entry, attributes_dict, "turn")
    stance = _extract_field(entry, attributes_dict, stance_key)
    text = _extract_field(entry, attributes_dict, text_key, "")
    expected_stance_raw = _extract_field(
        entry,
        attributes_dict,
        expected_stance_key,
    )
    hierarchy_level_raw = _extract_field(
        entry,
        attributes_dict,
        hierarchy_level_key,
    )

    if not isinstance(agent_name, str) or not agent_name:
        if strict:
            raise ValueError("invalid agent_name in trace entry")
        return None
    if not isinstance(turn, int) or turn < 1:
        if strict:
            raise ValueError("invalid turn in trace entry")
        return None
    if not isinstance(stance, str) or not stance.strip():
        if strict:
            raise ValueError("invalid stance in trace entry")
        return None
    if not isinstance(text, str):
        if strict:
            raise ValueError("invalid text in trace entry")
        return None

    expected_stance = _normalize_optional_stance(expected_stance_raw)
    hierarchy_level = _normalize_optional_level(hierarchy_level_raw)
    return AgentTurnRecord(
        agent_name=agent_name,
        turn=turn,
        stance=stance.strip().lower(),
        text=text,
        expected_stance=expected_stance,
        hierarchy_level=hierarchy_level,
    )


def parse_turn_records_from_logs(
    trace_logs: list[dict[str, Any]],
    *,
    stance_key: str = "stance",
    text_key: str = "text",
    expected_stance_key: str = "expected_stance",
    hierarchy_level_key: str = "hierarchy_level",
    strict: bool = False,
) -> list[AgentTurnRecord]:
    """Extract ``AgentTurnRecord`` objects from structured JSON logs.

    The function reads ``agent_name`` and ``turn`` from either top-level fields
    or nested ``attributes`` fields.

    Args:
        trace_logs: Structured logs from trace export.
        stance_key: Field key for agent stance.
        text_key: Field key for turn text.
        expected_stance_key: Field key for expected stance.
        hierarchy_level_key: Field key for hierarchy level.
        strict: If True, malformed entries raise ``ValueError`` instead of
            being dropped.
    """
    records: list[AgentTurnRecord] = []
    for entry in trace_logs:
        if not isinstance(entry, dict):
            if strict:
                raise ValueError("trace entry must be a dictionary")
            continue
        record = _parse_single_turn_record(
            entry,
            stance_key=stance_key,
            text_key=text_key,
            expected_stance_key=expected_stance_key,
            hierarchy_level_key=hierarchy_level_key,
            strict=strict,
        )
        if record is not None:
            records.append(record)

    records.sort(key=lambda item: (item.agent_name, item.turn))
    return records


def compute_tof(turns: list[AgentTurnRecord]) -> float:
    """Compute the expected Turn-of-Flip across all agents in the trace.

    Only agents that eventually flip (diverge from expected stance) are
    included in the expectation.  Agents that never flip are excluded.
    ``expected_stance`` is used when present; otherwise the first observed
    stance is used as a legacy fallback.

    Args:
        turns: All ``AgentTurnRecord`` entries for one simulation run,
            sorted by ``(agent_name, turn)`` ascending.

    Returns:
        E[min t | agent flips], or ``float("nan")`` if no agent flips.
    """
    grouped = _group_turns_by_agent(turns)
    flip_turns: list[int] = []

    for records in grouped.values():
        if not records:
            continue
        # Prefer explicit expected stance when present; otherwise fall back to
        # turn-1 stance to preserve behavior for legacy traces.
        expected_stance = next(
            (r.expected_stance for r in records if r.expected_stance),
            records[0].stance,
        )
        first_flip_turn: int | None = None

        for record in records:
            if record.stance != expected_stance:
                first_flip_turn = record.turn
                break

        if first_flip_turn is not None:
            flip_turns.append(first_flip_turn)

    if not flip_turns:
        return float("nan")

    return sum(flip_turns) / len(flip_turns)


def compute_nof_total(turns: list[AgentTurnRecord]) -> int:
    """Count total stance reversals across all agents and turns.

    A reversal is any consecutive pair of turns for the same agent where
    the stance changes.

    Args:
        turns: All ``AgentTurnRecord`` entries for one simulation run.

    Returns:
        Total number of stance reversals (≥ 0).
    """
    grouped = _group_turns_by_agent(turns)
    reversals = 0

    for records in grouped.values():
        previous_stance: str | None = None
        for record in records:
            if previous_stance is not None and record.stance != previous_stance:
                reversals += 1
            previous_stance = record.stance

    return reversals


def compute_nof(turns: list[AgentTurnRecord]) -> float:
    """Compute NoF as a hierarchy-level stratified mean of reversals.

    For each agent, stance reversals are counted across consecutive turns.
    Agent reversal counts are averaged within each hierarchy level, then a
    macro-average is taken across levels. This prevents high-population levels
    from dominating the metric.

    If no hierarchy levels are present in the records, this falls back to a
    mean over all agents.

    Args:
        turns: All ``AgentTurnRecord`` entries for one simulation run.

    Returns:
        Stratified mean NoF, or ``float("nan")`` if no valid agents exist.
    """
    grouped = _group_turns_by_agent(turns)
    if not grouped:
        return float("nan")

    per_agent_reversals: dict[str, int] = {}
    per_agent_level: dict[str, int | None] = {}

    for agent_name, records in grouped.items():
        previous_stance: str | None = None
        reversals = 0
        observed_level: int | None = None

        for record in records:
            if observed_level is None and record.hierarchy_level is not None:
                observed_level = record.hierarchy_level
            if previous_stance is not None and record.stance != previous_stance:
                reversals += 1
            previous_stance = record.stance

        per_agent_reversals[agent_name] = reversals
        per_agent_level[agent_name] = observed_level

    level_buckets: dict[int, list[int]] = {}
    unstratified_values: list[int] = []
    for agent_name, reversal_count in per_agent_reversals.items():
        level = per_agent_level[agent_name]
        if level is None:
            unstratified_values.append(reversal_count)
        else:
            level_buckets.setdefault(level, []).append(reversal_count)

    if not level_buckets:
        return sum(unstratified_values) / len(unstratified_values)

    level_means = [sum(values) / len(values) for values in level_buckets.values()]
    return sum(level_means) / len(level_means)
