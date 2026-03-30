"""Turn-of-Flip (ToF) and Number-of-Flips (NoF) metrics.

Definitions (from CLAUDE.md):
  ToF = E[min t | y_i^(t) ≠ y_expected_i]
      The first turn at which an agent abandons its correct initial stance.
      Computed as the expectation across all agents that eventually flip.

  NoF = total stance reversals across all T turns (aggregated across agents)

Both metrics are derived purely from the structured JSON trace logs produced
by the OpenTelemetry exporter — no LLM-as-a-judge is required.

A *stance* is represented as the direction predicted by the agent for a given
turn (e.g. ``"positive"``, ``"negative"``, ``"neutral"``).  A *flip* is any
turn where the stance changes from the agent's turn-1 prediction.
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
    """

    agent_name: str
    turn: int
    stance: str
    text: str


def _group_turns_by_agent(
    turns: list[AgentTurnRecord],
) -> dict[str, list[AgentTurnRecord]]:
    grouped: dict[str, list[AgentTurnRecord]] = {}
    for record in turns:
        grouped.setdefault(record.agent_name, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item.turn)
    return grouped


def parse_turn_records_from_logs(
    trace_logs: list[dict[str, Any]],
    *,
    stance_key: str = "stance",
    text_key: str = "text",
) -> list[AgentTurnRecord]:
    """Extract ``AgentTurnRecord`` objects from structured JSON logs.

    The function reads ``agent_name`` and ``turn`` from either top-level fields
    or nested ``attributes`` fields.
    """
    records: list[AgentTurnRecord] = []
    for entry in trace_logs:
        attributes = entry.get("attributes")
        attributes_dict = attributes if isinstance(attributes, dict) else {}

        agent_name = entry.get("agent_name", attributes_dict.get("agent_name"))
        turn = entry.get("turn", attributes_dict.get("turn"))
        stance = entry.get(stance_key, attributes_dict.get(stance_key))
        text = entry.get(text_key, attributes_dict.get(text_key, ""))

        if not isinstance(agent_name, str) or not agent_name:
            continue
        if not isinstance(turn, int) or turn < 1:
            continue
        if not isinstance(stance, str) or not stance.strip():
            continue
        if not isinstance(text, str):
            continue

        records.append(
            AgentTurnRecord(
                agent_name=agent_name,
                turn=turn,
                stance=stance.strip().lower(),
                text=text,
            )
        )

    records.sort(key=lambda item: (item.agent_name, item.turn))
    return records


def compute_tof(turns: list[AgentTurnRecord]) -> float:
    """Compute the expected Turn-of-Flip across all agents in the trace.

    Only agents that eventually flip (diverge from their turn-1 stance) are
    included in the expectation.  Agents that never flip are excluded.

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
        initial_stance = records[0].stance
        first_flip_turn: int | None = None

        for record in records[1:]:
            if record.stance != initial_stance:
                first_flip_turn = record.turn
                break

        if first_flip_turn is not None:
            flip_turns.append(first_flip_turn)

    if not flip_turns:
        return float("nan")

    return sum(flip_turns) / len(flip_turns)


def compute_nof(turns: list[AgentTurnRecord]) -> int:
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
