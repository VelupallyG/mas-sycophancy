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
    raise NotImplementedError


def compute_nof(turns: list[AgentTurnRecord]) -> int:
    """Count total stance reversals across all agents and turns.

    A reversal is any consecutive pair of turns for the same agent where
    the stance changes.

    Args:
        turns: All ``AgentTurnRecord`` entries for one simulation run.

    Returns:
        Total number of stance reversals (≥ 0).
    """
    raise NotImplementedError
