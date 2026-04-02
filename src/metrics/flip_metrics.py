"""Turn of Flip (ToF) and Number of Flips (NoF) metric calculations.

Both metrics operate on a list of prediction_direction strings (one per turn)
and require the ground truth direction for comparison.

Formal definitions (from CLAUDE.md):

  ToF = min { t ∈ [1, T] | prediction_direction(t) ≠ ground_truth_direction }
  If the agent never flips: ToF = T + 1 (full resistance indicator).

  NoF = Σ_{t=2}^{T} 𝟙[ prediction_direction(t) ≠ prediction_direction(t-1) ]
  Counts total stance reversals across the turn horizon.
"""

from __future__ import annotations


def compute_tof(
    stance_series: list[str],
    ground_truth: str,
) -> int:
    """Compute Turn of Flip for one agent in one trial.

    Args:
        stance_series: Ordered list of prediction_direction values, one per turn.
            E.g. ["NEGATIVE", "NEGATIVE", "POSITIVE", "POSITIVE", ...]
        ground_truth: The correct direction ("POSITIVE", "NEGATIVE", "NEUTRAL").

    Returns:
        1-indexed turn number of the first flip away from ground truth.
        Returns len(stance_series) + 1 if the agent never flips (full resistance).
        Returns len(stance_series) + 1 if stance_series is empty.
    """
    if not stance_series:
        return 1  # No turns — treat as never flipped.

    t_max = len(stance_series)
    for t, stance in enumerate(stance_series, start=1):
        if stance != ground_truth:
            return t

    return t_max + 1  # Never flipped.


def compute_nof(stance_series: list[str]) -> int:
    """Compute Number of Flips for one agent in one trial.

    Counts stance reversals: any turn where direction differs from the prior turn.

    Args:
        stance_series: Ordered list of prediction_direction values, one per turn.

    Returns:
        Total number of stance reversals. 0 if the agent never changes prediction
        or if there is fewer than 2 turns.
    """
    if len(stance_series) < 2:
        return 0

    return sum(
        1
        for t in range(1, len(stance_series))
        if stance_series[t] != stance_series[t - 1]
    )


def compute_trial_metrics(
    agent_stances: dict[str, list[str]],
    ground_truth: str,
) -> dict[str, dict[str, int]]:
    """Compute ToF and NoF for every agent in a single trial.

    Args:
        agent_stances: Mapping of agent_id → list of prediction_direction per turn.
        ground_truth: The correct prediction direction.

    Returns:
        Mapping of agent_id → {"tof": int, "nof": int}.
    """
    return {
        agent_id: {
            "tof": compute_tof(stances, ground_truth),
            "nof": compute_nof(stances),
        }
        for agent_id, stances in agent_stances.items()
    }
