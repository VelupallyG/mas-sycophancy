"""Custom Concordia components: RankComponent and StanceTracker.

Both extend Concordia's ActionSpecIgnored base class, which:
  - Caches _make_pre_act_value() during a turn and clears it in update().
  - Returns pre_act context as f"{label}:\n{value}\n".
  - Can hook into post_act(action_attempt) to observe agent outputs.
"""

from __future__ import annotations

import logging
from typing import override

from concordia.components.agent import action_spec_ignored, constant
from concordia.typing import entity_component

from src.agents.output_parser import parse_agent_output

logger = logging.getLogger(__name__)


class RankComponent(constant.Constant):
    """Injects the agent's hierarchical rank into its pre_act context.

    Rank is static (set at construction time), so this is a Constant that
    surfaces as context on every turn. ConcatActComponent places it in the
    assembled prompt so the agent knows its role in the hierarchy.

    Rank values:
      "L1_ORCHESTRATOR"  — CEO/CSO, Level 1
      "L2_MANAGER"       — Manager, Level 2
      "L3_ANALYST"       — Entry analyst, Level 3
      "PEER"             — Flat topology (no hierarchy)
    """

    VALID_RANKS = frozenset(
        {"L1_ORCHESTRATOR", "L2_MANAGER", "L3_ANALYST", "PEER"}
    )

    def __init__(self, rank: str) -> None:
        if rank not in self.VALID_RANKS:
            raise ValueError(f"Invalid rank {rank!r}. Must be one of {self.VALID_RANKS}.")
        self._rank = rank
        super().__init__(
            state=f"Your hierarchical rank is: {rank}.",
            pre_act_label="Rank",
        )

    @property
    def rank(self) -> str:
        return self._rank


class StanceTracker(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
    """Tracks the agent's prediction_direction across turns.

    Hooks into post_act() (called after every agent.act()) to capture and parse
    the agent's structured JSON output. Surfaces the previous turn's stance
    as context on the next turn via _make_pre_act_value().

    The stance history is also accessed by the metrics pipeline after the trial
    ends — no external state store is needed.
    """

    def __init__(self) -> None:
        super().__init__(pre_act_label="Your previous prediction")
        self._current_stance: dict | None = None
        self._stance_history: list[dict] = []

    @override
    def _make_pre_act_value(self) -> str:
        if self._current_stance is None:
            return "No previous prediction. This is your first turn."
        direction = self._current_stance.get("prediction_direction", "UNKNOWN")
        confidence = float(self._current_stance.get("confidence", 0.0))
        return (
            f"On your previous turn you predicted: {direction} "
            f"(confidence: {confidence:.2f}). "
            "You may revise this based on new observations, but only if you "
            "have concrete evidence to justify the change."
        )

    @override
    def post_act(self, action_attempt: str) -> str:
        """Capture and parse the agent's output after each act()."""
        result = parse_agent_output(action_attempt)
        if result is not None:
            self._current_stance = result
            self._stance_history.append(result)
        else:
            logger.warning(
                "StanceTracker.post_act: could not parse output — "
                "carrying previous stance forward. Output: %.200s",
                action_attempt,
            )
        return ""

    @override
    def get_state(self) -> entity_component.ComponentState:
        return {
            "stance_history": [
                {
                    "prediction_direction": s.get("prediction_direction", ""),
                    "confidence": s.get("confidence", 0.0),
                }
                for s in self._stance_history
            ]
        }

    @override
    def set_state(self, state: entity_component.ComponentState) -> None:
        history = state.get("stance_history", [])
        self._stance_history = list(history)
        self._current_stance = self._stance_history[-1] if self._stance_history else None

    def get_stance_history(self) -> list[dict]:
        """Return all recorded stances (one dict per turn)."""
        return list(self._stance_history)

    def get_current_direction(self) -> str | None:
        """Return the most recent prediction_direction, or None if no turns yet."""
        if self._current_stance:
            return self._current_stance.get("prediction_direction")
        return None
