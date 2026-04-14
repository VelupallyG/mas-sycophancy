"""Predictive Intelligence Task loader.

Loads a JSON seed document, formats the intelligence packet into an agent
prompt, and provides the ground truth direction for evaluation.

Current seed schema uses: "ground_truth": {"direction": ..., "magnitude": ..., "actual_price_change_pct": ...}

The legacy flat "ground_truth_direction" key is still accepted by
extract_ground_truth_direction() for backwards compatibility, but no
shipped seed documents use it.

Ground-truth fields are NEVER included in the formatted prompt sent to agents.
They are returned separately via TaskContext for evaluation only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_SEED_DOCUMENT_DIR = Path(__file__).parent / "seed_documents"


def extract_ground_truth_direction(payload: dict) -> str | None:
    """Return ground-truth direction from either supported seed schema."""
    direction = payload.get("ground_truth_direction")
    if isinstance(direction, str):
        return direction

    gt_obj = payload.get("ground_truth")
    if isinstance(gt_obj, dict):
        nested = gt_obj.get("direction")
        if isinstance(nested, str):
            return nested

    return None


@dataclass(frozen=True)
class TaskContext:
    """Immutable context object produced by PredictiveIntelligenceTask.get_context()."""

    ground_truth: str
    """The true market direction: "POSITIVE", "NEGATIVE", or "NEUTRAL"."""

    formatted_prompt: str
    """The intelligence packet formatted for agent observation. Does NOT
    contain ground truth."""

    seed_doc_id: str
    """Metadata ID from the seed document (e.g. "tech_earnings_google_2026_detailed")."""

    domain: str
    """Domain of the seed document (e.g. "finance", "geopolitics")."""


class PredictiveIntelligenceTask:
    """Loads and formats a seed document for the predictive intelligence task."""

    def __init__(self, seed_file_name: str) -> None:
        """Load a seed document JSON file.

        Args:
            seed_file_name: Stem of the JSON file (e.g., "finance_earnings_alphabet_ai_capex_2026_v1").
                The file must exist at src/tasks/seed_documents/{name}.json.

        Raises:
            FileNotFoundError: If the seed document does not exist.
            ValueError: If the seed document is missing required fields.
        """
        path = _SEED_DOCUMENT_DIR / f"{seed_file_name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Seed document not found: {path}. "
                "Check that the file exists under src/tasks/seed_documents/."
            )
        with path.open("r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._validate()

    def _validate(self) -> None:
        required = {"metadata", "task_prompt", "intelligence_packet"}
        missing = required - self._data.keys()
        if missing:
            raise ValueError(f"Seed document missing required fields: {missing}")

        valid_directions = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        direction = extract_ground_truth_direction(self._data)
        if direction not in valid_directions:
            raise ValueError(
                f"ground truth direction must be one of {valid_directions}, "
                f"got {direction!r}."
            )

    def get_context(self) -> TaskContext:
        """Format the intelligence packet for agent observation.

        Returns a TaskContext where formatted_prompt is safe to pass to agents
        (no ground truth leakage) and ground_truth is available for evaluation.
        """
        packet = self._data["intelligence_packet"]

        lines: list[str] = [
            f"TASK: {self._data['task_prompt']}",
            "",
            f"BACKGROUND: {packet['background']}",
            f"EVENT: {packet['catalyst_event']}",
            "",
            "POSITIVE SIGNALS:",
        ]
        for sig in packet.get("bullish_signals", []):
            lines.append(f"  - {sig}")

        lines.append("")
        lines.append("NEGATIVE SIGNALS:")
        for sig in packet.get("bearish_signals", []):
            lines.append(f"  - {sig}")

        formatted_prompt = "\n".join(lines)

        metadata = self._data.get("metadata", {})
        direction = extract_ground_truth_direction(self._data)
        if direction is None:
            raise ValueError("Seed document missing ground truth direction.")
        return TaskContext(
            ground_truth=direction,
            formatted_prompt=formatted_prompt,
            seed_doc_id=metadata.get("id", "unknown"),
            domain=metadata.get("domain", "unknown"),
        )

    def get_ground_truth(self) -> str:
        """Return the ground truth direction (POSITIVE/NEGATIVE/NEUTRAL)."""
        direction = extract_ground_truth_direction(self._data)
        if direction is None:
            raise ValueError("Seed document missing ground truth direction.")
        return direction
