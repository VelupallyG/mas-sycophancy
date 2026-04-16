"""Predictive Intelligence Task loader.

Loads a JSON seed document, formats the intelligence packet into an agent
prompt, and provides the ground truth direction for evaluation.

The ground_truth_direction field is NEVER included in the formatted prompt
sent to agents. It is returned separately via TaskContext for use by the
evaluation pipeline only.

See docs/TASK_GUIDE.md for the full seed document schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SEED_DOCUMENT_DIR = Path(__file__).parent / "seed_documents"
_VALID_DIRECTIONS = frozenset({"POSITIVE", "NEGATIVE", "NEUTRAL"})


def extract_ground_truth_direction(payload: dict[str, Any]) -> str | None:
    """Return ground truth direction from either supported seed schema."""
    legacy = payload.get("ground_truth_direction")
    if isinstance(legacy, str):
        return legacy

    nested = payload.get("ground_truth")
    if isinstance(nested, dict):
        direction = nested.get("direction")
        if isinstance(direction, str):
            return direction
    return None


@dataclass(frozen=True)
class TaskContext:
    """Immutable context object produced by PredictiveIntelligenceTask.get_context()."""

    ground_truth: str
    """The true market direction: "POSITIVE", "NEGATIVE", or "NEUTRAL"."""

    formatted_prompt: str
    """The intelligence packet formatted for agent observation. Does NOT
    contain ground_truth_direction."""

    seed_doc_id: str
    """Metadata ID from the seed document (e.g. "tech_earnings_meta_2022")."""

    domain: str
    """Domain of the seed document (e.g. "finance", "policy", "geopolitics")."""

    seed_file_name: str
    """Stem of the source seed JSON file."""


class PredictiveIntelligenceTask:
    """Loads and formats a seed document for the predictive intelligence task."""

    def __init__(self, seed_file_name: str) -> None:
        """Load a seed document JSON file.

        Args:
            seed_file_name: Stem of the JSON file (e.g., "tech_earnings").
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
        self._seed_file_name = seed_file_name

        self._validate()

    def _validate(self) -> None:
        required = {"metadata", "task_prompt", "intelligence_packet"}
        missing = required - self._data.keys()
        if missing:
            raise ValueError(f"Seed document missing required fields: {missing}")

        direction = self._ground_truth_direction()
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"ground truth direction must be one of {_VALID_DIRECTIONS}, "
                f"got {direction!r}."
            )

    def _ground_truth_direction(self) -> str | None:
        """Return ground truth from either supported seed schema."""
        return extract_ground_truth_direction(self._data)

    @property
    def data(self) -> dict[str, Any]:
        """Return a shallow copy of the raw seed document payload."""
        return dict(self._data)

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
        ground_truth = self._ground_truth_direction()
        if ground_truth is None:
            raise ValueError("Seed document has no valid ground truth direction.")
        return TaskContext(
            ground_truth=ground_truth,
            formatted_prompt=formatted_prompt,
            seed_doc_id=metadata.get("id", "unknown"),
            domain=metadata.get("domain", "unknown"),
            seed_file_name=self._seed_file_name,
        )

    def get_ground_truth(self) -> str:
        """Return the ground truth direction (POSITIVE/NEGATIVE/NEUTRAL)."""
        direction = self._ground_truth_direction()
        if direction is None:
            raise ValueError("Seed document has no valid ground truth direction.")
        return direction
