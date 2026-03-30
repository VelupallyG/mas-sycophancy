"""Predictive Intelligence simulation task.

Agents are given a seed document (breaking news event) and must collectively
predict the market or public reaction.  The ground truth is the actual
reaction documented in the seed JSON.  The hallucinated signal is the
fabricated incorrect prediction injected into the Orchestrator.

Seed documents live in ``src/tasks/seed_documents/*.json`` and follow the
schema defined in PROTOTYPING_GUIDE.md §6.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_SEED_DOC_DIR = Path(__file__).parent / "seed_documents"


@dataclass
class GroundTruthReaction:
    """The actual market or public reaction documented in the seed."""

    direction: str  # "positive" | "negative" | "neutral"
    magnitude: str  # "minor" | "moderate" | "major"
    key_factors: list[str] = field(default_factory=list)


@dataclass
class HallucinatedSignal:
    """The fabricated signal injected into the Orchestrator."""

    fabricated_claim: str
    expected_incorrect_prediction: str
    semantic_distance_from_truth: float


@dataclass
class SeedDocument:
    """Fully parsed seed document for a single experiment instance."""

    id: str
    title: str
    seed_text: str
    ground_truth_reaction: GroundTruthReaction
    hallucinated_signal: HallucinatedSignal


class PredictiveIntelTask:
    """Task interface for the Predictive Intelligence simulation.

    Provides seed-document loading and prediction accuracy evaluation.
    """

    def load_seed(self, name: str) -> SeedDocument:
        """Load and parse a seed document by name.

        Args:
            name: Stem of the JSON file (e.g. ``"tech_earnings"``).

        Returns:
            Parsed ``SeedDocument`` instance.

        Raises:
            FileNotFoundError: If no JSON file with the given name exists.
            ValueError: If the JSON does not conform to the required schema.
        """
        raise NotImplementedError

    def evaluate(
        self,
        prediction: str,
        ground_truth: GroundTruthReaction,
    ) -> float:
        """Score a prediction string against the ground truth.

        Returns a float in [0.0, 1.0] where 1.0 is a perfect match.  The
        scoring is purely programmatic (keyword overlap + direction match) —
        no LLM-as-a-judge is used.

        Args:
            prediction: The consensus prediction text produced by the MAS.
            ground_truth: The ``GroundTruthReaction`` from the seed document.

        Returns:
            Accuracy score in [0.0, 1.0].
        """
        raise NotImplementedError

    @staticmethod
    def list_available_seeds() -> list[str]:
        """Return names of all seed documents present on disk.

        Returns:
            Sorted list of file stems (e.g. ``["geopolitical_event",
            "policy_draft", "tech_earnings"]``).
        """
        return sorted(p.stem for p in _SEED_DOC_DIR.glob("*.json"))
