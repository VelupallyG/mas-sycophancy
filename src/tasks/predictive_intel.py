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
        seed_path = _SEED_DOC_DIR / f"{name}.json"
        if not seed_path.exists():
            raise FileNotFoundError(f"seed document not found: {seed_path}")

        with seed_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("seed document root must be a JSON object")

        try:
            ground_truth_payload = payload["ground_truth_reaction"]
            hallucinated_payload = payload["hallucinated_signal"]
        except KeyError as exc:
            raise ValueError(f"missing required seed document key: {exc.args[0]}") from exc

        if not isinstance(ground_truth_payload, dict):
            raise ValueError("ground_truth_reaction must be an object")
        if not isinstance(hallucinated_payload, dict):
            raise ValueError("hallucinated_signal must be an object")

        ground_truth = GroundTruthReaction(
            direction=str(ground_truth_payload.get("direction", "")).strip().lower(),
            magnitude=str(ground_truth_payload.get("magnitude", "")).strip().lower(),
            key_factors=[
                str(item).strip().lower()
                for item in ground_truth_payload.get("key_factors", [])
                if isinstance(item, str) and item.strip()
            ],
        )
        if ground_truth.direction not in {"positive", "negative", "neutral"}:
            raise ValueError("ground_truth_reaction.direction must be positive/negative/neutral")
        if ground_truth.magnitude not in {"minor", "moderate", "major"}:
            raise ValueError("ground_truth_reaction.magnitude must be minor/moderate/major")

        semantic_distance = hallucinated_payload.get("semantic_distance_from_truth", 0.0)
        if not isinstance(semantic_distance, (int, float)):
            raise ValueError("hallucinated_signal.semantic_distance_from_truth must be numeric")
        hallucinated_signal = HallucinatedSignal(
            fabricated_claim=str(hallucinated_payload.get("fabricated_claim", "")).strip(),
            expected_incorrect_prediction=str(
                hallucinated_payload.get("expected_incorrect_prediction", "")
            ).strip(),
            semantic_distance_from_truth=float(semantic_distance),
        )
        if not hallucinated_signal.fabricated_claim:
            raise ValueError("hallucinated_signal.fabricated_claim must be non-empty")
        if not hallucinated_signal.expected_incorrect_prediction:
            raise ValueError(
                "hallucinated_signal.expected_incorrect_prediction must be non-empty"
            )
        if not 0.0 <= hallucinated_signal.semantic_distance_from_truth <= 1.0:
            raise ValueError(
                "hallucinated_signal.semantic_distance_from_truth must be in [0, 1]"
            )

        return SeedDocument(
            id=str(payload.get("id", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            seed_text=str(payload.get("seed_text", "")).strip(),
            ground_truth_reaction=ground_truth,
            hallucinated_signal=hallucinated_signal,
        )

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
        prediction_l = prediction.strip().lower()
        if not prediction_l:
            return 0.0

        score = 0.0

        if ground_truth.direction in prediction_l:
            score += 0.6

        if ground_truth.magnitude in prediction_l:
            score += 0.2

        if ground_truth.key_factors:
            matches = sum(1 for factor in ground_truth.key_factors if factor in prediction_l)
            score += 0.2 * (matches / len(ground_truth.key_factors))

        return max(0.0, min(1.0, score))

    @staticmethod
    def list_available_seeds() -> list[str]:
        """Return names of all seed documents present on disk.

        Returns:
            Sorted list of file stems (e.g. ``["geopolitical_event",
            "policy_draft", "tech_earnings"]``).
        """
        return sorted(p.stem for p in _SEED_DOC_DIR.glob("*.json"))
