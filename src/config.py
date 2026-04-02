"""Experiment configuration dataclasses."""

from __future__ import annotations

import dataclasses
import os
from enum import Enum
from pathlib import Path


class Condition(str, Enum):
    FLAT_BASELINE = "flat_baseline"
    FLAT_HALLUCINATION = "flat_hallucination"
    HIERARCHICAL_HALLUCINATION = "hierarchical_hallucination"


class SeedDocument(str, Enum):
    TECH_EARNINGS = "tech_earnings"
    POLICY_DRAFT = "policy_draft"
    GEOPOLITICAL_EVENT = "geopolitical_event"


@dataclasses.dataclass
class ExperimentConfig:
    condition: Condition
    seed_doc: SeedDocument
    n_trials: int = 30
    n_turns: int = 10
    random_seed: int = 42
    model_id: str = dataclasses.field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash-002")
    )
    gcp_project: str = dataclasses.field(
        default_factory=lambda: os.getenv("GCP_PROJECT", "")
    )
    gcp_location: str = dataclasses.field(
        default_factory=lambda: os.getenv("GCP_LOCATION", "us-central1")
    )
    temperature: float = 0.2
    rate_limit_rpm: int = dataclasses.field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_RPM", "60"))
    )
    # K=3: each flat-with-hallucination trial is run K times with different
    # randomly selected injector agents, then averaged.
    n_flat_injection_reruns: int = 3
    output_dir: Path = Path("data")
    hallucination_prompt_version: str = "v1"

    def trial_output_dir(self, trial_id: int) -> Path:
        return (
            self.output_dir
            / self.condition.value
            / self.seed_doc.value
            / f"trial_{trial_id:03d}"
        )

    def jsonl_path(self, trial_id: int) -> Path:
        return self.trial_output_dir(trial_id) / "trace.jsonl"

    def validate(self) -> None:
        if not self.gcp_project:
            raise ValueError(
                "gcp_project must be set. Export GCP_PROJECT or pass it explicitly."
            )
        if self.n_turns < 1:
            raise ValueError("n_turns must be >= 1.")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1.")
