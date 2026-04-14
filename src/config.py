"""Experiment configuration dataclasses."""

from __future__ import annotations

import dataclasses
import os
from enum import Enum
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Condition(str, Enum):
    FLAT_BASELINE = "flat_baseline"
    FLAT_HALLUCINATION = "flat_hallucination"
    HIERARCHICAL_HALLUCINATION = "hierarchical_hallucination"


class SeedDocument(str, Enum):
    FINANCE_EARNINGS = "finance_earnings_alphabet_ai_capex_2026_v1"
    GEOPOLITICS_SANCTIONS = "geopolitics_sanctions_oil_supplyshock_2025_v1"


@dataclasses.dataclass
class ExperimentConfig:
    condition: Condition
    seed_doc: SeedDocument
    n_trials: int = 30
    n_turns: int = 10
    random_seed: int = 42
    model_id: str = dataclasses.field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
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
    trail_use_llm_judge: bool = dataclasses.field(
        default_factory=lambda: _env_bool("TRAIL_USE_LLM_JUDGE", False)
    )
    trail_judge_model_id: str = dataclasses.field(
        default_factory=lambda: os.getenv("TRAIL_JUDGE_MODEL_ID", "gemini-2.5-flash")
    )
    trail_judge_temperature: float = dataclasses.field(
        default_factory=lambda: float(os.getenv("TRAIL_JUDGE_TEMPERATURE", "0.0"))
    )
    # K=3: each flat-with-hallucination trial is run K times with different
    # randomly selected injector agents, then averaged.
    n_flat_injection_reruns: int = 3
    output_dir: Path = Path("data")
    hallucination_prompt_version: str = "v1"

    def trial_output_dir(self, trial_id: int, rerun_id: int | None = None) -> Path:
        path = (
            self.output_dir
            / self.condition.value
            / self.seed_doc.value
            / f"trial_{trial_id:03d}"
        )
        if rerun_id is not None:
            path = path / f"rerun_{rerun_id}"
        return path

    def jsonl_path(self, trial_id: int, rerun_id: int | None = None) -> Path:
        return self.trial_output_dir(trial_id, rerun_id=rerun_id) / "trace.jsonl"

    def raw_trace_path(self, trial_id: int, rerun_id: int | None = None) -> Path:
        base = (
            self.output_dir
            / "raw_traces"
            / self.condition.value
            / self.seed_doc.value
            / f"trial_{trial_id:03d}"
        )
        if rerun_id is not None:
            base = base / f"rerun_{rerun_id}"
        return base / "conversation.jsonl"

    def validate(self) -> None:
        if not self.gcp_project:
            raise ValueError(
                "gcp_project must be set. Export GCP_PROJECT or pass it explicitly."
            )
        if self.n_turns < 1:
            raise ValueError("n_turns must be >= 1.")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1.")
        if self.rate_limit_rpm < 1:
            raise ValueError("rate_limit_rpm must be >= 1.")
        if self.trail_judge_temperature != 0.0 and self.trail_use_llm_judge:
            raise ValueError(
                "trail_judge_temperature must be 0.0 when trail_use_llm_judge is enabled."
            )
