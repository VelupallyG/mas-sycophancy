"""Controlled hallucination injection engine.

The ``HallucinationInjector`` takes a ``SeedDocument`` (ground truth news) and
produces a ``HallucinatedSignal`` that is:

* Semantically adjacent to the truth — sounds plausible in domain context
* Authoritative and fluent in tone — designed to be persuasive
* Factually wrong in a specific, measurable way — the ``semantic_distance``
  field quantifies the deviation
* Reproducible given a random seed — no stochastic drift between runs

The fabricated signal is sourced directly from the seed document's
``hallucinated_signal`` field for Phase 1 of the experiment.  Future versions
may use the LLM to *generate* novel hallucinations from the seed text, at
which point the prompt template in
``src/agents/prompts/orchestrator_hallucination_v1.txt`` will be used.
"""
from __future__ import annotations

from pathlib import Path

from src.config import HallucinationConfig
from src.tasks.predictive_intel import HallucinatedSignal, SeedDocument

_ALLOWED_PROMPT_VERSIONS = {"v1"}
_ALLOWED_PROMPT_SUFFIXES = {".md", ".txt"}


def _repo_root() -> Path:
    """Return repository root for resolving prompt template paths."""
    return Path(__file__).resolve().parents[2]


def _prompts_dir() -> Path:
    """Return the canonical prompts directory for hallucination templates."""
    return (_repo_root() / "src" / "agents" / "prompts").resolve()


class HallucinationInjector:
    """Produces reproducible hallucinated signals for Orchestrator injection.

    Args:
        config: Hallucination configuration (prompt version, random seed).
    """

    def __init__(self, config: HallucinationConfig) -> None:
        self.config = config
        self._validate_prompt_version()

    def _validate_prompt_version(self) -> None:
        if self.config.prompt_version not in _ALLOWED_PROMPT_VERSIONS:
            allowed = ", ".join(sorted(_ALLOWED_PROMPT_VERSIONS))
            raise ValueError(
                f"unsupported hallucination prompt version "
                f"'{self.config.prompt_version}'. Allowed: {allowed}"
            )

    @staticmethod
    def _validate_signal(signal: HallucinatedSignal) -> None:
        if not signal.fabricated_claim.strip():
            raise ValueError("fabricated_claim must be a non-empty string")
        if not signal.expected_incorrect_prediction.strip():
            raise ValueError("expected_incorrect_prediction must be non-empty")
        if not 0.0 <= signal.semantic_distance_from_truth <= 1.0:
            raise ValueError("semantic_distance_from_truth must be in [0.0, 1.0]")

    def inject(self, seed: SeedDocument) -> HallucinatedSignal:
        """Return the hallucinated signal for the given seed document.

        In Phase 1 this simply returns the pre-authored signal stored in the
        seed document.  The method signature is forward-compatible with a
        future LLM-generation mode.

        Args:
            seed: Parsed seed document containing the pre-authored signal.

        Returns:
            A ``HallucinatedSignal`` ready to be passed to
            ``OrchestratorAgent.__init__``.
        """
        signal = seed.hallucinated_signal
        self._validate_signal(signal)
        return HallucinatedSignal(
            fabricated_claim=signal.fabricated_claim,
            expected_incorrect_prediction=signal.expected_incorrect_prediction,
            semantic_distance_from_truth=signal.semantic_distance_from_truth,
        )

    def format_prompt(self, signal: HallucinatedSignal) -> str:
        """Format the hallucinated signal into the versioned prompt template.

        Reads the prompt template file specified in
        ``self.config.prompt_path``, substitutes ``{hallucinated_claim}``,
        and returns the full orchestrator system-prompt fragment.

        Args:
            signal: Hallucinated signal to embed in the prompt.

        Returns:
            Formatted prompt string ready to be prepended to the Orchestrator
            system prompt.
        """
        self._validate_signal(signal)

        prompt_path = (_repo_root() / self.config.prompt_path).resolve()
        prompts_dir = _prompts_dir()
        if not prompt_path.is_relative_to(prompts_dir):
            raise ValueError("prompt path must stay within src/agents/prompts")
        if prompt_path.suffix not in _ALLOWED_PROMPT_SUFFIXES:
            raise ValueError("prompt template must use .md or .txt extension")
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt template not found: {prompt_path}")

        template = prompt_path.read_text(encoding="utf-8")
        if "{hallucinated_claim}" not in template:
            raise ValueError("prompt template must include {hallucinated_claim}")

        return template.replace("{hallucinated_claim}", signal.fabricated_claim)
