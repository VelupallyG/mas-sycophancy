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

from src.config import HallucinationConfig
from src.tasks.predictive_intel import HallucinatedSignal, SeedDocument


class HallucinationInjector:
    """Produces reproducible hallucinated signals for Orchestrator injection.

    Args:
        config: Hallucination configuration (prompt version, random seed).
    """

    def __init__(self, config: HallucinationConfig) -> None:
        self.config = config

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
        raise NotImplementedError

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
        raise NotImplementedError
