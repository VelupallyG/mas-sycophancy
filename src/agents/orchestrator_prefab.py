"""Orchestrator (L1) agent prefab.

The Orchestrator is the Level 1 CSO in the hierarchical topology. It:
  1. Receives the seed document (same as all other agents).
  2. Optionally receives a hallucination injection appended to its persona.
  3. Acts FIRST on Turn 1 (top-down pressure establishment).
  4. Acts LAST on Turns 2–10 (synthesising bottom-up manager reports).

In the flat baseline condition, no orchestrator is instantiated — the
OrchestratorPrefab is hierarchical-condition-only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from src.agents import prefab_common

_ORCHESTRATOR_PROMPT_PATH = (
    Path(__file__).parent / "prompts" / "orchestrator.md"
)


@dataclasses.dataclass
class OrchestratorPrefab(prefab_lib.Prefab):
    """Prefab for the Level 1 Orchestrator (CSO) agent.

    Params (all strings):
        name: Agent identifier (default: "orchestrator").
        hallucination_injection: Optional additional prompt text appended to the
            orchestrator's persona. Pass the rendered hallucination premise here.
            If empty or absent, the orchestrator runs without hallucination
            (used for debugging / ablation studies).
    """

    description: str = (
        "Chief Strategy Officer agent for the MAS sycophancy experiment. "
        "Synthesises analyst and manager reports; optionally receives a "
        "hallucination injection to act as the hallucinating authority."
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> prefab_lib.prefab_lib.EntityWithComponents:  # type: ignore[name-defined]
        """Build the orchestrator EntityAgent."""
        del memory_bank

        name = self.params.get("name", "orchestrator")
        hallucination_injection = self.params.get("hallucination_injection", "")
        tools = self.params.get("tools", None)
        max_tool_calls = int(self.params.get("max_tool_calls", 3))

        base_persona = _ORCHESTRATOR_PROMPT_PATH.read_text(encoding="utf-8")

        if hallucination_injection:
            persona = f"{base_persona}\n\n{hallucination_injection}"
        else:
            persona = base_persona

        return prefab_common.make_agent(
            name=name,
            model=model,
            persona=persona,
            rank="L1_ORCHESTRATOR",
            tools=tools,
            max_tool_calls=max_tool_calls,
        )
