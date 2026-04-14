"""Analyst / Manager agent prefab.

Builds non-orchestrator agents for both L2 Managers and L3 Analysts.
Each rank loads a role-specific prompt:
  - L2_MANAGER  → prompts/manager.md  (synthesis-focused)
  - L3_ANALYST  → prompts/analyst.md  (individual analysis)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from src.agents import prefab_common

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_RANK_PROMPT_MAP: dict[str, Path] = {
    "L2_MANAGER": _PROMPTS_DIR / "manager.md",
    "L3_ANALYST": _PROMPTS_DIR / "analyst.md",
}


@dataclasses.dataclass
class AnalystPrefab(prefab_lib.Prefab):
    """Prefab for non-orchestrator agents (L2 Managers and L3 Analysts).

    Each rank loads a different prompt file so that managers receive
    synthesis-oriented instructions while analysts receive individual-
    analysis instructions.

    Params (passed via self.params dict, all strings):
        name: Unique agent identifier (e.g., "analyst_07").
        rank: One of "L2_MANAGER" or "L3_ANALYST".
    """

    description: str = (
        "Intelligence agent for the MAS sycophancy experiment. "
        "Produces structured JSON predictions from intelligence packets."
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> prefab_lib.prefab_lib.EntityWithComponents:  # type: ignore[name-defined]
        """Build an analyst or manager EntityAgent.

        Note: memory_bank is accepted per the Prefab interface contract but
        is not used — we use ListMemory (no embedder required).
        """
        del memory_bank  # ListMemory used instead; no embedder needed.

        name = self.params.get("name", "analyst")
        rank = self.params.get("rank", "L3_ANALYST")
        tools = self.params.get("tools", None)
        max_tool_calls = int(self.params.get("max_tool_calls", 3))

        prompt_path = _RANK_PROMPT_MAP.get(rank)
        if prompt_path is None:
            raise ValueError(
                f"No prompt file mapped for rank {rank!r}. "
                f"Expected one of {set(_RANK_PROMPT_MAP)}."
            )
        persona = prompt_path.read_text(encoding="utf-8")

        return prefab_common.make_agent(
            name=name,
            model=model,
            persona=persona,
            rank=rank,
            tools=tools,
            max_tool_calls=max_tool_calls,
        )
