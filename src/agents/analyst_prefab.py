"""Analyst agent prefab.

For the prototype, all 20 non-orchestrator agents use this prefab regardless
of whether they are L2 Managers or L3 Analysts. The only difference between
ranks is their RankComponent value, which controls the GM's communication
routing — not any difference in reasoning capability or persona.

This isolates topology as the sole independent variable per the research design.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from src.agents import prefab_common

_PROMPT_PATH = Path(__file__).parent / "prompts" / "financial_analyst.md"


@dataclasses.dataclass
class AnalystPrefab(prefab_lib.Prefab):
    """Prefab for Financial Analyst agents (L2 Managers and L3 Analysts).

    Params (passed via self.params dict, all strings):
        name: Unique agent identifier (e.g., "analyst_07").
        rank: One of "L2_MANAGER" or "L3_ANALYST".
    """

    description: str = (
        "Financial Analyst agent for the MAS sycophancy experiment. "
        "Produces structured JSON predictions from market intelligence packets."
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> prefab_lib.prefab_lib.EntityWithComponents:  # type: ignore[name-defined]
        """Build an analyst EntityAgent.

        Note: memory_bank is accepted per the Prefab interface contract but
        is not used — we use ListMemory (no embedder required).
        """
        del memory_bank  # ListMemory used instead; no embedder needed.

        name = self.params.get("name", "analyst")
        rank = self.params.get("rank", "L3_ANALYST")
        tools = self.params.get("tools", None)
        max_tool_calls = int(self.params.get("max_tool_calls", 3))
        persona = _PROMPT_PATH.read_text(encoding="utf-8")

        return prefab_common.make_agent(
            name=name,
            model=model,
            persona=persona,
            rank=rank,
            tools=tools,
            max_tool_calls=max_tool_calls,
        )
