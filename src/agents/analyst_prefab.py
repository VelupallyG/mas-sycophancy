"""Analyst agent prefab.

For the prototype, all 20 non-orchestrator agents use this prefab regardless
of whether they are L2 Managers or L3 Analysts. The difference between
ranks is their RankComponent value (which controls the GM's communication
routing) AND their level-specific prompt overlay, which establishes
appropriate authority dynamics for their position in the hierarchy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from src.agents import prefab_common

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_BASE_PROMPT_PATH = _PROMPTS_DIR / "financial_analyst.md"

# Mapping from rank string to the level-specific prompt overlay file.
_RANK_TO_OVERLAY: dict[str, str] = {
    "L2_MANAGER": "l2_manager.md",
    "L3_ANALYST": "l3_analyst.md",
    "PEER": "flat_peer.md",
}


def _load_persona(rank: str) -> str:
    """Load base persona + level-specific overlay for the given rank."""
    base = _BASE_PROMPT_PATH.read_text(encoding="utf-8")
    overlay_file = _RANK_TO_OVERLAY.get(rank)
    if overlay_file is not None:
        overlay_path = _PROMPTS_DIR / overlay_file
        if overlay_path.exists():
            overlay = overlay_path.read_text(encoding="utf-8")
            return f"{base}\n\n{overlay}"
    return base


@dataclasses.dataclass
class AnalystPrefab(prefab_lib.Prefab):
    """Prefab for Financial Analyst agents (L2 Managers, L3 Analysts, Flat Peers).

    Params (passed via self.params dict, all strings):
        name: Unique agent identifier (e.g., "analyst_07").
        rank: One of "L2_MANAGER", "L3_ANALYST", or "PEER".
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
        hallucination_text = self.params.get("hallucination_text")
        persona = _load_persona(rank)

        return prefab_common.make_agent(
            name=name,
            model=model,
            persona=persona,
            rank=rank,
            hallucination_text=hallucination_text if hallucination_text else None,
        )
