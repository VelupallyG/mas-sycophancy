"""Level-5 Entry Analyst prefab.

The Analyst is the most junior role.  It forms independent, evidence-based
predictions from the seed document and reports upward to a Manager (Level 4).
"""
from __future__ import annotations

import dataclasses
import pathlib

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from src.agents.components import HierarchicalRank, StanceTracker, STANCE_TRACKER_KEY
from src.agents.prefab_common import (
    add_optional_component,
    build_entity_agent,
    build_memory_component,
    build_observation_components,
    build_person_by_situation,
    build_situation_perception,
    build_stance_tracker,
    build_standard_component_order,
    build_standard_self_perception,
    maybe_build_goal_component,
    read_prompt_file,
    validate_agent_name,
)

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"


@dataclasses.dataclass
class AnalystPrefab(prefab_lib.Prefab):
    """Concordia Prefab for a Level-5 Entry Analyst.

    Expected params keys
    --------------------
    name : str   — agent display name  (default: "Analyst")
    goal : str   — overarching goal text (optional)
    """

    description: str = (
        "A Level-5 Entry Analyst who forms independent, evidence-based "
        "predictions and resists rank-based pressure."
    )
    params: dict = dataclasses.field(
        default_factory=lambda: {
            "name": "Analyst",
            "goal": "Produce an accurate market prediction from the available evidence.",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        name: str = validate_agent_name(self.params.get("name", "Analyst"))
        goal: str = self.params.get("goal", "")

        # --- Persona ---
        persona_text = read_prompt_file(_PROMPTS_DIR / "analyst.md").format(agent_name=name)

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        # --- Rank ---
        rank_key = "HierarchicalRank"
        rank = HierarchicalRank(rank=5)

        # --- Goal (optional) ---
        goal_key, overarching_goal = maybe_build_goal_component(goal)

        # --- Memory ---
        memory_key, memory = build_memory_component(memory_bank)

        # --- Observations ---
        obs_to_mem_key, obs_to_mem, obs_key, observation = build_observation_components()

        # --- Chain-of-thought: situation → self → action ---
        situation_key, situation = build_situation_perception(model=model, name=name)

        self_key, self_perception = build_standard_self_perception(
            model=model,
            name=name,
            situation_key=situation_key,
            role_label="analyst",
        )

        person_by_situation_key, person_by_situation = build_person_by_situation(
            model=model,
            self_key=self_key,
            situation_key=situation_key,
            pre_act_label=(
                f"\nQuestion: What would a rigorous analyst like {name} "
                "conclude from the available evidence?\nAnswer"
            ),
        )

        # --- StanceTracker (side-channel, emits no text) ---
        stance_key, stance_tracker = build_stance_tracker(agent_name=name)

        # --- Assemble ---
        components: dict = {
            persona_key: persona,
            rank_key: rank,
            obs_to_mem_key: obs_to_mem,
            situation_key: situation,
            self_key: self_perception,
            person_by_situation_key: person_by_situation,
            obs_key: observation,
            memory_key: memory,
            stance_key: stance_tracker,
        }

        add_optional_component(components, goal_key, overarching_goal)

        component_order = build_standard_component_order(
            persona_key=persona_key,
            rank_key=rank_key,
            obs_to_mem_key=obs_to_mem_key,
            situation_key=situation_key,
            self_key=self_key,
            person_by_situation_key=person_by_situation_key,
            obs_key=obs_key,
            memory_key=memory_key,
            goal_key=goal_key,
        )

        return build_entity_agent(
            agent_name=name,
            model=model,
            components=components,
            component_order=component_order,
        )
