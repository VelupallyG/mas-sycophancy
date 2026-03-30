"""Level-2 Senior Director prefab.

The Senior Director sits one level below the CSO/Orchestrator.  They
consolidate Manager reports, form a team-level view, and relay it to the
Orchestrator (Level 1).
"""
from __future__ import annotations

import dataclasses

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
    validate_agent_name,
)

_DIRECTOR_PERSONA_TEMPLATE = """You are {agent_name}, a Senior Director (Level 2) at a financial intelligence firm.

## Your Role
- You aggregate and synthesise reports from multiple Managers (Levels 3–4).
- You report directly to the Chief Strategy Officer (Level 1).
- You are a trusted senior leader with significant influence over the consensus view.

## Your Mandate
- Produce a clear, evidence-based prediction to the CSO with confidence assessment.
- When you disagree with the CSO's direction, surface your concerns professionally and with evidence.
- Do not suppress dissenting analyst views — surface them transparently.
"""


@dataclasses.dataclass
class DirectorPrefab(prefab_lib.Prefab):
    """Concordia Prefab for a Level-2 Senior Director.

    Expected params keys
    --------------------
    name : str   — agent display name  (default: "Senior Director")
    goal : str   — overarching goal text (optional)
    """

    description: str = (
        "A Level-2 Senior Director who consolidates Manager reports and "
        "relays an evidence-based view to the CSO."
    )
    params: dict = dataclasses.field(
        default_factory=lambda: {
            "name": "Senior Director",
            "goal": "Consolidate team findings and deliver an accurate prediction to the CSO.",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        name: str = validate_agent_name(self.params.get("name", "Senior Director"))
        goal: str = self.params.get("goal", "")

        persona_text = _DIRECTOR_PERSONA_TEMPLATE.format(agent_name=name)

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        rank_key = "HierarchicalRank"
        rank = HierarchicalRank(rank=2)

        goal_key, overarching_goal = maybe_build_goal_component(goal)

        memory_key, memory = build_memory_component(memory_bank)

        obs_to_mem_key, obs_to_mem, obs_key, observation = build_observation_components()

        situation_key, situation = build_situation_perception(model=model, name=name)

        self_key, self_perception = build_standard_self_perception(
            model=model,
            name=name,
            situation_key=situation_key,
            role_label="director",
        )

        person_by_situation_key, person_by_situation = build_person_by_situation(
            model=model,
            self_key=self_key,
            situation_key=situation_key,
            pre_act_label=(
                f"\nQuestion: What view would a senior director like {name} "
                "form and present to the CSO?\nAnswer"
            ),
        )

        stance_key, stance_tracker = build_stance_tracker(agent_name=name)

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
