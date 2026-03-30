"""Level-2 Senior Director prefab.

The Senior Director sits one level below the CSO/Orchestrator.  They
consolidate Manager reports, form a team-level view, and relay it to the
Orchestrator (Level 1).
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

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

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
        name: str = self.params.get("name", "Senior Director")
        goal: str = self.params.get("goal", "")

        persona_text = _DIRECTOR_PERSONA_TEMPLATE.format(agent_name=name)

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        rank_key = "HierarchicalRank"
        rank = HierarchicalRank(rank=2)

        if goal:
            goal_key = "Goal"
            overarching_goal = agent_components.constant.Constant(
                state=goal, pre_act_label="\nGoal"
            )
        else:
            goal_key = None
            overarching_goal = None

        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        obs_to_mem_key = "ObservationToMemory"
        obs_to_mem = agent_components.observation.ObservationToMemory()

        obs_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        observation = agent_components.observation.LastNObservations(
            history_length=1_000_000,
            pre_act_label="\nEvents so far (oldest to newest)",
        )

        situation_key = "SituationPerception"
        situation = agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            num_memories_to_retrieve=25,
            pre_act_label=f"\nQuestion: What situation is {name} in right now?\nAnswer",
        )

        self_key = "SelfPerception"
        self_perception = agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            num_memories_to_retrieve=1_000_000,
            components=[situation_key],
            pre_act_label=f"\nQuestion: What kind of director is {name}?\nAnswer",
        )

        person_by_situation_key = "PersonBySituation"
        person_by_situation = (
            agent_components.question_of_recent_memories.PersonBySituation(
                model=model,
                num_memories_to_retrieve=5,
                components=[self_key, situation_key],
                pre_act_label=(
                    f"\nQuestion: What view would a senior director like {name} "
                    "form and present to the CSO?\nAnswer"
                ),
            )
        )

        stance_key = STANCE_TRACKER_KEY
        stance_tracker = StanceTracker(agent_name=name)

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

        component_order = [
            persona_key,
            rank_key,
            obs_to_mem_key,
            situation_key,
            self_key,
            person_by_situation_key,
            obs_key,
            memory_key,
        ]

        if overarching_goal is not None:
            components[goal_key] = overarching_goal
            component_order.insert(2, goal_key)

        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
