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

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"


def _validate_agent_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if "{" in name or "}" in name:
        raise ValueError("name cannot contain '{' or '}'")
    return name


def _read_prompt_file(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"failed to load prompt file: {path}") from exc


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
        name: str = _validate_agent_name(self.params.get("name", "Analyst"))
        goal: str = self.params.get("goal", "")

        # --- Persona ---
        persona_text = _read_prompt_file(_PROMPTS_DIR / "analyst.md").format(agent_name=name)

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        # --- Rank ---
        rank_key = "HierarchicalRank"
        rank = HierarchicalRank(rank=5)

        # --- Goal (optional) ---
        if goal:
            goal_key = "Goal"
            overarching_goal = agent_components.constant.Constant(
                state=goal, pre_act_label="\nGoal"
            )
        else:
            goal_key = None
            overarching_goal = None

        # --- Memory ---
        memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
        memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

        # --- Observations ---
        obs_to_mem_key = "ObservationToMemory"
        obs_to_mem = agent_components.observation.ObservationToMemory()

        obs_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
        observation = agent_components.observation.LastNObservations(
            history_length=1_000_000,
            pre_act_label="\nEvents so far (oldest to newest)",
        )

        # --- Chain-of-thought: situation → self → action ---
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
            pre_act_label=f"\nQuestion: What kind of analyst is {name}?\nAnswer",
        )

        person_by_situation_key = "PersonBySituation"
        person_by_situation = (
            agent_components.question_of_recent_memories.PersonBySituation(
                model=model,
                num_memories_to_retrieve=5,
                components=[self_key, situation_key],
                pre_act_label=(
                    f"\nQuestion: What would a rigorous analyst like {name} "
                    "conclude from the available evidence?\nAnswer"
                ),
            )
        )

        # --- StanceTracker (side-channel, emits no text) ---
        stance_key = STANCE_TRACKER_KEY
        stance_tracker = StanceTracker(agent_name=name)

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
