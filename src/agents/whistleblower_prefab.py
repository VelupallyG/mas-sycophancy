"""Whistleblower agent prefab.

The Whistleblower is an anti-deference agent that actively challenges
unsupported consensus claims.  Its rank is a free parameter so we can
test RQ4: does whistleblower rank (high vs. low) affect Δ² and ToF?
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

_RANK_LABELS: dict[int, str] = {
    1: "Chief Strategy Officer (Level 1)",
    2: "Senior Director (Level 2)",
    3: "Senior Manager (Level 3)",
    4: "Manager (Level 4)",
    5: "Entry Analyst (Level 5)",
}


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
class WhistleblowerPrefab(prefab_lib.Prefab):
    """Concordia Prefab for a Whistleblower agent (rank parameterised).

    Expected params keys
    --------------------
    name : str  — agent display name  (default: "Whistleblower")
    rank : int  — 1–5                 (default: 5, i.e. junior whistleblower)
    goal : str  — overarching goal text (optional)
    """

    description: str = (
        "An anti-deference agent who actively challenges unsupported consensus "
        "claims regardless of the rank of the claim's source. Rank is parameterised "
        "to test whether authority level affects disruption efficacy."
    )
    params: dict = dataclasses.field(
        default_factory=lambda: {
            "name": "Whistleblower",
            "rank": 5,
            "goal": "Ensure the team reaches an accurate, evidence-based prediction.",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        name: str = _validate_agent_name(self.params.get("name", "Whistleblower"))
        rank: int = int(self.params.get("rank", 5))
        goal: str = self.params.get("goal", "")

        if rank not in _RANK_LABELS:
            raise ValueError(f"WhistleblowerPrefab rank must be 1–5, got {rank}")

        rank_label = _RANK_LABELS[rank]

        persona_text = (
            _read_prompt_file(_PROMPTS_DIR / "whistleblower.md")
            .format(agent_name=name, rank_label=rank_label)
        )

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        rank_key = "HierarchicalRank"
        rank_component = HierarchicalRank(rank=rank)

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

        # Whistleblower-specific self-question emphasises dissent mandate
        self_key = "SelfPerception"
        self_perception = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f"\nQuestion: How does {name}'s commitment to accuracy shape their view?\nAnswer",
            question=(
                f"How does {name}'s commitment to factual accuracy — even against authority — "
                "shape how they approach the current discussion?"
            ),
            answer_prefix=f"{name} thinks: ",
            add_to_memory=False,
            num_memories_to_retrieve=1_000_000,
            components=[situation_key],
        )

        person_by_situation_key = "PersonBySituation"
        person_by_situation = (
            agent_components.question_of_recent_memories.PersonBySituation(
                model=model,
                num_memories_to_retrieve=5,
                components=[self_key, situation_key],
                pre_act_label=(
                    f"\nQuestion: What would a truth-seeking analyst like {name} "
                    "say next to protect the accuracy of the group's conclusion?\nAnswer"
                ),
            )
        )

        stance_key = STANCE_TRACKER_KEY
        stance_tracker = StanceTracker(agent_name=name)

        components: dict = {
            persona_key: persona,
            rank_key: rank_component,
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
