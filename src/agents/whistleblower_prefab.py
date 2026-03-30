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

from src.agents.components import HierarchicalRank, STANCE_TRACKER_KEY
from src.agents.prefab_common import (
    add_optional_component,
    build_entity_agent,
    build_memory_component,
    build_observation_components,
    build_person_by_situation,
    build_situation_perception,
    build_stance_tracker,
    build_standard_component_order,
    maybe_build_goal_component,
    read_prompt_file,
    validate_agent_name,
)
from src.agents.whistleblower import WhistleblowerPolicy, rank_label

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

_WHISTLEBLOWER_MEMORY_RETRIEVAL_LIMIT = 200


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
        name: str = validate_agent_name(self.params.get("name", "Whistleblower"))
        rank: int = int(self.params.get("rank", 5))
        goal: str = self.params.get("goal", "")

        if rank not in {1, 2, 3, 4, 5}:
            raise ValueError(f"WhistleblowerPrefab rank must be 1–5, got {rank}")

        rank_text = rank_label(rank)

        persona_text = (
            read_prompt_file(_PROMPTS_DIR / "whistleblower.md")
            .format(agent_name=name, rank_label=rank_text)
        )

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        rank_key = "HierarchicalRank"
        rank_component = HierarchicalRank(rank=rank)

        policy_key = "CriticalReasoningDirective"
        policy = WhistleblowerPolicy()
        system_directive = agent_components.constant.Constant(
            state=policy.build_system_directive(agent_name=name, rank=rank),
            pre_act_label="\nCritical Reasoning Directive",
        )

        goal_key, overarching_goal = maybe_build_goal_component(goal)

        memory_key, memory = build_memory_component(memory_bank)

        obs_to_mem_key, obs_to_mem, obs_key, observation = build_observation_components()

        situation_key, situation = build_situation_perception(model=model, name=name)

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
            num_memories_to_retrieve=_WHISTLEBLOWER_MEMORY_RETRIEVAL_LIMIT,
            components=[situation_key],
        )

        person_by_situation_key, person_by_situation = build_person_by_situation(
            model=model,
            self_key=self_key,
            situation_key=situation_key,
            pre_act_label=(
                f"\nQuestion: What would a truth-seeking analyst like {name} "
                "say next to protect the accuracy of the group's conclusion?\nAnswer"
            ),
        )

        stance_key, stance_tracker = build_stance_tracker(agent_name=name)

        components: dict = {
            persona_key: persona,
            policy_key: system_directive,
            rank_key: rank_component,
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
        component_order.insert(1, policy_key)

        return build_entity_agent(
            agent_name=name,
            model=model,
            components=components,
            component_order=component_order,
        )
