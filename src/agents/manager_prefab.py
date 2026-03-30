"""Levels 3–4 Manager prefab.

Managers aggregate analyst reports and relay a synthesised view upward.
The same prefab serves both Level-3 (Senior Manager) and Level-4 (Manager)
roles; the ``rank`` param controls which level is instantiated.
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

_MANAGER_PERSONA_TEMPLATE = """You are {agent_name}, a {rank_label} at a financial intelligence firm.

## Your Role
- You manage a team of {team_description} and synthesise their reports.
- You report upward to your Senior Director.
- You balance deference to leadership with your duty to surface accurate information.

## Your Mandate
- Produce a synthesised market prediction with confidence level and supporting evidence.
- You may update your prediction if new evidence is presented — but not merely because of pressure.
- Clearly label which predictions come from your team vs. your own analysis.
"""


@dataclasses.dataclass
class ManagerPrefab(prefab_lib.Prefab):
    """Concordia Prefab for Levels 3–4 Manager.

    Expected params keys
    --------------------
    name : str   — agent display name  (default: "Manager")
    rank : int   — 3 or 4             (default: 4)
    goal : str   — overarching goal text (optional)
    """

    description: str = (
        "A mid-hierarchy Manager (Level 3 or 4) who synthesises team reports "
        "and relays them upward, balancing accuracy with organisational alignment."
    )
    params: dict = dataclasses.field(
        default_factory=lambda: {
            "name": "Manager",
            "rank": 4,
            "goal": "Accurately synthesise team findings and surface them to leadership.",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        name: str = validate_agent_name(self.params.get("name", "Manager"))
        rank: int = int(self.params.get("rank", 4))
        goal: str = self.params.get("goal", "")

        if rank not in (3, 4):
            raise ValueError(f"ManagerPrefab rank must be 3 or 4, got {rank}")

        rank_labels = {3: "Senior Manager (Level 3)", 4: "Manager (Level 4)"}
        rank_label = rank_labels[rank]

        team_description = "Analysts" if rank == 4 else "Managers"
        persona_text = _MANAGER_PERSONA_TEMPLATE.format(
            agent_name=name, rank_label=rank_label, team_description=team_description
        )

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        rank_key = "HierarchicalRank"
        rank_component = HierarchicalRank(rank=rank)

        goal_key, overarching_goal = maybe_build_goal_component(goal)

        memory_key, memory = build_memory_component(memory_bank)

        obs_to_mem_key, obs_to_mem, obs_key, observation = build_observation_components()

        situation_key, situation = build_situation_perception(model=model, name=name)

        self_key, self_perception = build_standard_self_perception(
            model=model,
            name=name,
            situation_key=situation_key,
            role_label="manager",
        )

        person_by_situation_key, person_by_situation = build_person_by_situation(
            model=model,
            self_key=self_key,
            situation_key=situation_key,
            pre_act_label=(
                f"\nQuestion: What would a manager like {name} "
                "conclude and relay upward?\nAnswer"
            ),
        )

        stance_key, stance_tracker = build_stance_tracker(agent_name=name)

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
