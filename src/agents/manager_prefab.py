"""Levels 3–4 Manager prefab.

Managers aggregate analyst reports and relay a synthesised view upward.
The same prefab serves both Level-3 (Senior Manager) and Level-4 (Manager)
roles; the ``rank`` param controls which level is instantiated.
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
        name: str = self.params.get("name", "Manager")
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
            pre_act_label=f"\nQuestion: What kind of manager is {name}?\nAnswer",
        )

        person_by_situation_key = "PersonBySituation"
        person_by_situation = (
            agent_components.question_of_recent_memories.PersonBySituation(
                model=model,
                num_memories_to_retrieve=5,
                components=[self_key, situation_key],
                pre_act_label=(
                    f"\nQuestion: What would a manager like {name} "
                    "conclude and relay upward?\nAnswer"
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
