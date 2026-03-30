"""Level-1 Orchestrator (CEO/CSO) prefab.

The Orchestrator has final decision-making authority.  In hallucination
experiments it receives a fabricated signal via a ``Constant`` component
that is injected at build time.  In the control condition the
``hallucinated_premise`` param is left empty.
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
    build_orchestrator_component_order,
    build_person_by_situation,
    build_situation_perception,
    build_stance_tracker,
    build_standard_self_perception,
    maybe_build_goal_component,
    read_prompt_file,
    validate_agent_name,
)

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
_ALLOWED_HALLUCINATION_PROMPT_VERSIONS = {"v1"}


def _hallucination_prompt_path(prompt_version: str) -> pathlib.Path:
    if prompt_version not in _ALLOWED_HALLUCINATION_PROMPT_VERSIONS:
        allowed = ", ".join(sorted(_ALLOWED_HALLUCINATION_PROMPT_VERSIONS))
        raise ValueError(
            f"unsupported hallucination prompt version '{prompt_version}'. Allowed: {allowed}"
        )

    prompts_dir = _PROMPTS_DIR.resolve()
    candidate = (prompts_dir / f"orchestrator_hallucination_{prompt_version}.md").resolve()
    if not candidate.is_relative_to(prompts_dir):
        raise ValueError("hallucination prompt path escapes prompts directory")
    return candidate


@dataclasses.dataclass
class OrchestratorPrefab(prefab_lib.Prefab):
    """Concordia Prefab for the Level-1 CEO / Chief Strategy Officer.

    Expected params keys
    --------------------
    name                : str — agent display name (default: "CSO")
    goal                : str — overarching goal text (optional)
    hallucinated_premise: str — fabricated signal injected as a private briefing.
                                Empty string = control condition (no hallucination).
    prompt_version      : str — version tag for the hallucination prompt (default: "v1")
    """

    description: str = (
        "The Level-1 Orchestrator (CEO/CSO) with final decision authority. "
        "Optionally receives a hallucinated market signal as a private briefing."
    )
    params: dict = dataclasses.field(
        default_factory=lambda: {
            "name": "CSO",
            "goal": "Lead the team to the most accurate consensus market prediction.",
            "hallucinated_premise": "",
            "prompt_version": "v1",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        name: str = validate_agent_name(self.params.get("name", "CSO"))
        goal: str = self.params.get("goal", "")
        hallucinated_premise: str = self.params.get("hallucinated_premise", "")
        prompt_version: str = self.params.get("prompt_version", "v1")

        # --- Base persona ---
        persona_text = read_prompt_file(_PROMPTS_DIR / "orchestrator.md").format(
            agent_name=name
        )

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        # --- Optional hallucinated briefing ---
        if hallucinated_premise:
            hallucination_template = read_prompt_file(
                _hallucination_prompt_path(prompt_version)
            )
            briefing_text = hallucination_template.replace(
                "{hallucinated_claim}", hallucinated_premise
            ).replace("{agent_name}", name)
            briefing_key = "ConfidentialBriefing"
            briefing = agent_components.constant.Constant(
                state=briefing_text,
                pre_act_label="\n[CONFIDENTIAL INTERNAL BRIEFING]",
            )
        else:
            briefing_key = None
            briefing = None

        rank_key = "HierarchicalRank"
        rank = HierarchicalRank(rank=1)

        goal_key, overarching_goal = maybe_build_goal_component(goal)

        memory_key, memory = build_memory_component(memory_bank)

        obs_to_mem_key, obs_to_mem, obs_key, observation = build_observation_components()

        situation_key, situation = build_situation_perception(model=model, name=name)

        self_key, self_perception = build_standard_self_perception(
            model=model,
            name=name,
            situation_key=situation_key,
            role_label="leader",
        )

        person_by_situation_key, person_by_situation = build_person_by_situation(
            model=model,
            self_key=self_key,
            situation_key=situation_key,
            pre_act_label=(
                f"\nQuestion: What strategic decision would a leader like {name} "
                "make given all available intelligence?\nAnswer"
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

        add_optional_component(components, briefing_key, briefing)
        add_optional_component(components, goal_key, overarching_goal)

        component_order = build_orchestrator_component_order(
            persona_key=persona_key,
            rank_key=rank_key,
            briefing_key=briefing_key,
            goal_key=goal_key,
            obs_to_mem_key=obs_to_mem_key,
            situation_key=situation_key,
            self_key=self_key,
            person_by_situation_key=person_by_situation_key,
            obs_key=obs_key,
            memory_key=memory_key,
        )

        return build_entity_agent(
            agent_name=name,
            model=model,
            components=components,
            component_order=component_order,
        )
