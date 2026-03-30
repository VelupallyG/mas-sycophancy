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

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"
_ALLOWED_HALLUCINATION_PROMPT_VERSIONS = {"v1"}


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
        name: str = _validate_agent_name(self.params.get("name", "CSO"))
        goal: str = self.params.get("goal", "")
        hallucinated_premise: str = self.params.get("hallucinated_premise", "")
        prompt_version: str = self.params.get("prompt_version", "v1")

        # --- Base persona ---
        persona_text = _read_prompt_file(_PROMPTS_DIR / "orchestrator.md").format(
            agent_name=name
        )

        persona_key = "Persona"
        persona = agent_components.constant.Constant(
            state=persona_text,
            pre_act_label="\nPersona",
        )

        # --- Optional hallucinated briefing ---
        if hallucinated_premise:
            hallucination_template = _read_prompt_file(
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
            pre_act_label=f"\nQuestion: What kind of leader is {name}?\nAnswer",
        )

        person_by_situation_key = "PersonBySituation"
        person_by_situation = (
            agent_components.question_of_recent_memories.PersonBySituation(
                model=model,
                num_memories_to_retrieve=5,
                components=[self_key, situation_key],
                pre_act_label=(
                    f"\nQuestion: What strategic decision would a leader like {name} "
                    "make given all available intelligence?\nAnswer"
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
        ]

        # Inject hallucinated briefing immediately after rank
        if briefing is not None:
            components[briefing_key] = briefing
            component_order.append(briefing_key)

        if overarching_goal is not None:
            components[goal_key] = overarching_goal
            component_order.append(goal_key)

        component_order += [
            obs_to_mem_key,
            situation_key,
            self_key,
            person_by_situation_key,
            obs_key,
            memory_key,
        ]

        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
