"""Shared helper utilities for Concordia prefab construction.

This module centralizes common component wiring patterns used by multiple
prefabs while preserving role-specific persona and reasoning prompts.
"""
from __future__ import annotations

import pathlib
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model

from src.agents.components import STANCE_TRACKER_KEY, StanceTracker


def validate_agent_name(name: Any) -> str:
    """Validate and normalize prefab agent names used in string templates."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if "{" in name or "}" in name:
        raise ValueError("name cannot contain '{' or '}'")
    return name


def read_prompt_file(path: pathlib.Path) -> str:
    """Load UTF-8 prompt text and raise a context-rich error on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"failed to load prompt file: {path}") from exc


def maybe_build_goal_component(goal: str) -> tuple[str | None, Any | None]:
    """Return optional Goal component tuple."""
    if not goal:
        return None, None

    goal_key = "Goal"
    overarching_goal = agent_components.constant.Constant(
        state=goal,
        pre_act_label="\nGoal",
    )
    return goal_key, overarching_goal


def build_memory_component(
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
) -> tuple[str, Any]:
    """Create the shared associative-memory component."""
    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
    return memory_key, memory


def build_observation_components(history_length: int = 1_000_000) -> tuple[str, Any, str, Any]:
    """Create shared observation components."""
    obs_to_mem_key = "ObservationToMemory"
    obs_to_mem = agent_components.observation.ObservationToMemory()

    obs_key = agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    observation = agent_components.observation.LastNObservations(
        history_length=history_length,
        pre_act_label="\nEvents so far (oldest to newest)",
    )
    return obs_to_mem_key, obs_to_mem, obs_key, observation


def build_situation_perception(
    model: language_model.LanguageModel,
    name: str,
    num_memories_to_retrieve: int = 25,
) -> tuple[str, Any]:
    """Create a standard SituationPerception component."""
    situation_key = "SituationPerception"
    situation = agent_components.question_of_recent_memories.SituationPerception(
        model=model,
        num_memories_to_retrieve=num_memories_to_retrieve,
        pre_act_label=f"\nQuestion: What situation is {name} in right now?\nAnswer",
    )
    return situation_key, situation


def build_standard_self_perception(
    model: language_model.LanguageModel,
    name: str,
    situation_key: str,
    role_label: str,
    num_memories_to_retrieve: int = 1_000_000,
) -> tuple[str, Any]:
    """Create the standard SelfPerception component used by most roles."""
    self_key = "SelfPerception"
    self_perception = agent_components.question_of_recent_memories.SelfPerception(
        model=model,
        num_memories_to_retrieve=num_memories_to_retrieve,
        components=[situation_key],
        pre_act_label=f"\nQuestion: What kind of {role_label} is {name}?\nAnswer",
    )
    return self_key, self_perception


def build_person_by_situation(
    model: language_model.LanguageModel,
    self_key: str,
    situation_key: str,
    pre_act_label: str,
    num_memories_to_retrieve: int = 5,
) -> tuple[str, Any]:
    """Create a standard PersonBySituation component."""
    person_by_situation_key = "PersonBySituation"
    person_by_situation = agent_components.question_of_recent_memories.PersonBySituation(
        model=model,
        num_memories_to_retrieve=num_memories_to_retrieve,
        components=[self_key, situation_key],
        pre_act_label=pre_act_label,
    )
    return person_by_situation_key, person_by_situation


def build_stance_tracker(agent_name: str) -> tuple[str, StanceTracker]:
    """Create side-channel stance tracking component."""
    return STANCE_TRACKER_KEY, StanceTracker(agent_name=agent_name)


def build_entity_agent(
    agent_name: str,
    model: language_model.LanguageModel,
    components: dict[str, Any],
    component_order: list[str],
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Create the final EntityAgentWithLogging from assembled components."""
    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components,
    )


def add_optional_component(
    components: dict[str, Any],
    key: str | None,
    component: Any | None,
) -> None:
    """Insert an optional component into a component map when present."""
    if key is not None and component is not None:
        components[key] = component


def build_standard_component_order(
    persona_key: str,
    rank_key: str,
    obs_to_mem_key: str,
    situation_key: str,
    self_key: str,
    person_by_situation_key: str,
    obs_key: str,
    memory_key: str,
    goal_key: str | None = None,
) -> list[str]:
    """Build the common component order used by non-orchestrator roles."""
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

    if goal_key is not None:
        component_order.insert(2, goal_key)

    return component_order


def build_orchestrator_component_order(
    persona_key: str,
    rank_key: str,
    obs_to_mem_key: str,
    situation_key: str,
    self_key: str,
    person_by_situation_key: str,
    obs_key: str,
    memory_key: str,
    briefing_key: str | None = None,
    goal_key: str | None = None,
) -> list[str]:
    """Build the orchestrator component order including optional briefing."""
    component_order = [persona_key, rank_key]

    if briefing_key is not None:
        component_order.append(briefing_key)

    if goal_key is not None:
        component_order.append(goal_key)

    component_order += [
        obs_to_mem_key,
        situation_key,
        self_key,
        person_by_situation_key,
        obs_key,
        memory_key,
    ]
    return component_order
