"""Shared prefab construction helpers.

make_agent() is the single factory used by both analyst_prefab.py and
orchestrator_prefab.py. It wires together the Concordia components in the
configuration validated by the API spike (scripts/spike_concordia_vertex.py).

Memory retention strategy (prototype):
    - Keep full observation history for the entire 10-turn horizon.
    - The GM still enforces communication topology; this only affects how much
        of the observed history is surfaced into each act() call.
    - Concordia's LastNObservations requires an integer history_length, so we use
        a very large value to avoid truncation under the prototype's turn cap.
"""

from __future__ import annotations

from collections.abc import Sequence

from concordia.agents import entity_agent
from concordia.components.agent import (
    concat_act_component,
    constant,
    memory as memory_lib,
    observation as observation_lib,
)
from concordia.document import tool as tool_module
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib

from src.agents.components import RankComponent, StanceTracker
from src.agents.tool_act_component import ToolUseActComponent


# The call-to-action appended to every agent's prompt each turn.
# Agents always output structured JSON — the JSON schema constraint is enforced
# here (in the prompt) and by response_mime_type in VertexAILanguageModel.
CALL_TO_ACTION = (
    "Based on the intelligence and observations above, output your updated "
    "prediction as a JSON object with exactly these keys:\n"
    '  "prediction_direction": one of "POSITIVE", "NEGATIVE", or "NEUTRAL"\n'
    '  "confidence": float 0.0–1.0\n'
    '  "prediction_summary": string, max 100 words\n'
    '  "key_factors": list of 2–4 strings citing specific data points\n'
    "Output ONLY the JSON object. Do not include any other text."
)

CALL_TO_ACTION_WITH_TOOLS = (
    "Before making your prediction, you may query the financial database for "
    "additional data using the available tools. Query only the data you need.\n\n"
    "When you are ready, output your prediction as a JSON object with exactly "
    "these keys:\n"
    '  "prediction_direction": one of "POSITIVE", "NEGATIVE", or "NEUTRAL"\n'
    '  "confidence": float 0.0–1.0\n'
    '  "prediction_summary": string, max 100 words\n'
    '  "key_factors": list of 2–4 strings citing specific data points\n'
    "Output ONLY the JSON object as your final answer."
)

ACTION_SPEC = entity_lib.free_action_spec(
    call_to_action=CALL_TO_ACTION,
    tag="prediction",
)

ACTION_SPEC_WITH_TOOLS = entity_lib.free_action_spec(
    call_to_action=CALL_TO_ACTION_WITH_TOOLS,
    tag="prediction",
)


def make_agent(
    name: str,
    model: language_model.LanguageModel,
    persona: str,
    rank: str,
    observation_history_length: int = 10_000,
    tools: Sequence[tool_module.Tool] | None = None,
    max_tool_calls: int = 3,
) -> entity_agent.EntityAgent:
    """Build a Concordia EntityAgent with ListMemory and stance tracking.

    Component composition (in context order):
      instructions  — persona + role (Constant)
      rank          — hierarchical rank label (RankComponent)
      obs_display   — recent observations surfaced into context (LastNObservations)
      stance_tracker — previous prediction surfaced into context (StanceTracker)

    Observation routing:
      - agent.observe(text) → ObservationToMemory → stored in ListMemory
      - On next agent.act() → LastNObservations retrieves last N entries

    Args:
        name: Unique agent name (e.g., "analyst_07", "orchestrator").
        model: Concordia LanguageModel (VertexAILanguageModel or MockModel).
        persona: System prompt / role description for this agent.
        rank: One of "L1_ORCHESTRATOR", "L2_MANAGER", "L3_ANALYST", "PEER".
        observation_history_length: Max number of memory entries surfaced per turn.
        tools: Optional sequence of Concordia Tools for financial data access.
        max_tool_calls: Max tool invocations per act() call (default 3).

    Returns:
        A ready EntityAgent. Call agent.observe(text) and agent.act(ACTION_SPEC).
    """
    mem = memory_lib.ListMemory(memory_bank=[])

    instructions = constant.Constant(
        state=persona,
        pre_act_label="Role and instructions",
    )

    rank_component = RankComponent(rank=rank)

    obs_to_mem = observation_lib.ObservationToMemory()

    obs_display = observation_lib.LastNObservations(
        history_length=observation_history_length,
        pre_act_label="Recent observations (oldest to latest)",
    )

    stance_tracker = StanceTracker()

    component_order = ["instructions", "rank", "obs_display", "stance_tracker"]

    use_tools = bool(tools)
    if use_tools:
        act_component = ToolUseActComponent(
            model=model,
            tools=tools,
            component_order=component_order,
            prefix_entity_name=False,
            max_tool_calls_per_question=max_tool_calls,
        )
    else:
        act_component = concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
            # False: we want pure JSON output, not "AgentName {json...}".
            prefix_entity_name=False,
        )

    return entity_agent.EntityAgent(
        agent_name=name,
        act_component=act_component,
        context_components={
            memory_lib.DEFAULT_MEMORY_COMPONENT_KEY: mem,
            "instructions": instructions,
            "rank": rank_component,
            observation_lib.DEFAULT_OBSERVATION_COMPONENT_KEY: obs_to_mem,
            "obs_display": obs_display,
            "stance_tracker": stance_tracker,
        },
    )


def get_stance_tracker(agent: entity_agent.EntityAgent) -> StanceTracker:
    """Retrieve the StanceTracker component from an agent built by make_agent()."""
    return agent.get_component("stance_tracker", type_=StanceTracker)
