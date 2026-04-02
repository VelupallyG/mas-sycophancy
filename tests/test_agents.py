"""Tests for agent prefabs, output parser, and custom components."""

from __future__ import annotations

import json

import pytest
from concordia.testing.mock_model import MockModel

from src.agents.output_parser import parse_agent_output, sanitize_json_string
from src.agents.components import RankComponent, StanceTracker
from src.agents.prefab_common import make_agent, ACTION_SPEC


# ---------------------------------------------------------------------------
# output_parser tests
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps({
    "prediction_direction": "NEGATIVE",
    "confidence": 0.82,
    "prediction_summary": "Bearish signals dominate.",
    "key_factors": ["FCF dropped 98%", "VR losses $3.7B"],
})


def test_parse_agent_output_valid():
    result = parse_agent_output(VALID_JSON)
    assert result is not None
    assert result["prediction_direction"] == "NEGATIVE"
    assert result["confidence"] == pytest.approx(0.82)
    assert isinstance(result["key_factors"], list)


def test_parse_agent_output_with_markdown_fence():
    fenced = f"```json\n{VALID_JSON}\n```"
    result = parse_agent_output(fenced)
    assert result is not None
    assert result["prediction_direction"] == "NEGATIVE"


def test_parse_agent_output_with_preamble():
    preambled = f"Here is my analysis:\n{VALID_JSON}"
    result = parse_agent_output(preambled)
    assert result is not None
    assert result["prediction_direction"] == "NEGATIVE"


def test_parse_agent_output_with_braces_in_preamble():
    raw = (
        "analysis metadata {not_json: true} -- ignore this preface\n"
        f"{VALID_JSON}\n"
        "trailing commentary"
    )
    result = parse_agent_output(raw)
    assert result is not None
    assert result["prediction_direction"] == "NEGATIVE"


def test_parse_agent_output_invalid_direction():
    bad = json.dumps({
        "prediction_direction": "MAYBE",
        "confidence": 0.5,
        "prediction_summary": "unsure",
        "key_factors": [],
    })
    assert parse_agent_output(bad) is None


def test_parse_agent_output_missing_key():
    incomplete = json.dumps({
        "prediction_direction": "POSITIVE",
        "confidence": 0.7,
        # missing prediction_summary and key_factors
    })
    assert parse_agent_output(incomplete) is None


def test_parse_agent_output_empty_string():
    assert parse_agent_output("") is None


def test_sanitize_json_string_strips_fence():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert sanitize_json_string(raw).startswith("{")


# ---------------------------------------------------------------------------
# RankComponent tests
# ---------------------------------------------------------------------------

def test_rank_component_valid():
    comp = RankComponent("L3_ANALYST")
    assert comp.rank == "L3_ANALYST"


def test_rank_component_invalid():
    with pytest.raises(ValueError):
        RankComponent("INVALID_RANK")


# ---------------------------------------------------------------------------
# Agent construction smoke test (MockModel, no API calls)
# ---------------------------------------------------------------------------

def test_make_agent_smoke():
    model = MockModel(response=VALID_JSON)
    agent = make_agent(
        name="test_analyst",
        model=model,
        persona="You are a financial analyst.",
        rank="L3_ANALYST",
    )
    assert agent.name == "test_analyst"


def test_agent_observe_and_act():
    model = MockModel(response=VALID_JSON)
    agent = make_agent(
        name="test_analyst",
        model=model,
        persona="You are a financial analyst.",
        rank="PEER",
    )
    agent.observe("Market intelligence: bearish signals.")
    output = agent.act(ACTION_SPEC)
    # MockModel returns the fixed JSON string regardless of prompt.
    result = parse_agent_output(output)
    assert result is not None
    assert result["prediction_direction"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def test_stance_tracker_captures_output():
    model = MockModel(response=VALID_JSON)
    agent = make_agent(
        name="test_analyst",
        model=model,
        persona="You are a financial analyst.",
        rank="L3_ANALYST",
    )
    agent.observe("Intelligence packet.")
    agent.act(ACTION_SPEC)

    tracker = agent.get_component("stance_tracker", type_=StanceTracker)
    assert len(tracker.get_stance_history()) == 1
    assert tracker.get_current_direction() == "NEGATIVE"
