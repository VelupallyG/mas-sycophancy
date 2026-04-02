"""JSON sanitizer and structured output validator for agent responses.

Vertex AI's generate_content may wrap JSON in markdown fences or add preamble
text even when response_mime_type="application/json" is set. This module
provides a defensive parse-and-validate pipeline called by every agent prefab.

On parse or validation failure, the caller should:
  1. Log the raw output.
  2. Flag the turn as a System Execution Error in the TRAIL log.
  3. Carry the agent's previous stance forward.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

VALID_DIRECTIONS: frozenset[str] = frozenset({"POSITIVE", "NEGATIVE", "NEUTRAL"})
REQUIRED_KEYS: frozenset[str] = frozenset(
    {"prediction_direction", "confidence", "prediction_summary", "key_factors"}
)


def sanitize_json_string(raw: str) -> str:
    """Strip markdown fences and preamble from a raw LLM response string.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Extract content between ```json ... ``` or ``` ... ``` fences if present.

    Args:
        raw: Raw string from the model.

    Returns:
        Cleaned string that should be parseable by json.loads().
    """
    text = raw.strip()

    # Step 2: strip markdown code fences.
    fence_match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
        return text

    return text


def _decode_first_json_object(text: str) -> dict | None:
    """Decode the first valid JSON object found in text.

    This handles model outputs that include preamble/epilogue text around JSON.
    """
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_agent_output(raw: str) -> dict | None:
    """Parse and validate a raw agent output string into a structured dict.

    Args:
        raw: Raw string returned by agent.act().

    Returns:
        Validated dict with keys: prediction_direction, confidence,
        prediction_summary, key_factors. Returns None on any failure.
    """
    if not raw or not raw.strip():
        logger.warning("parse_agent_output: received empty string.")
        return None

    cleaned = sanitize_json_string(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _decode_first_json_object(cleaned)
        if parsed is None:
            logger.warning(
                "parse_agent_output: JSON parse error after sanitization. "
                "Raw output: %.200s",
                raw,
            )
            return None

    if not isinstance(parsed, dict):
        logger.warning(
            "parse_agent_output: expected dict, got %s. Raw: %.200s",
            type(parsed).__name__,
            raw,
        )
        return None

    # Validate required keys.
    missing = REQUIRED_KEYS - parsed.keys()
    if missing:
        logger.warning(
            "parse_agent_output: missing keys %s. Raw: %.200s",
            missing,
            raw,
        )
        return None

    # Validate prediction_direction enum.
    direction = parsed.get("prediction_direction")
    if direction not in VALID_DIRECTIONS:
        logger.warning(
            "parse_agent_output: invalid prediction_direction %r. "
            "Must be one of %s. Raw: %.200s",
            direction,
            VALID_DIRECTIONS,
            raw,
        )
        return None

    # Coerce confidence to float.
    try:
        parsed["confidence"] = float(parsed["confidence"])
    except (TypeError, ValueError):
        logger.warning(
            "parse_agent_output: confidence %r is not a float. Raw: %.200s",
            parsed.get("confidence"),
            raw,
        )
        return None

    # Coerce key_factors to list of strings.
    if not isinstance(parsed.get("key_factors"), list):
        parsed["key_factors"] = [str(parsed.get("key_factors", ""))]

    return parsed
