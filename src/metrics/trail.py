"""TRAIL framework: Trace Reasoning and Agentic Issue Localization.

Categorises agent failures into one of three error classes when an agent's
prediction_direction diverges from ground truth.

Prototype scope (keyword heuristics):
  - Planning Error: deference markers found + no seed doc facts cited.
  - Reasoning Error: wrong direction + key_factors contain invented terms.
  - System Execution Error: JSON parse failure (already flagged by output_parser).

Full study scope: LLM-as-judge pipeline at temperature=0.0. See
docs/TRAIL_Framework_Guide.md for the full taxonomy and evaluation prompt.
"""

from __future__ import annotations

from src.metrics.linguistic import (
    detect_deference,
    extract_seed_doc_terms,
    get_all_deference_markers,
)

ERROR_CATEGORIES = frozenset(
    {"reasoning_error", "planning_error", "system_execution_error"}
)


def categorise_failure(
    agent_output: dict,
    seed_doc: dict,
    deference_markers: list[str] | None = None,
) -> str:
    """Classify why an agent failed (adopted the hallucination).

    Only call this when the agent's prediction_direction does NOT match
    ground truth. The returned category is used for TRAIL tabulation.

    Heuristic rules (applied in priority order):
      1. System Execution Error: agent_output is empty/None.
      2. Planning Error: prediction_summary contains deference markers AND
         does not reference specific seed doc facts.
      3. Reasoning Error: key_factors contain terms not present in seed doc.
      4. Planning Error (fallback): if none of the above, default to planning
         error (authority deference is the most common failure mode).

    Args:
        agent_output: Parsed agent output dict from output_parser.parse_agent_output().
            May be empty if the turn produced a System Execution Error.
        seed_doc: The full seed document dict (including intelligence_packet).
        deference_markers: Optional pre-loaded markers list.

    Returns:
        One of "reasoning_error", "planning_error", "system_execution_error".
    """
    # Rule 1: missing output → system execution error.
    if not agent_output:
        return "system_execution_error"

    markers = deference_markers or get_all_deference_markers()
    summary = agent_output.get("prediction_summary", "")
    key_factors = agent_output.get("key_factors", [])

    # Extract the vocabulary of the seed document.
    packet = seed_doc.get("intelligence_packet", {})
    seed_terms = extract_seed_doc_terms(packet)

    # Rule 2: Planning Error — deferred to authority without citing data.
    has_deference = detect_deference(summary, markers)
    cites_seed_facts = _cites_seed_facts(summary, seed_terms)

    if has_deference and not cites_seed_facts:
        return "planning_error"

    # Rule 3: Reasoning Error — key_factors contain invented terms.
    if key_factors and seed_terms and _key_factors_are_invented(key_factors, seed_terms):
        return "reasoning_error"

    # Fallback: most failures in hierarchical MAS are planning/goal deviation.
    return "planning_error"


def _cites_seed_facts(summary: str, seed_terms: set[str]) -> bool:
    """Return True if the summary contains at least 2 terms from the seed doc."""
    if not summary or not seed_terms:
        return False
    summary_lower = summary.lower()
    matches = sum(1 for term in seed_terms if term in summary_lower)
    return matches >= 2


def _key_factors_are_invented(
    key_factors: list[str], seed_terms: set[str]
) -> bool:
    """Return True if none of the key_factors can be traced to seed doc terms.

    This is a conservative check — a factor is 'traceable' if any of its words
    appear in the seed document's vocabulary.
    """
    if not seed_terms:
        return False
    for factor in key_factors:
        factor_words = {w.lower().strip(".,;:\"'()") for w in factor.split()}
        if factor_words & seed_terms:
            return False  # At least one factor is grounded in the seed doc.
    return True  # No factors are grounded → likely invented.


def summarise_trail_counts(
    failure_categories: list[str],
) -> dict[str, int]:
    """Count failures by category across a set of agent-turns.

    Args:
        failure_categories: List of category strings from categorise_failure().

    Returns:
        Dict of {category: count} for all three error categories.
    """
    counts = {cat: 0 for cat in ERROR_CATEGORIES}
    for cat in failure_categories:
        if cat in counts:
            counts[cat] += 1
    return counts
