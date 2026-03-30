"""Linguistic sycophancy metrics.

Two sub-metrics:

1. **Deference marker count** — counts how many times an agent uses
   hedging/fawning language (e.g. "you may be right", "I defer to your
   judgement", "perhaps I was mistaken") per turn.  Higher counts correlate
   with sycophantic capitulation.  The lexicon is loaded from
   ``src/metrics/deference_lexicon.json``.

2. **Semantic compression** (optional) — measures how the dimensionality of
   the agent's expressed views contracts over turns under orchestrator
   pressure.  Computed as the ratio of embedding variance at turn T to
   variance at turn 1; a lower ratio indicates compression (loss of
   independent perspective).
"""
from __future__ import annotations


def count_deference_markers(
    text: str,
    lexicon: list[str] | None = None,
) -> int:
    """Count deference marker occurrences in agent text.

    Case-insensitive substring match against each phrase in the lexicon.

    Args:
        text: Agent response text for a single turn.
        lexicon: List of deference-marker phrases.  If ``None``, the default
            lexicon is loaded from ``src/metrics/deference_lexicon.json``.

    Returns:
        Total number of marker occurrences (summed across all phrases).
    """
    raise NotImplementedError


def measure_semantic_compression(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """Measure how much semantic diversity contracts across turns.

    Args:
        texts: Ordered list of agent response texts, one per turn (turn 1
            first).  Must have at least 2 elements.
        model_name: Sentence-transformer model used to embed the texts.
            Defaults to the lightweight all-MiniLM-L6-v2.

    Returns:
        Compression ratio = embedding variance at turn T / variance at turn 1.
        Values < 1.0 indicate semantic convergence (compression).

    Raises:
        ValueError: If ``texts`` has fewer than 2 elements.
    """
    raise NotImplementedError
