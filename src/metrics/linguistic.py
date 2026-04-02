"""Deference marker detection for TRAIL linguistic analysis.

Provides keyword-matching heuristics against the deference lexicon for the
prototype. The full study upgrades this to an LLM-as-judge pipeline.

All matching is case-insensitive. Markers are checked against the agent's
prediction_summary field.
"""

from __future__ import annotations

import json
from pathlib import Path

_LEXICON_PATH = Path(__file__).parent / "deference_lexicon.json"
_lexicon_cache: dict | None = None


def _load_lexicon() -> dict:
    global _lexicon_cache
    if _lexicon_cache is None:
        with _LEXICON_PATH.open("r", encoding="utf-8") as f:
            _lexicon_cache = json.load(f)
    return _lexicon_cache


def get_all_deference_markers() -> list[str]:
    """Return the combined flat list of all deference marker phrases."""
    lexicon = _load_lexicon()
    markers: list[str] = []
    for key, phrases in lexicon.items():
        if key.startswith("_"):
            continue
        if isinstance(phrases, list):
            markers.extend(phrases)
    return markers


def detect_deference(text: str, markers: list[str] | None = None) -> bool:
    """Return True if any deference marker is found in text.

    Args:
        text: The agent's prediction_summary (or any free-text field).
        markers: Optional pre-loaded marker list. Loads from lexicon if None.

    Returns:
        True if at least one marker phrase is found (case-insensitive).
    """
    if not text:
        return False
    if markers is None:
        markers = get_all_deference_markers()
    text_lower = text.lower()
    return any(marker in text_lower for marker in markers)


def count_deference_markers(text: str, markers: list[str] | None = None) -> int:
    """Count distinct deference markers found in text.

    Args:
        text: The agent's prediction_summary or combined output text.
        markers: Optional pre-loaded marker list.

    Returns:
        Number of distinct marker phrases found.
    """
    if not text:
        return 0
    if markers is None:
        markers = get_all_deference_markers()
    text_lower = text.lower()
    return sum(1 for marker in markers if marker in text_lower)


def extract_seed_doc_terms(intelligence_packet: dict) -> set[str]:
    """Extract key noun-phrases from a seed document's intelligence_packet.

    Used by TRAIL reasoning error detection: if an agent's key_factors contain
    terms not present in the seed document, it may be fabricating evidence.

    Args:
        intelligence_packet: The 'intelligence_packet' dict from a seed document.

    Returns:
        Set of lowercased word tokens from the packet's text fields.
    """
    tokens: set[str] = set()

    def tokenise(text: str) -> None:
        for word in text.lower().split():
            # Strip common punctuation.
            word = word.strip(".,;:\"'()[]{}").strip()
            if len(word) > 3:  # Skip very short words.
                tokens.add(word)

    for field in ("background", "catalyst_event"):
        if field in intelligence_packet:
            tokenise(intelligence_packet[field])

    for signal_list in ("bullish_signals", "bearish_signals"):
        for signal in intelligence_packet.get(signal_list, []):
            tokenise(signal)

    return tokens
