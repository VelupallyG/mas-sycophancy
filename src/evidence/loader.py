"""Load evidence documents from local JSON files and distribute per-agent subsets.

This module bypasses the Postgres persistence layer and reads evidence files
directly from the local_evidence/ directory.  Each agent receives a different
random subset of evidence, creating information asymmetry — the core value
proposition of multi-agent systems.

Evidence allocation strategy:
  - All evidence files matching the seed document's seed_id are loaded.
  - Files are shuffled with a deterministic seed per trial.
  - Each agent receives ``docs_per_agent`` documents, sampled without
    replacement across agents when possible (round-robin deal), falling back
    to with-replacement when the pool is exhausted.
  - A reserve pool of documents is held back for information drip on later turns.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCAL_EVIDENCE_DIR = Path(__file__).resolve().parent.parent.parent / "local_evidence"


@dataclass(frozen=True)
class EvidenceDoc:
    """A single evidence document."""

    id: str
    source_type: str
    source_name: str
    title: str
    text_content: str
    document_date: str | None = None

    def format(self, index: int = 1) -> str:
        """Format for injection into agent context."""
        text = self.text_content.strip()
        if len(text) > 1200:
            text = f"{text[:1197]}..."
        lines = [
            f"[Evidence {index}] {self.title}",
            f"Source: {self.source_type} / {self.source_name}",
        ]
        if self.document_date:
            lines.append(f"Date: {self.document_date}")
        lines.append(text)
        return "\n".join(lines)


@dataclass
class EvidenceAllocation:
    """Per-agent evidence assignments and drip schedule."""

    agent_evidence: dict[str, list[EvidenceDoc]] = field(default_factory=dict)
    drip_pool: list[EvidenceDoc] = field(default_factory=list)
    drip_schedule: dict[int, list[EvidenceDoc]] = field(default_factory=dict)


def load_evidence_files(
    seed_id: str,
    evidence_dir: Path | None = None,
) -> list[EvidenceDoc]:
    """Load all evidence files matching a seed document ID.

    Searches all subdirectories of ``evidence_dir`` (default: local_evidence/).
    Files must be JSON with a "seed_id" field matching ``seed_id``.
    """
    base_dir = evidence_dir or _LOCAL_EVIDENCE_DIR
    if not base_dir.exists():
        logger.warning("Evidence directory not found: %s", base_dir)
        return []

    docs: list[EvidenceDoc] = []
    for json_file in sorted(base_dir.rglob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping %s: %s", json_file, exc)
            continue

        if data.get("seed_id") != seed_id:
            continue

        text = data.get("text_content", "")
        if not text or not text.strip():
            continue

        docs.append(
            EvidenceDoc(
                id=data.get("id", json_file.stem),
                source_type=data.get("source_type", "unknown"),
                source_name=data.get("source_name", "unknown"),
                title=data.get("title", "Untitled"),
                text_content=text,
                document_date=data.get("document_date"),
            )
        )

    logger.info("Loaded %d evidence files for seed_id=%s", len(docs), seed_id)
    return docs


def allocate_evidence(
    docs: list[EvidenceDoc],
    agent_names: list[str],
    docs_per_agent: int = 5,
    drip_turns: tuple[int, ...] = (3, 5, 7),
    docs_per_drip: int = 2,
    rng_seed: int = 42,
) -> EvidenceAllocation:
    """Distribute evidence documents across agents with held-back drip pool.

    Strategy:
      1. Shuffle all docs deterministically.
      2. Reserve ``len(drip_turns) * docs_per_drip`` docs for information drip.
      3. Deal remaining docs round-robin to agents, ``docs_per_agent`` each.
         If the pool runs out, reshuffle and continue (with-replacement).
      4. Schedule drip docs to specific turns.

    Args:
        docs: All available evidence documents.
        agent_names: Ordered list of agent names.
        docs_per_agent: How many docs each agent receives on turn 1.
        drip_turns: Which turns get new evidence injected.
        docs_per_drip: How many new docs per drip turn (broadcast to all agents).
        rng_seed: Deterministic random seed.

    Returns:
        EvidenceAllocation with per-agent assignments and drip schedule.
    """
    rng = random.Random(rng_seed)
    allocation = EvidenceAllocation()

    if not docs:
        for name in agent_names:
            allocation.agent_evidence[name] = []
        return allocation

    shuffled = list(docs)
    rng.shuffle(shuffled)

    # Reserve drip pool
    n_drip = len(drip_turns) * docs_per_drip
    if n_drip < len(shuffled):
        allocation.drip_pool = shuffled[:n_drip]
        available = shuffled[n_drip:]
    else:
        # Not enough docs for drip — skip drip, give all to agents
        allocation.drip_pool = []
        available = shuffled

    # Deal to agents round-robin
    for name in agent_names:
        allocation.agent_evidence[name] = []

    total_needed = len(agent_names) * docs_per_agent
    # Build a dealing deck — if pool is smaller than needed, cycle through
    deck: list[EvidenceDoc] = []
    while len(deck) < total_needed:
        batch = list(available)
        rng.shuffle(batch)
        deck.extend(batch)

    idx = 0
    for name in agent_names:
        allocation.agent_evidence[name] = deck[idx : idx + docs_per_agent]
        idx += docs_per_agent

    # Schedule drip
    drip_idx = 0
    for turn in drip_turns:
        turn_docs = allocation.drip_pool[drip_idx : drip_idx + docs_per_drip]
        if turn_docs:
            allocation.drip_schedule[turn] = turn_docs
        drip_idx += docs_per_drip

    return allocation


def format_evidence_packet(docs: list[EvidenceDoc]) -> str:
    """Format a list of evidence docs into a text block for agent observation."""
    if not docs:
        return ""
    lines = ["SUPPLEMENTARY EVIDENCE (unique to your analysis):"]
    for idx, doc in enumerate(docs, start=1):
        lines.append("")
        lines.append(doc.format(index=idx))
    return "\n".join(lines)


def format_drip_packet(docs: list[EvidenceDoc], turn: int) -> str:
    """Format drip evidence for a specific turn."""
    if not docs:
        return ""
    lines = [f"NEW INTELLIGENCE UPDATE (Turn {turn}):"]
    for idx, doc in enumerate(docs, start=1):
        lines.append("")
        lines.append(doc.format(index=idx))
    return "\n".join(lines)
