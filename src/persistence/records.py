"""Small record types used by the optional persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentRunRecord:
    """Metadata needed to create one experiment run row."""

    run_id: str
    seed_id: str
    topology: str
    condition: str
    trial_id: int
    rerun_id: int | None = None


@dataclass(frozen=True)
class AgentMessageRecord:
    """One persisted agent message or routed observation."""

    run_id: str
    agent_name: str
    agent_role: str
    round_number: int
    message_type: str
    content_json: dict[str, Any]


@dataclass(frozen=True)
class EvidenceDocumentRecord:
    """One local evidence document available for retrieval during experiments."""

    evidence_id: str
    source_type: str
    source_name: str
    title: str
    text_content: str
    seed_id: str | None = None
    entity: str | None = None
    ticker: str | None = None
    document_date: str | None = None
    full_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentRetrievalRecord:
    """One local evidence lookup performed for an experiment run."""

    run_id: str
    agent_name: str
    round_number: int
    query: str
    result_ids: list[str]
