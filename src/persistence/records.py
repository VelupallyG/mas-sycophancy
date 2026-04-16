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
