"""JSONL trace exporter — active ETL pipeline during simulation.

During each turn, this module extracts the structured stance JSON from each
agent's output and appends it to a flat JSONL file (one record per agent-turn).

Output schema (one JSON object per line):
  {
    "trial_id":          "...",
    "seed_doc":          "tech_earnings",
    "condition":         "hierarchical_hallucination",
    "turn":              3,
    "agent_id":          "analyst_07",
    "level":             3,
    "prediction_direction": "NEGATIVE",
    "confidence":        0.72,
    "prediction_summary": "...",
    "key_factors":       ["...", "..."],
    "parse_success":     true,
    "trail_category":    null,
    "timestamp_ms":      1234567890
  }

Raw Concordia traces are not recorded here — they live in data/raw_traces/.
This JSONL format is read directly by analysis/aggregate_results.py via
pd.read_json(path, lines=True).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AgentTurnRecord:
    """One row in the JSONL trace file — one agent, one turn."""

    trial_id: str
    seed_doc: str
    condition: str
    turn: int
    agent_id: str
    level: int
    prediction_direction: str
    confidence: float
    prediction_summary: str
    key_factors: list[str]
    parse_success: bool
    trail_category: str | None = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @classmethod
    def from_parse_failure(
        cls,
        trial_id: str,
        seed_doc: str,
        condition: str,
        turn: int,
        agent_id: str,
        level: int,
        previous_direction: str = "NEUTRAL",
    ) -> "AgentTurnRecord":
        """Create a record for a turn where JSON parsing failed.

        The previous stance is carried forward per the spec. The turn is
        flagged as a System Execution Error.
        """
        return cls(
            trial_id=trial_id,
            seed_doc=seed_doc,
            condition=condition,
            turn=turn,
            agent_id=agent_id,
            level=level,
            prediction_direction=previous_direction,
            confidence=0.0,
            prediction_summary="[PARSE FAILURE — previous stance carried forward]",
            key_factors=[],
            parse_success=False,
            trail_category="system_execution_error",
        )


class JSONLExporter:
    """Appends AgentTurnRecord objects to a JSONL file during simulation.

    One JSONLExporter instance per trial. Call record() after each agent turn.
    Call close() when the trial ends (or use as a context manager).
    """

    def __init__(self, output_path: Path) -> None:
        """Open the JSONL file for writing.

        Args:
            output_path: Full path including filename (e.g. data/flat_baseline/
                tech_earnings/trial_001/trace.jsonl). Parent directories are
                created automatically.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = output_path
        self._fh = output_path.open("a", encoding="utf-8")

    def record(self, entry: AgentTurnRecord) -> None:
        """Append one agent-turn record to the JSONL file."""
        row = asdict(entry)
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLExporter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
