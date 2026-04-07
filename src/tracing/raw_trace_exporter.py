"""Raw conversation exporter for full TRAIL post-processing context.

Writes one JSON line per routed observation and one JSON line per agent action.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RawTraceRecord:
    trial_id: str
    condition: str
    turn: int
    event_type: str
    sender: str
    receiver: str
    content: str
    level: int | None = None
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class RawTraceExporter:
    """Append raw routed messages and outputs to JSONL."""

    def __init__(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = output_path.open("w", encoding="utf-8")

    def record(self, entry: RawTraceRecord) -> None:
        self._fh.write(json.dumps(asdict(entry), ensure_ascii=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RawTraceExporter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
