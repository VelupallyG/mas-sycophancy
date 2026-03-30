"""OpenTelemetry trace exporter for simulation runs.

Every agent action, memory update, GM state transition, and stance
evaluation is recorded as an OTel span.  At the end of a simulation run
the full trace is serialised to a structured JSON file in ``data/``.

The JSON schema matches what the metrics pipeline expects — each entry
includes at minimum: ``agent_name``, ``turn``, ``span_name``, ``timestamp``,
``attributes`` (a dict with event-specific fields), and optionally ``error``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class OtelExporter:
    """Thin wrapper around the OTel Python SDK for simulation tracing.

    Args:
        service_name: OTel service name (typically the experiment ID).
        export_path: Directory where JSON trace files are written.
    """

    def __init__(self, service_name: str, export_path: str | Path = "data/") -> None:
        self.service_name = service_name
        self.export_path = Path(export_path)

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager that wraps a simulation event in an OTel span.

        Args:
            name: Span name (e.g. ``"agent.act"``, ``"gm.state_transition"``).
            attributes: Key-value metadata to attach to the span.

        Yields:
            Nothing.  The span is automatically ended on context exit.
        """
        raise NotImplementedError
        yield  # pragma: no cover — stub only

    def export_trace(self, filename: str) -> Path:
        """Serialise the completed trace to a JSON file.

        Args:
            filename: Output file name (without directory prefix).  Written
                under ``self.export_path``.

        Returns:
            Absolute path of the written file.
        """
        raise NotImplementedError

    def get_spans(self) -> list[dict[str, Any]]:
        """Return all recorded spans as a list of dicts.

        Used by the metrics pipeline to process traces without reading
        from disk.

        Returns:
            List of span dicts in recording order.
        """
        raise NotImplementedError
