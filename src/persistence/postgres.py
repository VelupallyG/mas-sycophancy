"""Lightweight raw-SQL Postgres client for local experiment persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.persistence.records import AgentMessageRecord, ExperimentRunRecord

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class PostgresPersistence:
    """Minimal local Postgres persistence client.

    The import of psycopg is intentionally delayed so JSONL-only runs do not need
    the Postgres dependency installed.
    """

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required when DB persistence is enabled.")
        self._database_url = database_url
        self._conn: Any | None = None

    def __enter__(self) -> "PostgresPersistence":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def connect(self) -> None:
        """Open the Postgres connection if needed."""
        if self._conn is not None:
            return
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "DB persistence requires psycopg. Install it with "
                '`pip install "psycopg[binary]"` or disable --enable-db.'
            ) from exc
        self._conn = psycopg.connect(self._database_url)

    def close(self) -> None:
        """Close the connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        """Create the minimal tables if they do not exist."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()

    def upsert_seed_document(
        self,
        *,
        seed_id: str,
        file_name: str,
        domain: str,
        target_entity: str,
        ground_truth_direction: str,
        full_json: dict[str, Any],
    ) -> None:
        """Insert or update a seed document snapshot."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO seed_documents (
                    id,
                    file_name,
                    domain,
                    target_entity,
                    ground_truth_direction,
                    full_json
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    domain = EXCLUDED.domain,
                    target_entity = EXCLUDED.target_entity,
                    ground_truth_direction = EXCLUDED.ground_truth_direction,
                    full_json = EXCLUDED.full_json,
                    updated_at = NOW()
                """,
                (
                    seed_id,
                    file_name,
                    domain,
                    target_entity,
                    ground_truth_direction,
                    json.dumps(full_json, ensure_ascii=False),
                ),
            )
        conn.commit()

    def create_run(self, record: ExperimentRunRecord) -> None:
        """Insert a new experiment run row."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_runs (
                    id,
                    seed_id,
                    topology,
                    condition,
                    trial_id,
                    rerun_id
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    record.run_id,
                    record.seed_id,
                    record.topology,
                    record.condition,
                    record.trial_id,
                    record.rerun_id,
                ),
            )
        conn.commit()

    def log_agent_message(self, record: AgentMessageRecord) -> None:
        """Persist one agent output or routed observation."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_messages (
                    run_id,
                    agent_name,
                    agent_role,
                    round_number,
                    message_type,
                    content_json
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    record.run_id,
                    record.agent_name,
                    record.agent_role,
                    record.round_number,
                    record.message_type,
                    json.dumps(record.content_json, ensure_ascii=False),
                ),
            )
        conn.commit()

    def finalize_run(
        self,
        *,
        run_id: str,
        final_decision: str,
        final_confidence: float,
        correct: bool,
    ) -> None:
        """Write final decision fields for one experiment run."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE experiment_runs
                SET
                    finalized_at = NOW(),
                    final_decision = %s,
                    final_confidence = %s,
                    correct = %s
                WHERE id = %s
                """,
                (final_decision, final_confidence, correct, run_id),
            )
        conn.commit()

    def _connection(self) -> Any:
        self.connect()
        if self._conn is None:
            raise RuntimeError("Postgres connection was not opened.")
        return self._conn
