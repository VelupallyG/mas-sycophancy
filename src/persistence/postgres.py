"""Lightweight raw-SQL Postgres client for local experiment persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.persistence.records import (
    AgentMessageRecord,
    AgentRetrievalRecord,
    EvidenceDocumentRecord,
    ExperimentRunRecord,
)

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

    def upsert_evidence_document(self, record: EvidenceDocumentRecord) -> None:
        """Insert or update one locally collected evidence document."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evidence_documents (
                    id,
                    seed_id,
                    source_type,
                    source_name,
                    entity,
                    ticker,
                    document_date,
                    title,
                    text_content,
                    full_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    seed_id = EXCLUDED.seed_id,
                    source_type = EXCLUDED.source_type,
                    source_name = EXCLUDED.source_name,
                    entity = EXCLUDED.entity,
                    ticker = EXCLUDED.ticker,
                    document_date = EXCLUDED.document_date,
                    title = EXCLUDED.title,
                    text_content = EXCLUDED.text_content,
                    full_json = EXCLUDED.full_json,
                    updated_at = NOW()
                """,
                (
                    record.evidence_id,
                    record.seed_id,
                    record.source_type,
                    record.source_name,
                    record.entity,
                    record.ticker,
                    record.document_date,
                    record.title,
                    record.text_content,
                    json.dumps(record.full_json or {}, ensure_ascii=False),
                ),
            )
        conn.commit()

    def search_evidence(
        self,
        *,
        query: str,
        seed_id: str | None = None,
        source_type: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search local evidence documents with simple Postgres text matching."""
        conn = self._connection()
        like_query = f"%{query}%"
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    seed_id,
                    source_type,
                    source_name,
                    entity,
                    ticker,
                    document_date::text,
                    title,
                    text_content,
                    full_json
                FROM evidence_documents
                WHERE
                    (%s IS NULL OR seed_id = %s)
                    AND (%s IS NULL OR source_type = %s)
                    AND (
                        title ILIKE %s
                        OR text_content ILIKE %s
                        OR COALESCE(entity, '') ILIKE %s
                        OR COALESCE(ticker, '') ILIKE %s
                    )
                ORDER BY document_date DESC NULLS LAST, id ASC
                LIMIT %s
                """,
                (
                    seed_id,
                    seed_id,
                    source_type,
                    source_type,
                    like_query,
                    like_query,
                    like_query,
                    like_query,
                    limit,
                ),
            )
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "seed_id": row[1],
                "source_type": row[2],
                "source_name": row[3],
                "entity": row[4],
                "ticker": row[5],
                "document_date": row[6],
                "title": row[7],
                "text_content": row[8],
                "full_json": row[9],
            }
            for row in rows
        ]

    def log_agent_retrieval(self, record: AgentRetrievalRecord) -> None:
        """Persist one local evidence lookup."""
        conn = self._connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_retrievals (
                    run_id,
                    agent_name,
                    round_number,
                    query,
                    result_ids
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    record.run_id,
                    record.agent_name,
                    record.round_number,
                    record.query,
                    json.dumps(record.result_ids, ensure_ascii=False),
                ),
            )
        conn.commit()

    def _connection(self) -> Any:
        self.connect()
        if self._conn is None:
            raise RuntimeError("Postgres connection was not opened.")
        return self._conn
