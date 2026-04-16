"""Tests for the optional Postgres persistence client.

These tests use a fake connection and do not require a running Postgres server.
"""

from __future__ import annotations

from src.persistence import AgentMessageRecord, ExperimentRunRecord, PostgresPersistence


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self._connection.executed.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple | None]] = []
        self.commits = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def _client_with_fake_connection() -> tuple[PostgresPersistence, FakeConnection]:
    client = PostgresPersistence("postgresql://local/test")
    fake = FakeConnection()
    client._conn = fake  # noqa: SLF001 - intentional fake connection injection.
    return client, fake


def test_init_schema_executes_schema_sql():
    client, fake = _client_with_fake_connection()

    client.init_schema()

    assert "CREATE TABLE IF NOT EXISTS seed_documents" in fake.executed[0][0]
    assert fake.commits == 1


def test_upsert_seed_document_writes_full_json():
    client, fake = _client_with_fake_connection()

    client.upsert_seed_document(
        seed_id="seed_1",
        file_name="seed_1.json",
        domain="finance",
        target_entity="Alphabet Inc.",
        ground_truth_direction="NEGATIVE",
        full_json={"metadata": {"id": "seed_1"}},
    )

    sql, params = fake.executed[0]
    assert "INSERT INTO seed_documents" in sql
    assert params is not None
    assert params[0] == "seed_1"
    assert '"metadata"' in params[-1]


def test_create_log_and_finalize_run_use_expected_tables():
    client, fake = _client_with_fake_connection()

    client.create_run(
        ExperimentRunRecord(
            run_id="run_1",
            seed_id="seed_1",
            topology="flat",
            condition="flat_baseline",
            trial_id=0,
        )
    )
    client.log_agent_message(
        AgentMessageRecord(
            run_id="run_1",
            agent_name="peer_00",
            agent_role="PEER",
            round_number=1,
            message_type="agent_output",
            content_json={"prediction_direction": "NEGATIVE"},
        )
    )
    client.finalize_run(
        run_id="run_1",
        final_decision="NEGATIVE",
        final_confidence=0.8,
        correct=True,
    )

    executed_sql = "\n".join(sql for sql, _ in fake.executed)
    assert "INSERT INTO experiment_runs" in executed_sql
    assert "INSERT INTO agent_messages" in executed_sql
    assert "UPDATE experiment_runs" in executed_sql
    assert fake.commits == 3
