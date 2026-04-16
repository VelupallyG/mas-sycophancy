"""Import locally collected evidence JSON files into Postgres.

Usage:
    DATABASE_URL="postgresql://localhost/mas_sycophancy" \
    python scripts/import_evidence.py --path local_evidence
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.persistence import EvidenceDocumentRecord, PostgresPersistence


def _iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.json"))


def _record_from_payload(payload: dict[str, Any]) -> EvidenceDocumentRecord:
    missing = {
        field
        for field in ("id", "source_type", "source_name", "title", "text_content")
        if not payload.get(field)
    }
    if missing:
        raise ValueError(f"Evidence document missing required fields: {missing}")

    return EvidenceDocumentRecord(
        evidence_id=str(payload["id"]),
        seed_id=payload.get("seed_id"),
        source_type=str(payload["source_type"]),
        source_name=str(payload["source_name"]),
        entity=payload.get("entity"),
        ticker=payload.get("ticker"),
        document_date=payload.get("document_date"),
        title=str(payload["title"]),
        text_content=str(payload["text_content"]),
        full_json=payload.get("full_json", {}),
    )


def import_evidence(path: Path, database_url: str) -> int:
    client = PostgresPersistence(database_url)
    client.init_schema()

    count = 0
    try:
        for json_path in _iter_json_files(path):
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            record = _record_from_payload(payload)
            client.upsert_evidence_document(record)
            count += 1
    finally:
        client.close()

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import local evidence JSON files.")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection URL. Defaults to DATABASE_URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("Set DATABASE_URL or pass --database-url.")

    count = import_evidence(args.path, args.database_url)
    print(f"Imported {count} evidence document(s).")


if __name__ == "__main__":
    main()
