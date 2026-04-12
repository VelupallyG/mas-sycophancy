"""Prepare the local SQLite financial database from raw CSV files.

Run once before experiments with --enable-tools:
    python scripts/prepare_financial_db.py

Idempotent: drops and recreates all tables on each run.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("data/financial_db/market_data.db")
RAW_DIR = Path("data/financial_db/raw")

SCHEMA = """
DROP TABLE IF EXISTS stock_prices;
CREATE TABLE stock_prices (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    PRIMARY KEY (ticker, date)
);

DROP TABLE IF EXISTS earnings;
CREATE TABLE earnings (
    ticker          TEXT NOT NULL,
    quarter         TEXT NOT NULL,
    revenue_actual  REAL,
    revenue_est     REAL,
    eps_actual      REAL,
    eps_est         REAL,
    operating_income REAL,
    capex           REAL,
    free_cash_flow  REAL,
    PRIMARY KEY (ticker, quarter)
);

DROP TABLE IF EXISTS macro_indicators;
CREATE TABLE macro_indicators (
    indicator   TEXT NOT NULL,
    date        TEXT NOT NULL,
    value       REAL NOT NULL,
    PRIMARY KEY (indicator, date)
);

DROP TABLE IF EXISTS entity_registry;
CREATE TABLE entity_registry (
    seed_doc_id TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    entity_name TEXT,
    sector      TEXT,
    PRIMARY KEY (seed_doc_id, ticker)
);
"""


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)

    # stock_prices
    for row in _load_csv(RAW_DIR / "stock_prices.csv"):
        conn.execute(
            "INSERT INTO stock_prices (ticker,date,open,high,low,close,volume) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                row["ticker"],
                row["date"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            ),
        )

    # earnings
    for row in _load_csv(RAW_DIR / "earnings.csv"):
        conn.execute(
            "INSERT INTO earnings "
            "(ticker,quarter,revenue_actual,revenue_est,eps_actual,eps_est,"
            "operating_income,capex,free_cash_flow) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row["ticker"],
                row["quarter"],
                float(row["revenue_actual"]),
                float(row["revenue_est"]),
                float(row["eps_actual"]),
                float(row["eps_est"]),
                float(row["operating_income"]),
                float(row["capex"]),
                float(row["free_cash_flow"]),
            ),
        )

    # macro_indicators
    for row in _load_csv(RAW_DIR / "macro_indicators.csv"):
        conn.execute(
            "INSERT INTO macro_indicators (indicator,date,value) VALUES (?,?,?)",
            (row["indicator"], row["date"], float(row["value"])),
        )

    # entity_registry
    for row in _load_csv(RAW_DIR / "entity_registry.csv"):
        conn.execute(
            "INSERT INTO entity_registry (seed_doc_id,ticker,entity_name,sector) "
            "VALUES (?,?,?,?)",
            (
                row["seed_doc_id"],
                row["ticker"],
                row["entity_name"],
                row["sector"],
            ),
        )

    conn.commit()

    # Print summary
    for table in ("stock_prices", "earnings", "macro_indicators", "entity_registry"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    main()
