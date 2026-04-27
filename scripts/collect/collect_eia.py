#!/usr/bin/env python3
"""Collect oil and energy data from the EIA Open Data API.

The U.S. Energy Information Administration provides free access to
petroleum prices, supply, demand, inventory, and production data.

API docs: https://www.eia.gov/opendata/documentation.php
Register for a free key: https://www.eia.gov/opendata/register.php

Usage:
    EIA_API_KEY="your_key" python scripts/collect/collect_eia.py \
        --preset geopolitics_oil \
        --output-dir local_evidence/geopolitics \
        --start 2024-01 \
        --end 2025-12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


EIA_API_V2 = "https://api.eia.gov/v2"


def _evidence_id(series_id: str, period: str) -> str:
    h = hashlib.sha256(f"eia:{series_id}:{period}".encode()).hexdigest()[:12]
    return f"eia_{h}"


def fetch_series(
    route: str,
    api_key: str,
    *,
    frequency: str = "monthly",
    start: str | None = None,
    end: str | None = None,
    sort_col: str = "period",
    sort_dir: str = "desc",
    length: int = 60,
    facets: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Fetch data from an EIA API v2 route.

    Args:
        route: API route like "petroleum/pri/spt/data" (spot prices)
        facets: Filter dict, e.g. {"product": ["EPCBRENT"]}
    """
    params: dict[str, str] = {
        "api_key": api_key,
        "frequency": frequency,
        "data[0]": "value",
        "sort[0][column]": sort_col,
        "sort[0][direction]": sort_dir,
        "length": str(length),
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if facets:
        for key, values in facets.items():
            for i, v in enumerate(values):
                params[f"facets[{key}][]"] = v  # EIA wants repeated keys

    url = f"{EIA_API_V2}/{route}?{urlencode(params, doseq=True)}"
    print(f"[EIA] Fetching: {url[:200]}...")

    req = Request(url, headers={"User-Agent": "mas-sycophancy-collector/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"[EIA] HTTP error {e.code}: {e.reason}", file=sys.stderr)
        return []

    rows = data.get("response", {}).get("data", [])
    print(f"[EIA] Got {len(rows)} data points")
    return rows


# ---------------------------------------------------------------------------
# Preset series definitions for geopolitics/oil seed
# ---------------------------------------------------------------------------

GEOPOLITICS_OIL_SERIES = [
    {
        "name": "Brent Crude Spot Price",
        "route": "petroleum/pri/spt/data",
        "facets": {"product": ["EPCBRENT"]},
        "frequency": "monthly",
        "description": "Monthly Brent crude oil spot price ($/barrel)",
        "value_label": "price_usd_per_barrel",
    },
    {
        "name": "WTI Crude Spot Price",
        "route": "petroleum/pri/spt/data",
        "facets": {"product": ["EPCWTI"]},
        "frequency": "monthly",
        "description": "Monthly WTI crude oil spot price ($/barrel)",
        "value_label": "price_usd_per_barrel",
    },
    {
        "name": "US Crude Oil Imports",
        "route": "petroleum/move/imp/data",
        "facets": {"product": ["EPC0"]},
        "frequency": "monthly",
        "description": "Monthly US crude oil imports (thousand barrels)",
        "value_label": "thousand_barrels",
    },
    {
        "name": "OECD Commercial Oil Inventory",
        "route": "international/data",
        "facets": {"activityId": ["2"], "productId": ["54"]},
        "frequency": "monthly",
        "description": "OECD commercial petroleum inventory levels",
        "value_label": "million_barrels",
    },
    {
        "name": "World Crude Oil Production",
        "route": "international/data",
        "facets": {"activityId": ["1"], "productId": ["57"]},
        "frequency": "monthly",
        "description": "World crude oil + condensate production",
        "value_label": "thousand_barrels_per_day",
    },
]


def series_to_evidence_docs(
    rows: list[dict],
    series_def: dict,
    seed_id: str,
    *,
    group_size: int = 6,
) -> list[dict]:
    """Convert EIA data rows into evidence documents.

    Groups `group_size` consecutive data points into one evidence doc
    to provide trend context rather than isolated snapshots.
    """
    docs = []
    # Sort by period ascending for chronological grouping
    rows_sorted = sorted(rows, key=lambda r: r.get("period", ""))

    for i in range(0, len(rows_sorted), group_size):
        chunk = rows_sorted[i : i + group_size]
        if not chunk:
            continue

        periods = [r.get("period", "?") for r in chunk]
        values = []
        for r in chunk:
            val = r.get("value")
            if val is not None:
                values.append(f"  {r.get('period', '?')}: {val}")

        date_range = f"{periods[0]} to {periods[-1]}"
        text_lines = [
            f"{series_def['name']} — {series_def['description']}",
            f"Period: {date_range}",
            f"Data ({series_def['value_label']}):",
            *values,
        ]

        doc = {
            "id": _evidence_id(series_def["route"], date_range),
            "seed_id": seed_id,
            "source_type": "market_data",
            "source_name": "U.S. Energy Information Administration (EIA)",
            "entity": "Brent Crude Oil Futures",
            "ticker": "BZ",
            "document_date": periods[-1] if periods else None,
            "title": f"{series_def['name']} ({date_range})",
            "text_content": "\n".join(text_lines),
            "full_json": {
                "eia_route": series_def["route"],
                "frequency": series_def["frequency"],
                "raw_data": chunk,
                "notes": "Auto-collected from EIA Open Data API v2.",
            },
        }
        docs.append(doc)

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect energy data from EIA Open Data API"
    )
    parser.add_argument(
        "--preset",
        choices=["geopolitics_oil"],
        default="geopolitics_oil",
        help="Preset series set (default: geopolitics_oil)",
    )
    parser.add_argument(
        "--seed-id",
        default="iran_oil_sanctions_tightening_march_2025",
    )
    parser.add_argument(
        "--output-dir",
        default="local_evidence/geopolitics",
    )
    parser.add_argument("--start", default="2023-01", help="Start period YYYY-MM")
    parser.add_argument("--end", default="2025-12", help="End period YYYY-MM")
    parser.add_argument(
        "--group-size",
        type=int,
        default=6,
        help="Months per evidence doc (default 6)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EIA_API_KEY"),
        help="EIA API key (or set EIA_API_KEY env var)",
    )

    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: EIA API key required. Set EIA_API_KEY env var or pass --api-key.",
            file=sys.stderr,
        )
        print("Register free at: https://www.eia.gov/opendata/register.php")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_defs = GEOPOLITICS_OIL_SERIES
    written = 0

    for series_def in series_defs:
        rows = fetch_series(
            series_def["route"],
            args.api_key,
            frequency=series_def["frequency"],
            start=args.start,
            end=args.end,
            facets=series_def.get("facets"),
        )
        if not rows:
            continue

        docs = series_to_evidence_docs(
            rows, series_def, args.seed_id, group_size=args.group_size
        )
        for doc in docs:
            out_path = output_dir / f"{doc['id']}.json"
            with open(out_path, "w") as f:
                json.dump(doc, f, indent=2)
            written += 1

    print(f"[EIA] Wrote {written} evidence files to {output_dir}")


if __name__ == "__main__":
    main()
