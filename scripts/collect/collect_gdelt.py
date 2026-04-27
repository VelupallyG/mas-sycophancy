#!/usr/bin/env python3
"""Collect news articles from the GDELT DOC 2.0 API.

GDELT indexes global news in near-real-time. The DOC API provides
full-text search with no authentication required.

API docs: https://blog.gdeltproject.org/gdelt-doc-2-0-api-unveiled/

Usage:
    python scripts/collect/collect_gdelt.py \
        --query "oil sanctions shipping crude supply" \
        --seed-id iran_oil_sanctions_tightening_march_2025 \
        --output-dir local_evidence/geopolitics \
        --max-records 50 \
        --start-date 2024-01-01 \
        --end-date 2025-12-31
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT rate limit: be polite, wait between requests
REQUEST_DELAY_S = 3.0
RATE_LIMIT_BACKOFF_S = 30


def _make_evidence_id(source: str, url: str) -> str:
    """Deterministic ID from source + URL."""
    h = hashlib.sha256(f"{source}:{url}".encode()).hexdigest()[:12]
    return f"gdelt_{h}"


def _gdelt_date(d: str) -> str:
    """Convert YYYY-MM-DD to GDELT format YYYYMMDDHHMMSS."""
    dt = datetime.strptime(d, "%Y-%m-%d")
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_articles(
    query: str,
    *,
    max_records: int = 50,
    start_date: str | None = None,
    end_date: str | None = None,
    source_country: str | None = None,
    sort: str = "DateDesc",
) -> list[dict]:
    """Fetch articles from GDELT DOC API.

    Returns a list of article dicts with keys: url, title, seendate,
    domain, language, sourcecountry.
    """
    params: dict[str, str] = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "format": "json",
        "sort": sort,
    }
    if start_date:
        params["startdatetime"] = _gdelt_date(start_date)
    if end_date:
        params["enddatetime"] = _gdelt_date(end_date)
    if source_country:
        params["sourcecountry"] = source_country

    url = f"{GDELT_DOC_API}?{urlencode(params)}"
    print(f"[GDELT] Fetching: {url}")

    req = Request(url, headers={"User-Agent": "mas-sycophancy-collector/1.0"})
    data: dict = {}

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 429:
            for attempt in range(3):
                wait = RATE_LIMIT_BACKOFF_S * (attempt + 1)
                print(
                    f"[GDELT] Rate limited, waiting {wait}s (attempt {attempt + 1}/3)..."
                )
                time.sleep(wait)
                try:
                    with urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except HTTPError as e2:
                    if e2.code != 429 or attempt == 2:
                        print(
                            f"[GDELT] Failed after retries: {e2.code}", file=sys.stderr
                        )
                        return []
        else:
            print(f"[GDELT] HTTP error {e.code}: {e.reason}", file=sys.stderr)
            return []
    except URLError as e:
        print(f"[GDELT] URL error: {e.reason}", file=sys.stderr)
        return []

    articles = data.get("articles", [])
    print(f"[GDELT] Got {len(articles)} articles")
    return articles


def fetch_article_text(url: str, max_chars: int = 3000) -> str | None:
    """Best-effort extraction of article text from a URL.

    Uses stdlib only — strips HTML tags and extracts <p> content.
    Returns None if the fetch fails (many sites block scraping).
    """
    import re
    from html.parser import HTMLParser

    class ParagraphExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._in_p = False
            self._in_script = False
            self.paragraphs: list[str] = []
            self._buf: list[str] = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in ("script", "style", "nav", "header", "footer"):
                self._in_script = True
            elif tag == "p" and not self._in_script:
                self._in_p = True
                self._buf = []

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "nav", "header", "footer"):
                self._in_script = False
            elif tag == "p" and self._in_p:
                self._in_p = False
                text = " ".join(self._buf).strip()
                # Filter out very short paragraphs (nav items, buttons, etc.)
                if len(text) > 40:
                    self.paragraphs.append(text)

        def handle_data(self, data: str) -> None:
            if self._in_p and not self._in_script:
                self._buf.append(data.strip())

    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; academic-research/1.0)",
            },
        )
        with urlopen(req, timeout=10) as resp:
            # Only process HTML
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct.lower() and "text" not in ct.lower():
                return None
            raw = resp.read(500_000).decode("utf-8", errors="replace")
    except Exception:
        return None

    parser = ParagraphExtractor()
    try:
        parser.feed(raw)
    except Exception:
        return None

    if not parser.paragraphs:
        return None

    text = "\n\n".join(parser.paragraphs)
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def article_to_evidence(
    article: dict,
    seed_id: str,
    entity: str | None = None,
    ticker: str | None = None,
    scrape_text: bool = True,
) -> dict:
    """Convert a GDELT article dict to local_evidence JSON format."""
    url = article.get("url", "")
    title = article.get("title", "No title")
    seen_date = article.get("seendate", "")
    domain = article.get("domain", "unknown")
    source_country = article.get("sourcecountry", "")
    language = article.get("language", "")

    # Parse date: GDELT returns "YYYYMMDDTHHMMSSZ" format
    doc_date = None
    if seen_date:
        try:
            dt = datetime.strptime(seen_date, "%Y%m%dT%H%M%SZ")
            doc_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            doc_date = seen_date[:10] if len(seen_date) >= 10 else None

    # Try to get article body; fall back to title-only
    body = fetch_article_text(url) if (url and scrape_text) else None
    has_full_text = body is not None
    if body:
        text_content = f"{title}\n\n{body}"
    else:
        text_content = title

    return {
        "id": _make_evidence_id("gdelt", url),
        "seed_id": seed_id,
        "source_type": "news_article",
        "source_name": f"GDELT / {domain}",
        "entity": entity,
        "ticker": ticker,
        "document_date": doc_date,
        "title": title,
        "text_content": text_content,
        "full_json": {
            "url": url,
            "domain": domain,
            "source_country": source_country,
            "language": language,
            "gdelt_seendate": seen_date,
            "notes": "Auto-collected via GDELT DOC API. Review and enrich text_content with real excerpts if possible."
            if not has_full_text
            else "Auto-collected via GDELT DOC API with scraped article text.",
        },
    }


def run_collection(
    queries: list[str],
    seed_id: str,
    output_dir: Path,
    *,
    max_per_query: int = 25,
    start_date: str | None = None,
    end_date: str | None = None,
    entity: str | None = None,
    ticker: str | None = None,
    scrape_text: bool = True,
) -> list[Path]:
    """Run collection across multiple queries, deduplicate, write files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    written: list[Path] = []

    for query in queries:
        articles = fetch_articles(
            query,
            max_records=max_per_query,
            start_date=start_date,
            end_date=end_date,
        )
        time.sleep(REQUEST_DELAY_S)

        for article in articles:
            url = article.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            evidence = article_to_evidence(
                article,
                seed_id,
                entity=entity,
                ticker=ticker,
                scrape_text=scrape_text,
            )
            out_path = output_dir / f"{evidence['id']}.json"
            with open(out_path, "w") as f:
                json.dump(evidence, f, indent=2)
            written.append(out_path)

            if scrape_text:
                # Be polite to article hosts
                time.sleep(0.5)

    print(f"[GDELT] Wrote {len(written)} evidence files to {output_dir}")
    return written


# ---------------------------------------------------------------------------
# Preset query sets for known seeds
# ---------------------------------------------------------------------------

GEOPOLITICS_OIL_QUERIES = [
    "oil sanctions shipping crude supply disruption",
    "sanctions maritime insurance tanker oil",
    "G7 sanctions oil exports enforcement",
    "crude oil supply shock sanctions",
    "oil sanctions shadow fleet shipping",
    "sanctions crude tanker insurance reinsurance",
    "OPEC spare capacity sanctions",
    "oil price sanctions geopolitical risk",
    "sanctions oil export ban shipping halt",
    "crude oil strategic petroleum reserve sanctions",
    # Iran-specific queries
    "Iran sanctions oil exports OFAC",
    "Shandong teapot refinery Iran crude",
    "Iran shadow fleet tanker sanctions",
    "Iran oil China teapot refinery",
    "maximum pressure Iran oil revenue",
    "Iran crude exports shadow shipping insurance",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect news articles from GDELT DOC API"
    )
    parser.add_argument(
        "--query",
        nargs="*",
        help="Search queries (space-separated). If omitted, uses preset for --preset.",
    )
    parser.add_argument(
        "--preset",
        choices=["geopolitics_oil"],
        default=None,
        help="Use a preset query set for a known seed document.",
    )
    parser.add_argument(
        "--seed-id",
        default="iran_oil_sanctions_tightening_march_2025",
        help="seed_id to tag evidence with",
    )
    parser.add_argument(
        "--output-dir",
        default="local_evidence/geopolitics",
        help="Output directory for evidence JSON files",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=25,
        help="Max articles per query (default 25)",
    )
    parser.add_argument("--start-date", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date YYYY-MM-DD")
    parser.add_argument("--entity", default="Brent Crude Oil Futures")
    parser.add_argument("--ticker", default="BZ")
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip article text scraping (faster, title-only evidence)",
    )

    args = parser.parse_args()

    if args.query:
        queries = args.query
    elif args.preset == "geopolitics_oil" or (not args.query and not args.preset):
        queries = GEOPOLITICS_OIL_QUERIES
        print(f"[GDELT] Using geopolitics_oil preset ({len(queries)} queries)")
    else:
        parser.error("Provide --query or --preset")

    run_collection(
        queries=queries,
        seed_id=args.seed_id,
        output_dir=Path(args.output_dir),
        max_per_query=args.max_per_query,
        start_date=args.start_date,
        end_date=args.end_date,
        entity=args.entity,
        ticker=args.ticker,
        scrape_text=not args.no_scrape,
    )


if __name__ == "__main__":
    main()
