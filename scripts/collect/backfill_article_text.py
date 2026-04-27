#!/usr/bin/env python3
"""Backfill article body text into existing GDELT evidence files.

Reads GDELT evidence JSON files that have only title text, fetches the
article URL, extracts body text using trafilatura (with BeautifulSoup
fallback), and updates the evidence file in-place.

Usage:
    python scripts/collect/backfill_article_text.py \
        --evidence-dir local_evidence/geopolitics \
        --max-chars 3000 \
        --delay 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Extraction backends (trafilatura preferred, BS4 fallback)
# ---------------------------------------------------------------------------


def _extract_trafilatura(html: str, url: str) -> str | None:
    """Extract article text using trafilatura."""
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
        return text if text and len(text.strip()) > 80 else None
    except Exception:
        return None


def _extract_bs4(html: str) -> str | None:
    """Fallback: extract <p> tag text using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style/nav
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 40:
                paragraphs.append(text)
        result = "\n\n".join(paragraphs)
        return result if len(result) > 80 else None
    except Exception:
        return None


def fetch_and_extract(url: str, max_chars: int = 3000) -> tuple[str | None, str]:
    """Fetch URL and extract article text.

    Returns (text, method) where method is 'trafilatura', 'bs4', or 'failed'.
    """
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        with urlopen(req, timeout=15) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct.lower() and "text" not in ct.lower():
                return None, "not_html"
            raw = resp.read(500_000).decode("utf-8", errors="replace")
    except Exception as e:
        return None, f"fetch_error: {type(e).__name__}"

    # Try trafilatura first
    text = _extract_trafilatura(raw, url)
    if text:
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text, "trafilatura"

    # Fallback to BS4
    text = _extract_bs4(raw)
    if text:
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text, "bs4"

    return None, "no_content"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill article text into existing GDELT evidence files"
    )
    parser.add_argument(
        "--evidence-dir",
        default="local_evidence/geopolitics",
        help="Directory containing evidence JSON files",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Max characters of article text to keep (default 3000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between requests (default 1.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape even if text_content already has body text",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be scraped without writing",
    )
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    files = sorted(evidence_dir.glob("gdelt_*.json"))

    if not files:
        print(f"No GDELT evidence files found in {evidence_dir}")
        return

    stats = {"scraped": 0, "skipped": 0, "failed": 0, "already_has_text": 0}

    for i, fpath in enumerate(files):
        with open(fpath) as f:
            doc = json.load(f)

        url = doc.get("full_json", {}).get("url", "")
        title = doc.get("title", "?")[:60]

        # Skip if already has body text (unless --force)
        current_text = doc.get("text_content", "")
        if not args.force and len(current_text) > len(doc.get("title", "")) + 50:
            stats["already_has_text"] += 1
            continue

        if not url:
            stats["skipped"] += 1
            continue

        print(f"[{i + 1}/{len(files)}] {title}...")

        if args.dry_run:
            print(f"  Would fetch: {url[:80]}")
            continue

        text, method = fetch_and_extract(url, max_chars=args.max_chars)

        if text:
            doc["text_content"] = f"{doc.get('title', '')}\n\n{text}"
            doc["full_json"]["notes"] = (
                f"Auto-collected via GDELT DOC API with article text extracted by {method}."
            )
            with open(fpath, "w") as f:
                json.dump(doc, f, indent=2)
            stats["scraped"] += 1
            print(f"  OK ({method}, {len(text)} chars)")
        else:
            stats["failed"] += 1
            print(f"  FAILED ({method})")

        time.sleep(args.delay)

    print(
        f"\nDone: {stats['scraped']} scraped, {stats['failed']} failed, "
        f"{stats['already_has_text']} already had text, {stats['skipped']} skipped (no URL)"
    )


if __name__ == "__main__":
    main()
