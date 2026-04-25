# Local Evidence Documents

Put locally collected evidence JSON files in this directory. These files are
ingested into Postgres before experiment runs. Experiment runs should query the
local Postgres evidence store only; do not call SEC, Crunchbase, market-data, or
news APIs during a run.

## File Format

Each evidence file is one JSON object:

```json
{
  "id": "googl_capex_2026_001",
  "seed_id": "tech_earnings_google_2026_detailed",
  "source_type": "sec_filing",
  "source_name": "SEC EDGAR",
  "entity": "Alphabet Inc.",
  "ticker": "GOOGL",
  "document_date": "2026-04-25",
  "title": "Alphabet quarterly filing excerpt on AI infrastructure capex",
  "text_content": "Short, relevant excerpt or analyst-curated summary...",
  "full_json": {
    "url": "optional original source URL",
    "notes": "optional metadata"
  }
}
```

Required fields:

- `id`
- `source_type`
- `source_name`
- `title`
- `text_content`

Recommended fields:

- `seed_id`: Use the seed metadata ID so retrieval can target one benchmark.
- `entity`
- `ticker`
- `document_date`
- `full_json`: Put source URLs, filing accession numbers, collection notes, or
  other provenance metadata here.

Current benchmark seed IDs:

- `tech_earnings_google_2026_detailed`
- `geopolitical_oil_sanctions_2025_detailed`

## Import Command

```bash
DATABASE_URL="postgresql://localhost/mas_sycophancy" \
python scripts/import_evidence.py --path local_evidence
```
