# Evidence Collection Scripts

Semi-automated scripts for populating `local_evidence/` with real data.
Each script hits one data source, outputs JSON files in the local_evidence schema,
and requires manual review before import.

## Workflow

```bash
# 1. Run a collection script
python scripts/collect/collect_gdelt.py \
  --query "oil sanctions shipping crude" \
  --seed-id iran_oil_sanctions_tightening_march_2025 \
  --output-dir local_evidence/geopolitics

# 2. Review output, delete junk files

# 3. Import into Postgres
DATABASE_URL="postgresql://localhost/mas_sycophancy" \
python scripts/import_evidence.py --path local_evidence/geopolitics
```

## Available Scripts

| Script | Source | Auth | Relevant Seeds |
|--------|--------|------|----------------|
| `collect_gdelt.py` | GDELT DOC API (global news) | None | geopolitics |
| `collect_eia.py` | EIA Open Data API (energy) | Free key | geopolitics |

## API Keys

- **EIA**: Register at https://www.eia.gov/opendata/register.php — set `EIA_API_KEY` env var
- **GDELT**: No key needed
