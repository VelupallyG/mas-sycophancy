# Vertex AI Setup and Validation

This guide gets you from a working `--mock` run to real Vertex AI runs with rate limiting, plus TRAIL evaluation with an LLM judge.

## 1) Prerequisites

- Python environment installed for this repo
- Google Cloud project with Vertex AI enabled
- IAM permission for your account/service account to call Vertex AI
- `gcloud` CLI installed

## 2) Authenticate and set project

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

If you use a service account locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/key.json
```

## 3) Configure environment variables

Copy `.env.example` and set values:

```bash
cp .env.example .env
```

Required:

- `GCP_PROJECT`

Recommended defaults:

- `GCP_LOCATION=us-central1`
- `GEMINI_MODEL_ID=gemini-2.5-flash-002`
- `RATE_LIMIT_RPM=60`

Optional TRAIL judge settings:

- `TRAIL_USE_LLM_JUDGE=false`
- `TRAIL_JUDGE_MODEL_ID=gemini-2.5-flash-002`
- `TRAIL_JUDGE_TEMPERATURE=0.0`

Load env vars into your shell:

```bash
set -a; source .env; set +a
```

## 4) Quick API smoke test (real Vertex)

Run the Concordia/Vertex spike script without `--mock`:

```bash
python scripts/spike_concordia_vertex.py
```

Expected result:

- Script prints a valid JSON prediction from an agent
- Observation-routing proof runs end-to-end

## 5) Run a minimal real trial

Start with 1 trial to verify end-to-end path:

```bash
python -m experiments.run_flat_baseline --seed-doc finance_earnings --n-trials 1
python -m experiments.run_hierarchical --seed-doc finance_earnings --n-trials 1
```

Then run the suite on small settings first:

```bash
python -m experiments.run_full_suite --n-trials 1
```

## 6) Rate limiting behavior

The runtime uses a shared synchronous limiter in `src/rate_limiter.py`:

- Global per-process limiter via `get_shared_rate_limiter()`
- Ceiling from `RATE_LIMIT_RPM` (default 60)
- Retry on 429/503 with exponential backoff:
  - start: 1s
  - multiplier: 2x
  - cap: 60s
  - max attempts: 5

If you still see repeated quota errors, lower `RATE_LIMIT_RPM` (for example 30).

## 7) TRAIL with LLM judge (post-hoc)

TRAIL LLM categorization is run as a post-processing step over generated `trace.jsonl` files:

```bash
python -m analysis.evaluate_trail --data-dir data --output data/trail_eval.jsonl --use-llm-judge --gcp-project "$GCP_PROJECT"
```

Notes:

- Judge decoding is deterministic (`temperature=0.0`)
- Judge output must be strict JSON category
- On judge failure/invalid output, code falls back to heuristic categorization

## 8) Raw conversation logging for TRAIL

For full trace-level TRAIL analysis, each trial now writes both:

- Structured stance trace: `data/<condition>/<seed>/trial_xxx/trace.jsonl`
- Raw routed conversation trace: `data/raw_traces/<condition>/<seed>/trial_xxx/conversation.jsonl`

For flat hallucination reruns, raw traces are under:

- `data/raw_traces/flat_hallucination/<seed>/trial_xxx/rerun_k/conversation.jsonl`

The raw conversation trace includes:

- Every routed observation (`event_type="observation"`)
- Every agent output (`event_type="agent_output"`)
- Sender/receiver IDs and turn index

This is enough context for post-hoc LLM judging and auditability.

## 9) Common failure modes

- `ValueError: gcp_project must be set`: export `GCP_PROJECT`
- Auth errors: rerun `gcloud auth application-default login`
- 429 quota errors: reduce `RATE_LIMIT_RPM`, retry later
- Model not found in region: verify `GCP_LOCATION` and model availability
