"""Run TRAIL categorisation over trace JSONL files.

This script classifies failed agent-turns into TRAIL categories using either:
  - LLM judge (Vertex AI, temperature=0.0), or
  - Heuristic fallback from src.metrics.trail.categorise_failure.

Usage:
  python -m analysis.evaluate_trail --data-dir data --output data/trail_eval.jsonl
  python -m analysis.evaluate_trail --data-dir data --use-llm-judge
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.metrics.trail import categorise_failure, categorise_failure_with_llm
from src.metrics.trail_judge import VertexAITrailJudge
from src.tasks.predictive_intel import extract_ground_truth_direction

logger = logging.getLogger(__name__)


def _load_seed_docs() -> dict[str, dict]:
    seed_dir = (
        Path(__file__).resolve().parent.parent / "src" / "tasks" / "seed_documents"
    )
    out: dict[str, dict] = {}
    for seed_path in sorted(seed_dir.glob("*.json")):
        with seed_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        out[seed_path.stem] = payload
        metadata = payload.get("metadata", {})
        metadata_id = metadata.get("id")
        if isinstance(metadata_id, str) and metadata_id:
            out[metadata_id] = payload
    return out


def _resolve_seed_doc_payload(
    seed_docs: dict[str, dict], seed_key: object
) -> dict | None:
    if not isinstance(seed_key, str) or not seed_key:
        return None
    payload = seed_docs.get(seed_key)
    if payload is not None:
        return payload
    # Backward compatibility for older trace rows that used only the metadata id.
    for candidate in seed_docs.values():
        metadata = candidate.get("metadata", {}) if isinstance(candidate, dict) else {}
        if metadata.get("id") == seed_key:
            return candidate
    return None


def _iter_trace_rows(data_dir: Path):
    for trace_path in sorted(data_dir.rglob("trace.jsonl")):
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield trace_path, json.loads(line)


def run(args: argparse.Namespace) -> None:
    seed_docs = _load_seed_docs()

    judge = None
    if args.use_llm_judge:
        judge = VertexAITrailJudge(
            model_id=args.trail_judge_model_id,
            project=args.gcp_project,
            location=args.gcp_location,
            requests_per_minute=args.rate_limit_rpm,
            temperature=0.0,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for trace_path, row in _iter_trace_rows(Path(args.data_dir)):
            if not row.get("parse_success", False):
                continue

            seed_key = row.get("seed_doc")
            seed_doc = _resolve_seed_doc_payload(seed_docs, seed_key)
            if seed_doc is None:
                logger.warning(
                    "Skipping row with unknown seed_doc=%r in %s", seed_key, trace_path
                )
                continue

            ground_truth = extract_ground_truth_direction(seed_doc)
            if not isinstance(ground_truth, str) or not ground_truth:
                logger.warning(
                    "Skipping row with invalid ground_truth for seed_doc=%r in %s",
                    seed_key,
                    trace_path,
                )
                continue
            prediction_direction = row.get("prediction_direction")
            if prediction_direction == ground_truth:
                continue

            agent_output = {
                "prediction_direction": prediction_direction,
                "predicted_magnitude": row.get("predicted_magnitude", "MEDIUM"),
                "predicted_price_change_pct": row.get(
                    "predicted_price_change_pct", 0.0
                ),
                "prediction_summary": row.get("prediction_summary", ""),
                "key_factors": row.get("key_factors", []),
            }

            if judge is not None:
                category = categorise_failure_with_llm(
                    agent_output=agent_output,
                    seed_doc=seed_doc,
                    ground_truth_direction=ground_truth,
                    judge_fn=judge.judge,
                )
            else:
                category = categorise_failure(
                    agent_output=agent_output, seed_doc=seed_doc
                )

            out_row = {
                "trial_id": row.get("trial_id"),
                "seed_doc": seed_key,
                "condition": row.get("condition"),
                "turn": row.get("turn"),
                "agent_id": row.get("agent_id"),
                "prediction_direction": prediction_direction,
                "ground_truth_direction": ground_truth,
                "trail_category": category,
            }
            out.write(json.dumps(out_row, ensure_ascii=True) + "\n")
            rows_written += 1

    logger.info("Wrote %d TRAIL rows to %s", rows_written, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TRAIL categories from trace JSONL files"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/trail_eval.jsonl")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--gcp-project", default="")
    parser.add_argument("--gcp-location", default="us-central1")
    parser.add_argument("--trail-judge-model-id", default="gemini-2.5-flash")
    parser.add_argument("--rate-limit-rpm", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
