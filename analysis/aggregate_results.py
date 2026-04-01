"""Aggregate experiment results into a tidy DataFrame.

Reads all ``SimulationResult`` JSON files from the data directory, computes
per-run metrics (delta squared, ToF, NoF, TRAIL categories, deference counts), and
returns a single ``pd.DataFrame`` with one row per simulation run.

CLI usage::

    python -m analysis.aggregate_results --data-dir data/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics.flip_metrics import (
    AgentTurnRecord,
    compute_nof_total,
    compute_tof,
    parse_turn_records_from_logs,
)
from src.metrics.linguistic import count_deference_markers
from src.metrics.scoring import compute_score_from_dicts, score_to_grade
from src.metrics.trail import TrailCategory, categorize_failures


def _load_result_json(path: Path) -> dict[str, Any] | None:
    """Load and validate a single result JSON file."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "experiment_id" not in data:
        return None
    return data


def _build_trail_entries(
    turn_records: list[dict[str, Any]],
    ground_truth_direction: str,
) -> list[dict[str, Any]]:
    """Build per-agent TRAIL trace entries from turn records.

    Groups records by agent, extracts final stance and flip count, and
    returns one entry per agent suitable for ``categorize_failures``.
    """
    agents: dict[str, list[dict[str, Any]]] = {}
    for rec in turn_records:
        name = rec.get("agent_name", "")
        agents.setdefault(name, []).append(rec)

    trail_entries: list[dict[str, Any]] = []
    for agent_name, records in agents.items():
        records.sort(key=lambda r: r.get("turn", 0))
        final_stance = records[-1].get("stance", "")
        # Count stance reversals
        flips = 0
        prev_stance = None
        for r in records:
            s = r.get("stance", "")
            if prev_stance is not None and s != prev_stance:
                flips += 1
            prev_stance = s

        trail_entries.append({
            "agent_name": agent_name,
            "final_stance": final_stance,
            "ground_truth_direction": ground_truth_direction,
            "flip_count": flips,
            "deferred_to_authority": final_stance != ground_truth_direction and flips > 0,
        })
    return trail_entries


def _compute_mean_deference(turn_records: list[dict[str, Any]]) -> float:
    """Mean deference marker count per agent turn."""
    total = 0
    count = 0
    for rec in turn_records:
        text = rec.get("text", "")
        if isinstance(text, str) and text:
            total += count_deference_markers(text)
            count += 1
    return total / count if count > 0 else 0.0


def aggregate(data_dir: str | Path) -> pd.DataFrame:
    """Load all result JSON files and compute aggregate metrics.

    Args:
        data_dir: Directory containing ``SimulationResult`` JSON files
            produced by the experiment runners.

    Returns:
        DataFrame with columns: experiment_id, topology, seed_doc,
        condition, whistleblower_rank, accuracy, score, grade,
        delta_squared, mean_tof, total_nof, trail_reasoning_pct,
        trail_planning_pct, trail_system_pct, mean_deference_count.
    """
    data_path = Path(data_dir)
    result_files = sorted(data_path.glob("*_result.json"))

    rows: list[dict[str, Any]] = []

    for result_file in result_files:
        data = _load_result_json(result_file)
        if data is None:
            continue

        experiment_id = data["experiment_id"]
        condition = data.get("condition", "unknown")
        seed_doc = data.get("seed_doc", "")
        accuracy = data.get("accuracy", 0.0)
        metadata = data.get("metadata", {})
        topology = metadata.get("topology", "flat" if condition == "baseline" else "hierarchical")
        ground_truth = metadata.get("ground_truth_direction", "")
        turn_records = data.get("agent_turn_records", [])
        whistleblower_rank = data.get("whistleblower_rank", None)

        # Score and grade
        score = compute_score_from_dicts(turn_records)
        grade = score_to_grade(score)

        # ToF and NoF
        parsed_records = parse_turn_records_from_logs(turn_records, strict=False)
        tof = compute_tof(parsed_records)
        nof = compute_nof_total(parsed_records)

        # TRAIL breakdown
        trail_entries = _build_trail_entries(turn_records, ground_truth)
        trail_counts = categorize_failures(trail_entries)
        total_trail = sum(trail_counts.values())
        trail_reasoning_pct = (
            trail_counts[TrailCategory.REASONING] / total_trail * 100.0
            if total_trail > 0 else 0.0
        )
        trail_planning_pct = (
            trail_counts[TrailCategory.PLANNING] / total_trail * 100.0
            if total_trail > 0 else 0.0
        )
        trail_system_pct = (
            trail_counts[TrailCategory.SYSTEM_EXECUTION] / total_trail * 100.0
            if total_trail > 0 else 0.0
        )

        # Deference
        mean_deference = _compute_mean_deference(turn_records)

        rows.append({
            "experiment_id": experiment_id,
            "topology": topology,
            "condition": condition,
            "seed_doc": seed_doc,
            "whistleblower_rank": whistleblower_rank,
            "accuracy": accuracy,
            "score": score,
            "grade": grade,
            "mean_tof": tof if not math.isnan(tof) else None,
            "total_nof": nof,
            "trail_reasoning_pct": trail_reasoning_pct,
            "trail_planning_pct": trail_planning_pct,
            "trail_system_pct": trail_system_pct,
            "mean_deference_count": mean_deference,
        })

    # Compute delta_squared: pair baseline vs influenced results by seed_doc
    baseline_acc: dict[str, float] = {}
    for row in rows:
        if row["condition"] == "baseline":
            baseline_acc[row["seed_doc"]] = row["accuracy"]

    for row in rows:
        seed = row["seed_doc"]
        if row["condition"] != "baseline" and seed in baseline_acc:
            row["delta_squared"] = baseline_acc[seed] - row["accuracy"]
        else:
            row["delta_squared"] = None

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment result JSONs.")
    parser.add_argument("--data-dir", default="data/", help="Input data directory.")
    parser.add_argument(
        "--output",
        default="data/aggregate_results.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = aggregate(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
