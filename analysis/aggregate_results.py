"""Aggregate experiment outputs into summary statistics.

Reads JSONL trace files produced by the simulation runner and computes:
  - Per-trial agent accuracy
  - Population-level Δ² (Sycophancy Effect Size)
  - ToF and NoF distributions

Usage:
  python -m analysis.aggregate_results --data-dir data/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.metrics.flip_metrics import compute_nof, compute_tof
from src.metrics.sycophancy_effect import (
    compute_agent_accuracy,
    compute_delta_squared,
)

logger = logging.getLogger(__name__)


def load_traces(data_dir: Path) -> pd.DataFrame:
    """Load all JSONL trace files from data_dir into a single DataFrame."""
    records: list[dict] = []
    for jsonl_path in sorted(data_dir.rglob("trace.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        logger.warning("No trace files found in %s", data_dir)
        return pd.DataFrame()
    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame, ground_truth: str) -> dict:
    """Compute summary statistics for one condition + seed_doc combination.

    Args:
        df: DataFrame filtered to one condition and seed_doc.
        ground_truth: The correct prediction direction.

    Returns:
        Dict with keys: mean_accuracy, mean_tof, mean_nof, n_trials, n_agents.
    """
    if df.empty:
        return {}

    # Group by trial and agent to get per-agent stance series.
    agent_trial_groups = df.groupby(["trial_id", "agent_id"])
    accuracies: list[float] = []
    tofs: list[int] = []
    nofs: list[int] = []

    for (trial_id, agent_id), group in agent_trial_groups:
        stances = group.sort_values("turn")["prediction_direction"].tolist()
        accuracies.append(compute_agent_accuracy(stances, ground_truth))
        tofs.append(compute_tof(stances, ground_truth))
        nofs.append(compute_nof(stances))

    n = len(accuracies)
    return {
        "mean_accuracy": sum(accuracies) / n if n else 0.0,
        "mean_tof": sum(tofs) / n if n else 0.0,
        "mean_nof": sum(nofs) / n if n else 0.0,
        "n_agent_trials": n,
    }


def run(data_dir: Path) -> pd.DataFrame:
    """Load all traces, compute summary statistics per condition × seed_doc."""
    df = load_traces(data_dir)
    if df.empty:
        return df

    # TODO: load ground_truth per seed_doc from seed document files.
    # For now, placeholder — replace with actual lookup.
    ground_truth_map = {
        "tech_earnings": "NEGATIVE",
        "policy_draft": "NEGATIVE",
        "geopolitical_event": "NEGATIVE",
    }

    rows = []
    for (condition, seed_doc), group in df.groupby(["condition", "seed_doc"]):
        gt = ground_truth_map.get(seed_doc, "NEGATIVE")
        summary = compute_summary(group, gt)
        summary["condition"] = condition
        summary["seed_doc"] = seed_doc
        rows.append(summary)

    summary_df = pd.DataFrame(rows)

    # Compute Δ² for each seed_doc.
    for seed_doc in df["seed_doc"].unique():
        baseline = summary_df[
            (summary_df["seed_doc"] == seed_doc)
            & (summary_df["condition"] == "flat_baseline")
        ]
        hierarchical = summary_df[
            (summary_df["seed_doc"] == seed_doc)
            & (summary_df["condition"] == "hierarchical_hallucination")
        ]
        if not baseline.empty and not hierarchical.empty:
            a0 = baseline.iloc[0]["mean_accuracy"]
            ai = hierarchical.iloc[0]["mean_accuracy"]
            delta = compute_delta_squared(a0, ai)
            logger.info(
                "Δ² for %s: A₀=%.3f, Aᵢ=%.3f, Δ²=%.3f",
                seed_doc, a0, ai, delta,
            )

    return summary_df


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    args = parser.parse_args()
    summary = run(args.data_dir)
    print(summary.to_string())


if __name__ == "__main__":
    main()
