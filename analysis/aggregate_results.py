"""Aggregate experiment results into a tidy DataFrame.

Reads all ``SimulationResult`` JSON files from the data directory, computes
per-run metrics (Δ², ToF, NoF, TRAIL categories, deference counts), and
returns a single ``pd.DataFrame`` with one row per simulation run.

CLI usage::

    python -m analysis.aggregate_results --data-dir data/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate(data_dir: str | Path) -> pd.DataFrame:
    """Load all result JSON files and compute aggregate metrics.

    Args:
        data_dir: Directory containing ``SimulationResult`` JSON files
            produced by the experiment runners.

    Returns:
        DataFrame with columns: experiment_id, topology, seed_doc,
        whistleblower_rank, baseline_acc, influenced_acc, delta_squared,
        mean_tof, total_nof, trail_reasoning_pct, trail_planning_pct,
        trail_system_pct, mean_deference_count.
    """
    raise NotImplementedError


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
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
