"""Generate charts and figures from aggregated experiment results.

Reads the CSV produced by ``aggregate_results.py`` and produces:
  1. Δ² comparison: flat vs hierarchical × seed document (bar chart)
  2. ToF / NoF distributions: violin or box plots per condition
  3. TRAIL failure breakdown: stacked bar per topology condition
  4. Deference marker trajectory: line chart over turns per topology

CLI usage::

    python -m analysis.visualize --data-dir data/ --output-dir figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_delta_squared(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of Δ² by topology and seed document.

    Args:
        df: Aggregated results DataFrame from ``aggregate_results.aggregate``.
        output_dir: Directory where the figure PNG is written.
    """
    raise NotImplementedError


def plot_tof_nof(df: pd.DataFrame, output_dir: Path) -> None:
    """Side-by-side violin plots of ToF and NoF distributions.

    Args:
        df: Aggregated results DataFrame.
        output_dir: Directory where the figure PNG is written.
    """
    raise NotImplementedError


def plot_trail_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar chart of TRAIL error categories per condition.

    Args:
        df: Aggregated results DataFrame.
        output_dir: Directory where the figure PNG is written.
    """
    raise NotImplementedError


def plot_deference_trajectory(df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart of mean deference marker count per turn per topology.

    Args:
        df: Aggregated results DataFrame.
        output_dir: Directory where the figure PNG is written.
    """
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise aggregated experiment results.")
    parser.add_argument("--data-dir", default="data/", help="Directory with result CSVs.")
    parser.add_argument("--output-dir", default="figures/", help="Output directory for PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_dir) / "aggregate_results.csv"
    df = pd.read_csv(csv_path)

    plot_delta_squared(df, output_dir)
    plot_tof_nof(df, output_dir)
    plot_trail_breakdown(df, output_dir)
    plot_deference_trajectory(df, output_dir)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
