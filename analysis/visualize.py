"""Generate charts and figures from aggregated experiment results.

Usage:
  python -m analysis.visualize --data-dir data/ --output-dir figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_delta_squared(summary_df, output_dir: Path) -> None:
    """Bar chart of Δ² per seed document."""
    import matplotlib.pyplot as plt
    import pandas as pd

    # TODO: implement actual plotting once data is available.
    logger.info("Plotting Δ² (stub — implement when data is available).")
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_tof_distribution(df, output_dir: Path) -> None:
    """Box plot of Turn-of-Flip by condition and seed document."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # TODO: implement.
    logger.info("Plotting ToF distribution (stub).")


def plot_nof_distribution(df, output_dir: Path) -> None:
    """Box plot of Number-of-Flips by condition."""
    # TODO: implement.
    logger.info("Plotting NoF distribution (stub).")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--output-dir", default="figures", type=Path)
    args = parser.parse_args()

    from analysis.aggregate_results import load_traces, run
    summary_df = run(args.data_dir)
    df = load_traces(args.data_dir)

    plot_delta_squared(summary_df, args.output_dir)
    plot_tof_distribution(df, args.output_dir)
    plot_nof_distribution(df, args.output_dir)
    logger.info("Figures written to %s", args.output_dir)


if __name__ == "__main__":
    main()
