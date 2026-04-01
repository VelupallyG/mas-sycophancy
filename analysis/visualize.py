"""Generate charts and figures from aggregated experiment results.

Reads the CSV produced by ``aggregate_results.py`` and produces:
  1. Delta-squared comparison: flat vs hierarchical x seed document (bar chart)
  2. ToF / NoF distributions: violin or box plots per condition
  3. TRAIL failure breakdown: stacked bar per topology condition
  4. Deference marker trajectory: line chart over turns per topology

CLI usage::

    python -m analysis.visualize --data-dir data/ --output-dir figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Consistent styling across all charts
_PALETTE = {"flat": "#4C72B0", "hierarchical": "#DD8452"}
_FIG_SIZE = (10, 6)


def plot_delta_squared(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of delta-squared by topology and seed document.

    Args:
        df: Aggregated results DataFrame from ``aggregate_results.aggregate``.
        output_dir: Directory where the figure PNG is written.
    """
    if "delta_squared" not in df.columns or df.empty:
        return
    plot_df = df.dropna(subset=["delta_squared"]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    sns.barplot(
        data=plot_df,
        x="seed_doc",
        y="delta_squared",
        hue="topology",
        palette=_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("Seed Document")
    ax.set_ylabel("Delta-Squared (Sycophancy Effect Size)")
    ax.set_title("Sycophancy Effect: Flat vs Hierarchical by Seed Document")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(title="Topology")
    fig.tight_layout()
    fig.savefig(output_dir / "delta_squared.png", dpi=150)
    plt.close(fig)


def plot_tof_nof(df: pd.DataFrame, output_dir: Path) -> None:
    """Side-by-side violin plots of ToF and NoF distributions.

    Args:
        df: Aggregated results DataFrame.
        output_dir: Directory where the figure PNG is written.
    """
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(_FIG_SIZE[0], _FIG_SIZE[1]))

    # ToF violin
    tof_df = df.dropna(subset=["mean_tof"]) if "mean_tof" in df.columns else pd.DataFrame()
    if not tof_df.empty:
        sns.violinplot(
            data=tof_df,
            x="topology",
            y="mean_tof",
            palette=_PALETTE,
            ax=axes[0],
            inner="box",
            cut=0,
        )
    axes[0].set_xlabel("Topology")
    axes[0].set_ylabel("Turn of Flip (ToF)")
    axes[0].set_title("Turn of First Stance Flip")

    # NoF violin
    nof_df = df[df["total_nof"] > 0] if "total_nof" in df.columns else pd.DataFrame()
    if not nof_df.empty:
        sns.violinplot(
            data=nof_df,
            x="topology",
            y="total_nof",
            palette=_PALETTE,
            ax=axes[1],
            inner="box",
            cut=0,
        )
    axes[1].set_xlabel("Topology")
    axes[1].set_ylabel("Number of Flips (NoF)")
    axes[1].set_title("Total Stance Reversals")

    fig.tight_layout()
    fig.savefig(output_dir / "tof_nof.png", dpi=150)
    plt.close(fig)


def plot_trail_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Stacked bar chart of TRAIL error categories per condition.

    Args:
        df: Aggregated results DataFrame.
        output_dir: Directory where the figure PNG is written.
    """
    if df.empty:
        return

    # Group by topology and condition, take mean of TRAIL percentages
    group_cols = ["topology", "condition"]
    available_cols = [c for c in group_cols if c in df.columns]
    if not available_cols:
        return

    trail_cols = ["trail_reasoning_pct", "trail_planning_pct", "trail_system_pct"]
    plot_df = df.groupby(available_cols, as_index=False)[trail_cols].mean()

    if plot_df.empty:
        return

    # Create a label column for x-axis
    if len(available_cols) == 2:
        plot_df["label"] = plot_df["topology"] + " / " + plot_df["condition"]
    else:
        plot_df["label"] = plot_df[available_cols[0]]

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    bottom = [0.0] * len(plot_df)
    colors = {"Reasoning": "#E24A33", "Planning": "#FBC15E", "System": "#8EBA42"}

    for col, label in [
        ("trail_reasoning_pct", "Reasoning"),
        ("trail_planning_pct", "Planning"),
        ("trail_system_pct", "System"),
    ]:
        values = plot_df[col].tolist()
        ax.bar(plot_df["label"], values, bottom=bottom, label=label, color=colors[label])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Condition")
    ax.set_ylabel("Error Category (%)")
    ax.set_title("TRAIL Failure Breakdown by Condition")
    ax.legend(title="Error Type")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "trail_breakdown.png", dpi=150)
    plt.close(fig)


def plot_deference_trajectory(df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart of mean deference marker count per turn per topology.

    If the aggregated DataFrame has a ``mean_deference_count`` column (one
    value per run), this plots it grouped by topology. For richer per-turn
    data, pass a DataFrame with a ``turn`` column.

    Args:
        df: Aggregated results DataFrame.
        output_dir: Directory where the figure PNG is written.
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    if "turn" in df.columns and "deference_count" in df.columns:
        # Per-turn granularity available
        sns.lineplot(
            data=df,
            x="turn",
            y="deference_count",
            hue="topology",
            palette=_PALETTE,
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Turn")
        ax.set_ylabel("Deference Marker Count")
    else:
        # Fallback: one value per experiment run
        plot_df = df[df["mean_deference_count"] > 0]
        if not plot_df.empty:
            sns.barplot(
                data=plot_df,
                x="seed_doc",
                y="mean_deference_count",
                hue="topology",
                palette=_PALETTE,
                ax=ax,
            )
            ax.set_xlabel("Seed Document")
            ax.set_ylabel("Mean Deference Markers per Turn")

    ax.set_title("Deference Marker Trajectory by Topology")
    ax.legend(title="Topology")
    fig.tight_layout()
    fig.savefig(output_dir / "deference_trajectory.png", dpi=150)
    plt.close(fig)


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
