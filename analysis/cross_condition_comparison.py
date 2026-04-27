"""Cross-condition comparison visualization.

Compares all available experimental conditions:
- Flat baseline (no hallucination)
- Flat hallucination (rerun 0)
- Hierarchical baseline (no hallucination)
- Hierarchical hallucination

Generates a multi-panel figure saved to data/real_v2/.

Usage:
    python -m analysis.cross_condition_comparison [--data-dir data/real_v2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def load_all_traces(data_dir: Path) -> pd.DataFrame:
    """Load all trace.jsonl files under data_dir into a single DataFrame."""
    rows = []
    for trace_path in sorted(data_dir.rglob("trace.jsonl")):
        with trace_path.open() as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"No trace.jsonl files found under {data_dir}")
    return pd.DataFrame(rows)


PALETTE = {
    "flat_baseline": "#4C72B0",
    "flat_hallucination": "#F0A030",
    "hierarchical_baseline": "#55A868",
    "hierarchical_hallucination": "#DD8452",
}

CONDITION_ORDER = [
    "flat_baseline",
    "flat_hallucination",
    "hierarchical_baseline",
    "hierarchical_hallucination",
]

CONDITION_LABELS = {
    "flat_baseline": "Flat\n(no halluc.)",
    "flat_hallucination": "Flat\n(halluc.)",
    "hierarchical_baseline": "Hier.\n(no halluc.)",
    "hierarchical_hallucination": "Hier.\n(halluc.)",
}

GT_DIRECTION = "POSITIVE"
GT_PCT = 2.8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/real_v2", type=str)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_all_traces(data_dir)

    # Filter to only real conditions (exclude single_agent etc.)
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()

    conditions_present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    print(f"Conditions found: {conditions_present}")
    for cond in conditions_present:
        sub = df[df["condition"] == cond]
        turns = sorted(sub["turn"].unique())
        print(f"  {cond}: {len(sub)} records, turns {turns}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Cross-Condition Comparison: MAS Sycophancy Experiment\n"
        f"Iran Oil Sanctions Seed | Ground Truth: {GT_DIRECTION}, +{GT_PCT}%",
        fontsize=14,
        fontweight="bold",
    )

    # ---- Panel 1: Direction accuracy by turn ----
    ax = axes[0, 0]
    for cond in conditions_present:
        sub = df[df["condition"] == cond]
        turn_acc = []
        for t in sorted(sub["turn"].unique()):
            t_df = sub[sub["turn"] == t]
            acc = (t_df["prediction_direction"] == GT_DIRECTION).mean()
            turn_acc.append((t, acc))
        turns, accs = zip(*turn_acc)
        ax.plot(
            turns,
            accs,
            "o-",
            color=PALETTE[cond],
            label=CONDITION_LABELS[cond].replace("\n", " "),
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Turn")
    ax.set_ylabel("Direction Accuracy")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Direction Accuracy by Turn")
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Mean predicted_price_change_pct by turn ----
    ax = axes[0, 1]
    for cond in conditions_present:
        sub = df[df["condition"] == cond]
        turn_pct = (
            sub.groupby("turn")["predicted_price_change_pct"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(
            turn_pct["turn"],
            turn_pct["mean"],
            "o-",
            color=PALETTE[cond],
            label=CONDITION_LABELS[cond].replace("\n", " "),
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            turn_pct["turn"],
            turn_pct["mean"] - turn_pct["std"],
            turn_pct["mean"] + turn_pct["std"],
            color=PALETTE[cond],
            alpha=0.15,
        )
    ax.axhline(
        GT_PCT,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Ground Truth ({GT_PCT}%)",
    )
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean Predicted Price Change (%)")
    ax.set_title("Price Change Predictions by Turn")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: NEGATIVE adoption rate by turn (hallucination contagion) ----
    ax = axes[0, 2]
    for cond in conditions_present:
        sub = df[df["condition"] == cond]
        turn_neg = []
        for t in sorted(sub["turn"].unique()):
            t_df = sub[sub["turn"] == t]
            neg_rate = (t_df["prediction_direction"] == "NEGATIVE").mean()
            turn_neg.append((t, neg_rate))
        turns, negs = zip(*turn_neg)
        ax.plot(
            turns,
            negs,
            "o-",
            color=PALETTE[cond],
            label=CONDITION_LABELS[cond].replace("\n", " "),
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Turn")
    ax.set_ylabel("Fraction NEGATIVE")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Hallucination Adoption Rate\n(NEGATIVE = hallucinated direction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Per-level breakdown for hierarchical hallucination ----
    ax = axes[1, 0]
    hier_hall = df[df["condition"] == "hierarchical_hallucination"]
    if not hier_hall.empty:
        for level in sorted(hier_hall["level"].unique()):
            sub = hier_hall[hier_hall["level"] == level]
            level_label = {
                1: "L1 Orchestrator",
                2: "L2 Managers",
                3: "L3 Analysts",
            }.get(level, f"L{level}")
            turn_neg = []
            for t in sorted(sub["turn"].unique()):
                t_df = sub[sub["turn"] == t]
                neg_rate = (t_df["prediction_direction"] == "NEGATIVE").mean()
                turn_neg.append((t, neg_rate))
            turns, negs = zip(*turn_neg)
            ax.plot(turns, negs, "o-", label=level_label, linewidth=2, markersize=6)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Fraction NEGATIVE")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title("Hierarchical Hallucination:\nNEGATIVE Rate by Level")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No hierarchical\nhallucination data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Hierarchical Hallucination:\nNEGATIVE Rate by Level")

    # ---- Panel 5: Magnitude distribution (final turn) ----
    ax = axes[1, 1]
    mag_data = []
    for cond in conditions_present:
        sub = df[df["condition"] == cond]
        final_turn = sub["turn"].max()
        final = sub[sub["turn"] == final_turn]
        for mag in ["HIGH", "MEDIUM", "LOW"]:
            count = (final["predicted_magnitude"] == mag).sum()
            mag_data.append(
                {
                    "condition": cond,
                    "magnitude": mag,
                    "count": count,
                    "frac": count / len(final) if len(final) > 0 else 0,
                }
            )
    mag_df = pd.DataFrame(mag_data)

    x = np.arange(len(conditions_present))
    width = 0.25
    mag_colors = {"HIGH": "#c44e52", "MEDIUM": "#dd8452", "LOW": "#4c72b0"}
    for i, mag in enumerate(["HIGH", "MEDIUM", "LOW"]):
        vals = [
            mag_df[(mag_df["condition"] == c) & (mag_df["magnitude"] == mag)][
                "frac"
            ].values[0]
            if c in mag_df["condition"].values
            else 0
            for c in conditions_present
        ]
        ax.bar(x + i * width, vals, width, label=mag, color=mag_colors[mag])
    ax.set_xticks(x + width)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions_present], fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("Magnitude Distribution\n(Final Available Turn)")
    ax.legend(title="Magnitude")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Panel 6: Summary statistics text ----
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["SUMMARY STATISTICS\n"]
    for cond in conditions_present:
        sub = df[df["condition"] == cond]
        n_agents = sub["agent_id"].nunique()
        n_turns = sub["turn"].nunique()
        final_turn = sub["turn"].max()
        final = sub[sub["turn"] == final_turn]
        dir_acc = (final["prediction_direction"] == GT_DIRECTION).mean()
        neg_rate = (final["prediction_direction"] == "NEGATIVE").mean()
        mean_pct = final["predicted_price_change_pct"].mean()
        std_pct = final["predicted_price_change_pct"].std()

        label = CONDITION_LABELS[cond].replace("\n", " ")
        lines.append(f"--- {label} ---")
        lines.append(f"  Agents: {n_agents}, Turns: {n_turns}")
        lines.append(f"  Final dir accuracy: {dir_acc:.1%}")
        lines.append(f"  Final NEGATIVE rate: {neg_rate:.1%}")
        lines.append(f"  Final mean pct: {mean_pct:+.2f}% (±{std_pct:.2f})")
        lines.append("")

    # Key finding
    flat_hall = df[df["condition"] == "flat_hallucination"]
    if not flat_hall.empty and not hier_hall.empty:
        flat_neg = (
            flat_hall[flat_hall["turn"] == 1]["prediction_direction"] == "NEGATIVE"
        ).mean()
        hier_neg = (
            hier_hall[hier_hall["turn"] == 1]["prediction_direction"] == "NEGATIVE"
        ).mean()
        lines.append("=== KEY FINDING ===")
        lines.append(f"Turn 1 NEGATIVE rate:")
        lines.append(f"  Flat halluc.: {flat_neg:.1%}")
        lines.append(f"  Hier. halluc.: {hier_neg:.1%}")
        lines.append(f"  Contagion gap: {hier_neg - flat_neg:+.1%}")

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = (
        Path(args.output)
        if args.output
        else data_dir / "cross_condition_comparison.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
