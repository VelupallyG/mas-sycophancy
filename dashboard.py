"""Streamlit dashboard for MAS Sycophancy experiment results (matplotlib/seaborn).

A second dashboard option using matplotlib/seaborn instead of Plotly.
Includes a sample data generator for demo purposes.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from analysis.aggregate_results import load_traces, load_ground_truth_map
from src.metrics.flip_metrics import compute_nof, compute_tof
from src.metrics.sycophancy_effect import (
    compute_agent_accuracy,
    compute_delta_squared,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MAS Sycophancy Dashboard",
    page_icon="🔬",
    layout="wide",
)

_PALETTE = {
    "flat_baseline": "#4C72B0",
    "flat_hallucination": "#F0A030",
    "hierarchical_hallucination": "#DD8452",
}

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
data_dir = st.sidebar.text_input("Data directory", value="data/")
data_path = Path(data_dir)

# ---------------------------------------------------------------------------
# Sample data generator
# ---------------------------------------------------------------------------


def _generate_sample_data(output_dir: Path) -> None:
    """Write synthetic JSONL trace files for demo purposes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [
        "finance_earnings_alphabet_ai_capex_2026_v1",
        "geopolitics_sanctions_oil_supplyshock_2025_v1",
    ]
    directions = {
        "finance_earnings_alphabet_ai_capex_2026_v1": "NEGATIVE",
        "geopolitics_sanctions_oil_supplyshock_2025_v1": "POSITIVE",
    }

    for seed in seeds:
        gt = directions[seed]
        opposite = "POSITIVE" if gt == "NEGATIVE" else "NEGATIVE"

        for condition in ["flat_baseline", "hierarchical_hallucination"]:
            trial_dir = output_dir / condition / seed / "trial_000"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trial_dir / "trace.jsonl"

            records = []
            for i in range(5):
                level = 3 if condition == "hierarchical_hallucination" else 0
                for t in range(1, 6):
                    if condition == "hierarchical_hallucination" and t >= 3:
                        direction = opposite
                    else:
                        direction = gt
                    records.append(
                        {
                            "trial_id": "trial_000",
                            "seed_doc": seed,
                            "condition": condition,
                            "turn": t,
                            "agent_id": f"analyst_{i:02d}",
                            "level": level,
                            "prediction_direction": direction,
                            "predicted_magnitude": "MEDIUM",
                            "predicted_price_change_pct": -6.0
                            if direction == "NEGATIVE"
                            else 6.0,
                            "prediction_summary": f"Demo prediction turn {t}.",
                            "key_factors": ["demo factor"],
                            "parse_success": True,
                        }
                    )

            with trace_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


@st.cache_data
def load_and_compute(data_dir_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load traces and compute per-agent metrics + summary."""
    df = load_traces(Path(data_dir_str))
    if df.empty:
        return df, pd.DataFrame()

    gt_map = load_ground_truth_map()

    agent_rows = []
    for _, group in df.groupby(["trial_id", "agent_id"]):
        trial_id = str(group["trial_id"].iloc[0])
        agent_id = str(group["agent_id"].iloc[0])
        seed_doc = str(group["seed_doc"].iloc[0])
        condition = str(group["condition"].iloc[0])
        level = group["level"].iloc[0]

        gt = gt_map.get(seed_doc)
        if gt is None:
            continue

        stances = group.sort_values("turn")["prediction_direction"].tolist()
        accuracy = compute_agent_accuracy(stances, gt)
        tof = compute_tof(stances, gt)
        nof = compute_nof(stances)

        agent_rows.append(
            {
                "trial_id": trial_id,
                "agent_id": agent_id,
                "condition": condition,
                "seed_doc": seed_doc,
                "level": level,
                "accuracy": accuracy,
                "tof": tof,
                "nof": nof,
                "ground_truth": gt,
            }
        )

    agent_df = pd.DataFrame(agent_rows)

    # Summary per condition x seed_doc
    summary_rows = []
    for _, grp in agent_df.groupby(["condition", "seed_doc"]):
        cond = str(grp["condition"].iloc[0])
        sd = str(grp["seed_doc"].iloc[0])
        summary_rows.append(
            {
                "condition": cond,
                "seed_doc": sd,
                "mean_accuracy": grp["accuracy"].mean(),
                "std_accuracy": grp["accuracy"].std(),
                "mean_tof": grp["tof"].mean(),
                "std_tof": grp["tof"].std(),
                "mean_nof": grp["nof"].mean(),
                "std_nof": grp["nof"].std(),
                "n_agent_trials": len(grp),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Compute delta-squared
    for sd in agent_df["seed_doc"].unique():
        bl = summary_df[
            (summary_df["seed_doc"] == sd)
            & (summary_df["condition"] == "flat_baseline")
        ]
        hi = summary_df[
            (summary_df["seed_doc"] == sd)
            & (summary_df["condition"] == "hierarchical_hallucination")
        ]
        if not bl.empty and not hi.empty:
            delta = compute_delta_squared(
                bl.iloc[0]["mean_accuracy"], hi.iloc[0]["mean_accuracy"]
            )
            summary_df.loc[
                (summary_df["seed_doc"] == sd)
                & (summary_df["condition"] == "hierarchical_hallucination"),
                "delta_squared",
            ] = delta

    return agent_df, summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not data_path.exists():
    st.title("MAS Sycophancy & Hallucination Dashboard")
    st.warning(f"Directory `{data_dir}` not found.")
    if st.button("Generate sample data for demo"):
        _generate_sample_data(data_path)
        st.rerun()
    st.stop()

# Check for trace files
trace_files = list(data_path.rglob("trace.jsonl"))
if not trace_files:
    st.title("MAS Sycophancy & Hallucination Dashboard")
    st.warning(
        f"No trace.jsonl files found in `{data_dir}`. Run experiments first or generate sample data."
    )
    if st.button("Generate sample data for demo"):
        _generate_sample_data(data_path)
        st.rerun()
    st.stop()

agent_df, summary_df = load_and_compute(data_dir)

if agent_df.empty:
    st.error("Trace files found but produced an empty DataFrame.")
    st.stop()

# ---------------------------------------------------------------------------
# Title + high-level metrics
# ---------------------------------------------------------------------------
st.title("MAS Sycophancy & Hallucination Dashboard")
st.markdown(
    "Comparing hallucination propagation in **flat** vs **hierarchical** multi-agent systems."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trials", int(agent_df["trial_id"].nunique()))
col2.metric("Seed Documents", int(agent_df["seed_doc"].nunique()))

baseline = agent_df[agent_df["condition"] == "flat_baseline"]
influenced = agent_df[agent_df["condition"] == "hierarchical_hallucination"]
col3.metric(
    "Avg Baseline Accuracy",
    f"{baseline['accuracy'].mean():.2f}" if not baseline.empty else "N/A",
)
col4.metric(
    "Avg Hierarchical Accuracy",
    f"{influenced['accuracy'].mean():.2f}" if not influenced.empty else "N/A",
)

st.divider()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
conditions = st.sidebar.multiselect(
    "Conditions",
    options=sorted(agent_df["condition"].unique()),
    default=sorted(agent_df["condition"].unique()),
)
seed_docs_filter = st.sidebar.multiselect(
    "Seed Documents",
    options=sorted(agent_df["seed_doc"].unique()),
    default=sorted(agent_df["seed_doc"].unique()),
)

filtered_agents = agent_df[
    agent_df["condition"].isin(conditions) & agent_df["seed_doc"].isin(seed_docs_filter)
]
filtered_summary = summary_df[
    summary_df["condition"].isin(conditions)
    & summary_df["seed_doc"].isin(seed_docs_filter)
]

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Summary Statistics")
if not filtered_summary.empty:
    display_cols = [
        "condition",
        "seed_doc",
        "mean_accuracy",
        "std_accuracy",
        "mean_tof",
        "std_tof",
        "mean_nof",
        "std_nof",
        "n_agent_trials",
    ]
    if "delta_squared" in filtered_summary.columns:
        display_cols.append("delta_squared")
    available = [c for c in display_cols if c in filtered_summary.columns]
    st.dataframe(filtered_summary[available], use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

# Delta-squared bar chart
with chart_col1:
    st.subheader("Sycophancy Effect (Delta-Squared)")
    if "delta_squared" in filtered_summary.columns:
        ds_df = filtered_summary.dropna(subset=["delta_squared"])
    else:
        ds_df = pd.DataFrame()

    if ds_df.empty:
        st.info(
            "No delta-squared data (need both flat_baseline and hierarchical_hallucination)."
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(ds_df["seed_doc"], ds_df["delta_squared"], color="#DD8452")
        ax.set_xlabel("Seed Document")
        ax.set_ylabel("Delta-Squared")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ToF / NoF
with chart_col2:
    st.subheader("Stance Flip Metrics")
    tab_tof, tab_nof = st.tabs(["Turn of Flip (ToF)", "Number of Flips (NoF)"])

    with tab_tof:
        if filtered_agents.empty:
            st.info("No ToF data.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(
                data=filtered_agents, x="condition", y="tof", palette=_PALETTE, ax=ax
            )
            ax.set_xlabel("Condition")
            ax.set_ylabel("Turn of First Flip")
            plt.xticks(rotation=15, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_nof:
        nof_data = (
            filtered_agents[filtered_agents["nof"] > 0]
            if not filtered_agents.empty
            else pd.DataFrame()
        )
        if nof_data.empty:
            st.info("No flip data available.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=nof_data, x="condition", y="nof", palette=_PALETTE, ax=ax)
            ax.set_xlabel("Condition")
            ax.set_ylabel("Total Flips")
            plt.xticks(rotation=15, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.divider()

# Accuracy by condition
st.subheader("Accuracy by Condition and Seed Document")
if not filtered_summary.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=filtered_summary,
        x="seed_doc",
        y="mean_accuracy",
        hue="condition",
        palette=_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("Seed Document")
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Condition")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Raw data explorer
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Raw agent data explorer"):
    if not filtered_agents.empty:
        st.dataframe(
            filtered_agents[
                [
                    "trial_id",
                    "agent_id",
                    "condition",
                    "seed_doc",
                    "level",
                    "accuracy",
                    "tof",
                    "nof",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
