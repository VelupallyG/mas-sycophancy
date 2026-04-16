"""Streamlit dashboard for MAS Sycophancy experiment results.

Usage:
    pip install -e ".[analysis]"
    streamlit run analysis/visualize.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.metrics.flip_metrics import compute_nof, compute_tof
from src.metrics.sycophancy_effect import (
    compute_agent_accuracy,
    compute_delta_squared,
)
from src.tasks.predictive_intel import extract_ground_truth_direction


@st.cache_data
def load_ground_truth_map() -> tuple[dict[str, str], dict[str, str]]:
    seed_dir = (
        Path(__file__).resolve().parent.parent / "src" / "tasks" / "seed_documents"
    )
    by_stem: dict[str, str] = {}
    by_metadata_id: dict[str, str] = {}
    for seed_path in sorted(seed_dir.glob("*.json")):
        with seed_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        direction = extract_ground_truth_direction(payload)
        if direction in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
            by_stem[seed_path.stem] = direction
            metadata_id = payload.get("metadata", {}).get("id")
            if metadata_id:
                by_metadata_id[metadata_id] = direction
    return by_stem, by_metadata_id


@st.cache_data
def load_traces(data_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for jsonl_path in sorted(Path(data_dir).rglob("trace.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["seed_doc_short"] = df["seed_doc"].apply(
        lambda x: {
            "tech_earnings_meta_2022": "Tech Earnings",
            "policy_lehman_2008": "Policy (Lehman)",
            "geopolitical_brexit_2016": "Geopolitical (Brexit)",
        }.get(x, x)
    )
    return df


@st.cache_data
def compute_agent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    _, by_metadata_id = load_ground_truth_map()
    rows = []

    for (trial_id, agent_id), group in df.groupby(["trial_id", "agent_id"]):
        seed_doc = group["seed_doc"].iloc[0]
        condition = group["condition"].iloc[0]
        level = group["level"].iloc[0]

        gt = by_metadata_id.get(seed_doc)
        if gt is None:
            continue

        stances = group.sort_values("turn")["prediction_direction"].tolist()
        accuracy = compute_agent_accuracy(stances, gt)
        tof = compute_tof(stances, gt)
        nof = compute_nof(stances)

        rows.append(
            {
                "trial_id": trial_id,
                "agent_id": agent_id,
                "condition": condition,
                "seed_doc": seed_doc,
                "seed_doc_short": group["seed_doc_short"].iloc[0],
                "level": level,
                "accuracy": accuracy,
                "tof": tof,
                "nof": nof,
                "ground_truth": gt,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def compute_summary(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()

    rows = []
    for (condition, seed_doc), group in agent_df.groupby(["condition", "seed_doc"]):
        rows.append(
            {
                "condition": condition,
                "seed_doc": seed_doc,
                "seed_doc_short": group["seed_doc_short"].iloc[0],
                "mean_accuracy": group["accuracy"].mean(),
                "mean_tof": group["tof"].mean(),
                "mean_nof": group["nof"].mean(),
                "std_accuracy": group["accuracy"].std(),
                "std_tof": group["tof"].std(),
                "std_nof": group["nof"].std(),
                "n_agent_trials": len(group),
            }
        )

    summary = pd.DataFrame(rows)

    for seed in summary["seed_doc"].unique():
        baseline = summary[
            (summary["seed_doc"] == seed) & (summary["condition"] == "flat_baseline")
        ]
        hierarchical = summary[
            (summary["seed_doc"] == seed)
            & (summary["condition"] == "hierarchical_hallucination")
        ]
        if not baseline.empty and not hierarchical.empty:
            delta = compute_delta_squared(
                baseline.iloc[0]["mean_accuracy"],
                hierarchical.iloc[0]["mean_accuracy"],
            )
            summary.loc[
                (summary["seed_doc"] == seed)
                & (summary["condition"] == "hierarchical_hallucination"),
                "delta_squared",
            ] = delta

    return summary


def main() -> None:
    st.set_page_config(
        page_title="MAS Sycophancy Dashboard",
        layout="wide",
        page_icon="🔬",
    )

    st.title("🔬 MAS Sycophancy Experiment Dashboard")
    st.markdown(
        "Investigating hallucination propagation in hierarchical vs. flat multi-agent systems."
    )

    data_dir = st.sidebar.text_input("Data directory", value="data")
    if not Path(data_dir).exists():
        st.error(f"Directory not found: {data_dir}")
        return

    df = load_traces(Path(data_dir))
    if df.empty:
        st.warning("No trace files found. Run experiments first.")
        return

    agent_df = compute_agent_metrics(df)
    summary_df = compute_summary(agent_df)

    with st.sidebar:
        st.header("Filters")
        conditions = st.multiselect(
            "Conditions",
            options=sorted(df["condition"].unique()),
            default=sorted(df["condition"].unique()),
        )
        seed_docs = st.multiselect(
            "Seed Documents",
            options=sorted(df["seed_doc_short"].unique()),
            default=sorted(df["seed_doc_short"].unique()),
        )

    filtered_summary = summary_df[
        summary_df["condition"].isin(conditions)
        & summary_df["seed_doc_short"].isin(seed_docs)
    ]
    filtered_agents = agent_df[
        agent_df["condition"].isin(conditions)
        & agent_df["seed_doc_short"].isin(seed_docs)
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trials", agent_df["trial_id"].nunique())
    col2.metric("Total Agents", agent_df["agent_id"].nunique())
    col3.metric("Conditions", agent_df["condition"].nunique())
    col4.metric("Seed Docs", agent_df["seed_doc"].nunique())
    st.divider()

    if not filtered_summary.empty:
        st.subheader("Δ² (Sycophancy Effect Size)")
        st.markdown(
            "**Δ² = A₀ − Aᵢ**: Positive values indicate hierarchical MAS degraded accuracy "
            "(regressive sycophancy). Higher = more sycophancy."
        )
        delta_data = filtered_summary[
            filtered_summary["condition"] == "hierarchical_hallucination"
        ][["seed_doc_short", "delta_squared", "mean_accuracy"]].dropna(
            subset=["delta_squared"]
        )

        if not delta_data.empty:
            fig_delta = px.bar(
                delta_data,
                x="seed_doc_short",
                y="delta_squared",
                color="delta_squared",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 1],
                labels={
                    "seed_doc_short": "Seed Document",
                    "delta_squared": "Δ² (Effect Size)",
                },
                text_auto=True,
            )
            fig_delta.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_range=[0, 1],
            )
            fig_delta.update_traces(texttemplate="%{y:.3f}", textposition="outside")
            st.plotly_chart(fig_delta, use_container_width=True)
        else:
            st.info(
                "Δ² requires both flat_baseline and hierarchical_hallucination conditions."
            )

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Turn of Flip (ToF) Distribution")
        st.caption(
            "Turn when agent first deviates from ground truth. "
            "Lower = faster capitulation to hallucination."
        )
        if not filtered_agents.empty:
            fig_tof = px.violin(
                filtered_agents,
                x="condition",
                y="tof",
                color="condition",
                facet_col="seed_doc_short",
                box=True,
                points="outliers",
                labels={"tof": "Turn of Flip", "condition": "Condition"},
            )
            fig_tof.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(fig_tof, use_container_width=True)

    with col_right:
        st.subheader("Number of Flips (NoF) Distribution")
        st.caption(
            "Total stance reversals across the trial. "
            "Higher = more cognitive oscillation / instability."
        )
        if not filtered_agents.empty:
            fig_nof = px.violin(
                filtered_agents,
                x="condition",
                y="nof",
                color="condition",
                facet_col="seed_doc_short",
                box=True,
                points="outliers",
                labels={"nof": "Number of Flips", "condition": "Condition"},
            )
            fig_nof.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(fig_nof, use_container_width=True)

    st.divider()
    st.subheader("Accuracy by Condition and Seed Document")

    if not filtered_summary.empty:
        fig_acc = px.bar(
            filtered_summary,
            x="seed_doc_short",
            y="mean_accuracy",
            color="condition",
            barmode="group",
            error_y="std_accuracy",
            labels={
                "mean_accuracy": "Mean Accuracy",
                "seed_doc_short": "Seed Document",
                "condition": "Condition",
            },
            color_discrete_map={
                "flat_baseline": "#2ecc71",
                "flat_hallucination": "#f39c12",
                "hierarchical_hallucination": "#e74c3c",
            },
        )
        fig_acc.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()
    st.subheader("Summary Statistics")

    if not filtered_summary.empty:
        display_cols = [
            "condition",
            "seed_doc_short",
            "mean_accuracy",
            "std_accuracy",
            "mean_tof",
            "std_tof",
            "mean_nof",
            "std_nof",
            "n_agent_trials",
        ]
        st.dataframe(
            filtered_summary[display_cols].rename(
                columns={
                    "condition": "Condition",
                    "seed_doc_short": "Seed Document",
                    "mean_accuracy": "Mean Accuracy",
                    "std_accuracy": "Std Accuracy",
                    "mean_tof": "Mean ToF",
                    "std_tof": "Std ToF",
                    "mean_nof": "Mean NoF",
                    "std_nof": "Std NoF",
                    "n_agent_trials": "N Agent-Trials",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    with st.expander("Raw data explorer"):
        st.dataframe(
            filtered_agents[
                [
                    "trial_id",
                    "agent_id",
                    "condition",
                    "seed_doc_short",
                    "level",
                    "accuracy",
                    "tof",
                    "nof",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
