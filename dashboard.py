"""Streamlit dashboard for MAS Sycophancy experiment results.

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

from analysis.aggregate_results import aggregate
from src.metrics.scoring import score_to_grade

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MAS Sycophancy Dashboard",
    page_icon="🔬",  # user requested emojis indirectly via dashboard context
    layout="wide",
)

_PALETTE = {"flat": "#4C72B0", "hierarchical": "#DD8452"}

# ---------------------------------------------------------------------------
# Sidebar: data source
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")
data_dir = st.sidebar.text_input("Data directory", value="data/")
data_path = Path(data_dir)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(directory: str) -> pd.DataFrame:
    return aggregate(directory)


def _generate_sample_data(output_dir: Path) -> None:
    """Write synthetic result JSONs for demo purposes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = ["tech_earnings", "policy_draft", "geopolitical_event"]
    directions = {"tech_earnings": "positive", "policy_draft": "negative", "geopolitical_event": "neutral"}

    for seed in seeds:
        gt = directions[seed]
        opposite = "negative" if gt == "positive" else "positive"

        # Flat baseline
        flat_records = []
        for i in range(5):
            for t in range(1, 6):
                flat_records.append({
                    "agent_name": f"Analyst_{i:02d}",
                    "turn": t,
                    "stance": gt,
                    "text": f"Analyst_{i:02d} turn {t}: prediction {gt}",
                    "expected_stance": gt,
                    "hierarchy_level": None,
                })

        flat_result = {
            "experiment_id": f"flat_{seed}_demo",
            "condition": "baseline",
            "seed_doc": seed,
            "accuracy": 0.85,
            "consensus_prediction": f"{gt} moderate",
            "trace_path": "",
            "agent_turn_records": flat_records,
            "metadata": {
                "seed_id": f"{seed}_001",
                "seed_doc": seed,
                "ground_truth_direction": gt,
                "ground_truth_magnitude": "moderate",
                "topology": "flat",
                "max_turns": 5,
            },
        }

        # Hierarchical influenced
        hier_records = []
        for i in range(5):
            level = 5 if i >= 2 else (4 if i == 1 else 2)
            for t in range(1, 6):
                stance = gt if t <= (2 if level >= 4 else 3) else opposite
                text = f"Agent_{i:02d} turn {t}: prediction {stance}"
                if stance != gt:
                    text += " i defer to your judgment"
                hier_records.append({
                    "agent_name": f"Agent_{i:02d}",
                    "turn": t,
                    "stance": stance,
                    "text": text,
                    "expected_stance": gt,
                    "hierarchy_level": level,
                })

        hier_result = {
            "experiment_id": f"hier_{seed}_demo",
            "condition": "influenced",
            "seed_doc": seed,
            "accuracy": 0.30,
            "consensus_prediction": f"{opposite} moderate",
            "trace_path": "",
            "agent_turn_records": hier_records,
            "metadata": {
                "seed_id": f"{seed}_001",
                "seed_doc": seed,
                "ground_truth_direction": gt,
                "ground_truth_magnitude": "moderate",
                "topology": "hierarchical",
                "max_turns": 5,
            },
        }

        (output_dir / f"flat_{seed}_demo_result.json").write_text(
            json.dumps(flat_result, indent=2), encoding="utf-8"
        )
        (output_dir / f"hier_{seed}_demo_result.json").write_text(
            json.dumps(hier_result, indent=2), encoding="utf-8"
        )


if not data_path.exists() or not list(data_path.glob("*_result.json")):
    st.title("MAS Sycophancy & Hallucination Dashboard")
    st.warning(
        f"No result JSON files found in `{data_dir}`. "
        "Run experiments first or generate sample data below."
    )

    if st.button("Generate sample data for demo"):
        _generate_sample_data(data_path)
        st.rerun()

    st.stop()


df = load_data(data_dir)

if df.empty:
    st.error("Result files found but produced an empty DataFrame.")
    st.stop()

# ---------------------------------------------------------------------------
# Title + high-level metrics
# ---------------------------------------------------------------------------
st.title("MAS Sycophancy & Hallucination Dashboard")
st.markdown("Comparing hallucination propagation in **flat** vs **hierarchical** multi-agent systems.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments", len(df))
col2.metric("Seed Documents", df["seed_doc"].nunique())

baseline = df[df["condition"] == "baseline"]
influenced = df[df["condition"] != "baseline"]
col3.metric("Avg Baseline Accuracy", f"{baseline['accuracy'].mean():.2f}" if not baseline.empty else "N/A")
col4.metric("Avg Influenced Accuracy", f"{influenced['accuracy'].mean():.2f}" if not influenced.empty else "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
seed_docs = ["All"] + sorted(df["seed_doc"].dropna().unique().tolist())
selected_seed = st.sidebar.selectbox("Seed Document", seed_docs)

topologies = ["All"] + sorted(df["topology"].dropna().unique().tolist())
selected_topology = st.sidebar.selectbox("Topology", topologies)

filtered = df.copy()
if selected_seed != "All":
    filtered = filtered[filtered["seed_doc"] == selected_seed]
if selected_topology != "All":
    filtered = filtered[filtered["topology"] == selected_topology]

# ---------------------------------------------------------------------------
# Scores table
# ---------------------------------------------------------------------------
st.subheader("Experiment Results")
display_cols = [
    "experiment_id", "topology", "condition", "seed_doc",
    "accuracy", "score", "grade", "delta_squared",
    "mean_tof", "total_nof", "mean_deference_count",
]
available_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(
    filtered[available_cols].style.format({
        "accuracy": "{:.3f}",
        "score": "{:.1f}",
        "delta_squared": "{:.3f}",
        "mean_tof": "{:.2f}",
        "mean_deference_count": "{:.2f}",
    }, na_rep="—"),
    use_container_width=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

# Delta-squared bar chart
with chart_col1:
    st.subheader("Sycophancy Effect (Delta-Squared)")
    ds_df = filtered.dropna(subset=["delta_squared"])
    if ds_df.empty:
        st.info("No delta-squared data available (need both baseline and influenced runs for the same seed).")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=ds_df, x="seed_doc", y="delta_squared", hue="topology", palette=_PALETTE, ax=ax)
        ax.set_xlabel("Seed Document")
        ax.set_ylabel("Delta-Squared")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.legend(title="Topology")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ToF / NoF
with chart_col2:
    st.subheader("Stance Flip Metrics")
    tab_tof, tab_nof = st.tabs(["Turn of Flip (ToF)", "Number of Flips (NoF)"])

    with tab_tof:
        tof_df = filtered.dropna(subset=["mean_tof"]) if "mean_tof" in filtered.columns else pd.DataFrame()
        if tof_df.empty:
            st.info("No ToF data (no agents flipped).")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=tof_df, x="topology", y="mean_tof", palette=_PALETTE, ax=ax)
            ax.set_xlabel("Topology")
            ax.set_ylabel("Turn of First Flip")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_nof:
        nof_df = filtered[filtered["total_nof"] > 0] if "total_nof" in filtered.columns else pd.DataFrame()
        if nof_df.empty:
            st.info("No flip data available.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=nof_df, x="topology", y="total_nof", palette=_PALETTE, ax=ax)
            ax.set_xlabel("Topology")
            ax.set_ylabel("Total Flips")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

st.divider()

chart_col3, chart_col4 = st.columns(2)

# TRAIL breakdown
with chart_col3:
    st.subheader("TRAIL Error Breakdown")
    trail_cols = ["trail_reasoning_pct", "trail_planning_pct", "trail_system_pct"]
    if all(c in filtered.columns for c in trail_cols):
        group_cols = [c for c in ["topology", "condition"] if c in filtered.columns]
        if group_cols:
            trail_df = filtered.groupby(group_cols, as_index=False)[trail_cols].mean()
            trail_df["label"] = trail_df.apply(
                lambda r: " / ".join(str(r[c]) for c in group_cols), axis=1
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            bottom = [0.0] * len(trail_df)
            colors = {"Reasoning": "#E24A33", "Planning": "#FBC15E", "System": "#8EBA42"}
            for col, label in [
                ("trail_reasoning_pct", "Reasoning"),
                ("trail_planning_pct", "Planning"),
                ("trail_system_pct", "System"),
            ]:
                values = trail_df[col].tolist()
                ax.bar(trail_df["label"], values, bottom=bottom, label=label, color=colors[label])
                bottom = [b + v for b, v in zip(bottom, values)]

            ax.set_ylabel("Error Category (%)")
            ax.legend(title="Error Type")
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("TRAIL data not available.")

# Deference markers
with chart_col4:
    st.subheader("Deference Markers")
    if "mean_deference_count" in filtered.columns:
        def_df = filtered[filtered["mean_deference_count"] > 0]
        if def_df.empty:
            st.info("No deference markers detected.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=def_df, x="seed_doc", y="mean_deference_count", hue="topology", palette=_PALETTE, ax=ax)
            ax.set_xlabel("Seed Document")
            ax.set_ylabel("Mean Deference Markers per Turn")
            ax.legend(title="Topology")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Deference data not available.")

# ---------------------------------------------------------------------------
# Raw JSON viewer
# ---------------------------------------------------------------------------
st.divider()
with st.expander("View raw result JSON files"):
    json_files = sorted(data_path.glob("*_result.json"))
    if json_files:
        selected_file = st.selectbox("Select file", [f.name for f in json_files])
        if selected_file:
            content = json.loads((data_path / selected_file).read_text(encoding="utf-8"))
            st.json(content)
    else:
        st.info("No result files found.")

