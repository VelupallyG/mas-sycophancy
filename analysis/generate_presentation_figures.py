"""Generate presentation-ready figures for class slideshow."""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "real_v2"
OUT = DATA / "presentation"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
ORANGE = "#e67e22"
GRAY = "#95a5a6"
DARK = "#2c3e50"
LIGHT_GREEN = "#a9dfbf"
LIGHT_RED = "#f5b7b1"
PURPLE = "#9b59b6"


# ═══════════════════════════════════════════════════════════════════════════
# 1. TOPOLOGY DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════


def fig1_topology_diagrams():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── Flat topology ──
    G_flat = nx.Graph()
    nodes = [f"P{i + 1}" for i in range(21)]
    G_flat.add_nodes_from(nodes)
    # Connect every pair (just draw a subset of edges for clarity)
    pos_flat = nx.circular_layout(G_flat)
    # Draw a few cross-connections to suggest full connectivity
    edge_subset = []
    for i in range(21):
        for j in range(i + 1, 21):
            if (i + j) % 3 == 0:  # sparse subset for readability
                edge_subset.append((nodes[i], nodes[j]))
    G_flat.add_edges_from(edge_subset)

    nx.draw_networkx_edges(
        G_flat, pos_flat, ax=ax1, alpha=0.08, edge_color=GRAY, width=0.8
    )
    nx.draw_networkx_nodes(
        G_flat,
        pos_flat,
        ax=ax1,
        node_color=BLUE,
        node_size=350,
        edgecolors=DARK,
        linewidths=1.2,
    )
    ax1.set_title("Flat Topology (21 Peers)", fontsize=20, fontweight="bold", pad=15)
    ax1.text(
        0,
        0,
        "All-to-all\ncommunication",
        ha="center",
        va="center",
        fontsize=13,
        color=DARK,
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor=GRAY, alpha=0.9
        ),
    )
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.axis("off")

    # ── Hierarchical topology ──
    G_hier = nx.DiGraph()
    # Build tree
    l1 = "CSO"
    l2 = [f"M{i + 1}" for i in range(4)]
    l3 = [[f"A{i * 4 + j + 1}" for j in range(4)] for i in range(4)]

    G_hier.add_node(l1)
    for m in l2:
        G_hier.add_edge(l1, m)
    for i, m in enumerate(l2):
        for a in l3[i]:
            G_hier.add_edge(m, a)

    # Manual positions for clean tree
    pos_hier = {}
    pos_hier[l1] = (0, 2)
    for i, m in enumerate(l2):
        pos_hier[m] = (-3 + i * 2, 1)
    for i in range(4):
        for j in range(4):
            x = pos_hier[l2[i]][0] - 0.75 + j * 0.5
            pos_hier[l3[i][j]] = (x, 0)

    # Color by level
    colors = {l1: RED}
    for m in l2:
        colors[m] = ORANGE
    for group in l3:
        for a in group:
            colors[a] = BLUE

    node_colors = [colors[n] for n in G_hier.nodes()]
    nx.draw_networkx_edges(
        G_hier,
        pos_hier,
        ax=ax2,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color=DARK,
        width=1.5,
        alpha=0.7,
    )
    nx.draw_networkx_nodes(
        G_hier,
        pos_hier,
        ax=ax2,
        node_color=node_colors,
        node_size=350,
        edgecolors=DARK,
        linewidths=1.2,
    )
    # Labels for levels
    ax2.text(-5.5, 2, "L1", fontsize=14, fontweight="bold", color=RED, va="center")
    ax2.text(-5.5, 1, "L2", fontsize=14, fontweight="bold", color=ORANGE, va="center")
    ax2.text(-5.5, 0, "L3", fontsize=14, fontweight="bold", color=BLUE, va="center")

    ax2.text(5, 2, "1 Orchestrator", fontsize=11, color=GRAY, va="center")
    ax2.text(5, 1, "4 Managers", fontsize=11, color=GRAY, va="center")
    ax2.text(5, 0, "16 Analysts", fontsize=11, color=GRAY, va="center")

    ax2.set_title(
        "Hierarchical Topology (1→4→16)", fontsize=20, fontweight="bold", pad=15
    )
    ax2.set_xlim(-6, 6.5)
    ax2.set_ylim(-0.7, 2.7)
    ax2.axis("off")

    fig.suptitle(
        "Multi-Agent System Topologies — 21 Agents Each",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "01_topology_diagrams.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ 01_topology_diagrams.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONTAGION TREE (Turn 1 → 2 → 3)
# ═══════════════════════════════════════════════════════════════════════════


def fig2_contagion_tree():
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    l1 = "CSO"
    l2 = [f"M{i + 1}" for i in range(4)]
    l3 = [[f"A{i * 4 + j + 1}" for j in range(4)] for i in range(4)]
    all_l3 = [a for group in l3 for a in group]

    G = nx.DiGraph()
    G.add_node(l1)
    for m in l2:
        G.add_edge(l1, m)
    for i, m in enumerate(l2):
        for a in l3[i]:
            G.add_edge(m, a)

    pos = {}
    pos[l1] = (0, 2)
    for i, m in enumerate(l2):
        pos[m] = (-3 + i * 2, 1)
    for i in range(4):
        for j in range(4):
            x = pos[l2[i]][0] - 0.75 + j * 0.5
            pos[l3[i][j]] = (x, 0)

    # Turn-by-turn infection state (from real data)
    # Turn 1: only orchestrator NEGATIVE, 1/4 managers, 0/16 analysts
    # Turn 2: orchestrator + 4/4 managers + 14/16 analysts
    # Turn 3: all NEGATIVE
    infected = [
        {l1},  # Turn 1
        {l1} | set(l2) | set(all_l3[:14]),  # Turn 2 (14/16 analysts)
        {l1} | set(l2) | set(all_l3),  # Turn 3 (all)
    ]

    titles = [
        "Turn 1: Hallucination Injected",
        "Turn 2: Cascade Begins (88%)",
        "Turn 3: Total Collapse (100%)",
    ]

    for idx, ax in enumerate(axes):
        inf = infected[idx]
        node_colors = []
        for n in G.nodes():
            if n in inf:
                node_colors.append(RED)
            else:
                node_colors.append(GREEN)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            edge_color=DARK,
            width=1.5,
            alpha=0.6,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=300,
            edgecolors=DARK,
            linewidths=1.2,
        )

        ax.set_title(titles[idx], fontsize=16, fontweight="bold", pad=10)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.7, 2.7)
        ax.axis("off")

        # Count
        n_inf = len(inf)
        ax.text(
            0,
            -0.5,
            f"{n_inf}/21 agents adopt hallucination",
            ha="center",
            fontsize=12,
            color=RED if n_inf > 1 else DARK,
            fontweight="bold",
        )

    # Legend
    correct_patch = mpatches.Patch(color=GREEN, label="Correct stance (POSITIVE)")
    halluc_patch = mpatches.Patch(color=RED, label="Hallucinated stance (NEGATIVE)")
    fig.legend(
        handles=[correct_patch, halluc_patch],
        loc="lower center",
        ncol=2,
        fontsize=14,
        frameon=True,
        edgecolor=GRAY,
    )

    fig.suptitle(
        "Hallucination Contagion Through Hierarchy",
        fontsize=22,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUT / "02_contagion_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ 02_contagion_tree.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. SMOKING GUN — Flat vs Hierarchical Accuracy Over Turns
# ═══════════════════════════════════════════════════════════════════════════


def fig3_smoking_gun():
    # Real data from results
    turns = [1, 2, 3, 4, 5]

    # Flat baseline: 100% correct all turns
    flat_baseline = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Flat hallucination: ~95.2% (only injector agent wrong)
    flat_halluc = [20 / 21, 20 / 21, 20 / 21, 20 / 21, 20 / 21]

    # Hierarchical hallucination: catastrophic collapse
    # Turn 1: orchestrator wrong (1/21) + 1 manager → ~18/21 correct?
    # From data: Turn 1: 1 L1 NEG + 1 L2 NEG + 0 L3 NEG = 2 wrong → 19/21
    # Turn 2: 1 L1 + 4 L2 + 14 L3 = 19 wrong → 2/21
    # Turn 3-5: 21/21 wrong → 0/21
    hier_halluc = [19 / 21, 2 / 21, 0 / 21, 0 / 21, 0 / 21]

    # Hierarchical baseline: 100% correct
    hier_baseline = [1.0, 1.0, 1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        turns,
        flat_baseline,
        "o-",
        color=BLUE,
        linewidth=3,
        markersize=10,
        label="Flat (no hallucination)",
        zorder=5,
    )
    ax.plot(
        turns,
        flat_halluc,
        "s--",
        color=BLUE,
        linewidth=2.5,
        markersize=9,
        alpha=0.7,
        label="Flat + hallucination",
        zorder=4,
    )
    ax.plot(
        turns,
        hier_baseline,
        "o-",
        color=ORANGE,
        linewidth=3,
        markersize=10,
        label="Hierarchy (no hallucination)",
        zorder=5,
    )
    ax.plot(
        turns,
        hier_halluc,
        "s--",
        color=RED,
        linewidth=3.5,
        markersize=11,
        label="Hierarchy + hallucination",
        zorder=6,
    )

    # Annotate the collapse
    ax.annotate(
        "100% collapse\nby Turn 3",
        xy=(3, 0),
        xytext=(3.8, 0.3),
        fontsize=13,
        fontweight="bold",
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=2),
        ha="center",
    )

    # Shade the delta
    ax.fill_between(turns, flat_baseline, hier_halluc, alpha=0.1, color=RED)
    ax.text(
        3,
        0.5,
        "Δ² = 0.80",
        fontsize=16,
        fontweight="bold",
        color=RED,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.9
        ),
    )

    ax.set_xlabel("Turn", fontsize=16)
    ax.set_ylabel("Prediction Accuracy", fontsize=16)
    ax.set_title(
        "The Sycophancy Effect: Hierarchy Amplifies Hallucination",
        fontsize=20,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(turns)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="center right", fontsize=13, frameon=True, edgecolor=GRAY)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "03_smoking_gun.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ 03_smoking_gun.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SINGLE AGENT vs FLAT MAS vs HIERARCHICAL MAS
# ═══════════════════════════════════════════════════════════════════════════


def fig4_single_vs_mas():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Ground truth: +2.8%
    gt = 2.8

    # Single agent: mean 3.60% (no turns — just one shot)
    # Flat baseline by turn: [3.70, 3.48, 3.06, 1.58, 1.56] (approx from results)
    # Hierarchical hallucination by turn: [+3.1, -3.6, -4.8, -5.6, -5.83]
    turns = [1, 2, 3, 4, 5]
    single = [3.60] * 5  # constant line (single shot)
    flat = [3.70, 3.48, 3.06, 1.58, 1.56]
    hier_halluc = [3.1, -3.6, -4.8, -5.6, -5.83]

    ax.axhline(
        y=gt,
        color=GREEN,
        linewidth=2,
        linestyle=":",
        alpha=0.8,
        label=f"Ground Truth (+{gt}%)",
    )
    ax.axhline(y=0, color=GRAY, linewidth=0.5, linestyle="-", alpha=0.4)

    ax.plot(
        turns,
        single,
        "D-",
        color=PURPLE,
        linewidth=2.5,
        markersize=9,
        label="Single Agent (3.60%)",
    )
    ax.plot(
        turns,
        flat,
        "o-",
        color=BLUE,
        linewidth=3,
        markersize=10,
        label="Flat MAS (no hallucination)",
    )
    ax.plot(
        turns,
        hier_halluc,
        "s-",
        color=RED,
        linewidth=3,
        markersize=10,
        label="Hierarchical MAS + hallucination",
    )

    # Annotate Turn 3 sweet spot
    ax.annotate(
        "MAS beats single agent\n(Turn 3: 3.06% vs 3.60%)",
        xy=(3, 3.06),
        xytext=(4.2, 5.5),
        fontsize=11,
        color=BLUE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
        ha="center",
    )

    # Annotate hierarchy collapse
    ax.annotate(
        "Hierarchy: wrong direction",
        xy=(2, -3.6),
        xytext=(1.2, -7),
        fontsize=11,
        color=RED,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        ha="center",
    )

    ax.set_xlabel("Turn", fontsize=16)
    ax.set_ylabel("Mean Predicted Price Change (%)", fontsize=16)
    ax.set_title(
        "Single Agent vs. Multi-Agent Systems", fontsize=20, fontweight="bold", pad=15
    )
    ax.set_xticks(turns)
    ax.legend(loc="lower left", fontsize=12, frameon=True, edgecolor=GRAY)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "04_single_vs_mas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ 04_single_vs_mas.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE / METHODOLOGY DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════


def fig5_pipeline():
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis("off")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)

    box_style = dict(
        boxstyle="round,pad=0.5", facecolor="#eaf2f8", edgecolor=DARK, linewidth=2
    )
    inject_style = dict(
        boxstyle="round,pad=0.5", facecolor=LIGHT_RED, edgecolor=RED, linewidth=2
    )
    metric_style = dict(
        boxstyle="round,pad=0.5", facecolor="#d5f5e3", edgecolor=GREEN, linewidth=2
    )
    result_style = dict(
        boxstyle="round,pad=0.5", facecolor="#fdebd0", edgecolor=ORANGE, linewidth=2
    )

    # Row 1: Inputs
    ax.text(
        2,
        6.5,
        "Seed Document\n(Iran Sanctions\nIntelligence Packet)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=box_style,
    )
    ax.text(
        5.5,
        6.5,
        "Evidence Files\n(90 articles,\nEIA data, context)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=box_style,
    )
    ax.text(
        9,
        6.5,
        "Hallucination\nInjection\n(opposite direction)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=RED,
        bbox=inject_style,
    )

    # Arrow down
    for x in [2, 5.5, 9]:
        ax.annotate(
            "",
            xy=(x, 5.5),
            xytext=(x, 5.9),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2),
        )

    # Row 2: Agent system
    ax.text(
        5.5,
        4.8,
        "21-Agent MAS (Concordia + Gemini 2.5 Flash)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.6", facecolor="#d6eaf8", edgecolor=DARK, linewidth=2.5
        ),
    )

    # Sub-labels
    ax.text(
        2.5,
        4.0,
        "Flat Topology\n(all-to-all)",
        ha="center",
        va="center",
        fontsize=11,
        color=BLUE,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor=BLUE, linewidth=1.5
        ),
    )
    ax.text(
        8.5,
        4.0,
        "Hierarchical Topology\n(1→4→16 tree)",
        ha="center",
        va="center",
        fontsize=11,
        color=ORANGE,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor=ORANGE, linewidth=1.5
        ),
    )
    ax.text(
        5.5,
        4.0,
        "×",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=GRAY,
    )

    # Arrow down
    ax.annotate(
        "",
        xy=(5.5, 3.2),
        xytext=(5.5, 3.6),
        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2),
    )
    ax.text(
        7,
        3.4,
        "5 turns\nstructured JSON output",
        ha="left",
        va="center",
        fontsize=10,
        color=GRAY,
        style="italic",
    )

    # Row 3: Outputs
    ax.text(
        5.5,
        2.6,
        "JSONL Traces\n(agent_id, turn, direction,\nmagnitude, price_change_%)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=box_style,
    )

    # Arrow down to metrics
    ax.annotate(
        "",
        xy=(5.5, 1.6),
        xytext=(5.5, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2),
    )

    # Row 4: Metrics
    ax.text(
        2,
        1.0,
        "Δ² = 0.80\n(Sycophancy\nEffect Size)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=metric_style,
    )
    ax.text(
        5.5,
        1.0,
        "ToF = 2.0\n(Turn of Flip)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=metric_style,
    )
    ax.text(
        9,
        1.0,
        "NoF = 0.90\n(Flip Count)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=metric_style,
    )
    ax.text(
        12.5,
        1.0,
        "TRAIL\n100% Planning\nErrors",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=metric_style,
    )
    ax.text(
        15.5,
        1.0,
        "Deference\n63.8%\n(Linguistic)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=metric_style,
    )

    # Connect traces to metrics
    for x in [2, 5.5, 9, 12.5, 15.5]:
        ax.annotate(
            "",
            xy=(x, 1.5),
            xytext=(5.5, 1.6),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5, alpha=0.6),
        )

    # Title
    ax.set_title("Experiment Pipeline", fontsize=22, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(OUT / "05_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ 05_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Saving to {OUT}/\n")
    fig1_topology_diagrams()
    fig2_contagion_tree()
    fig3_smoking_gun()
    fig4_single_vs_mas()
    fig5_pipeline()
    print(f"\nAll 5 figures saved to {OUT}/")
