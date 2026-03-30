"""Run a hierarchical-topology experiment with hallucination injection.

CLI usage::

    python -m experiments.run_hierarchical --seed-doc tech_earnings
    python -m experiments.run_hierarchical --seed-doc geopolitical_event --turns 3
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical MAS experiment with hallucination injection."
    )
    parser.add_argument(
        "--seed-doc",
        required=True,
        choices=["tech_earnings", "policy_draft", "geopolitical_event"],
        help="Seed document to use for this run.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override max_turns from config (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Directory for trace and result JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build hierarchical topology, inject hallucination, run simulation."""
    args = parse_args()
    raise NotImplementedError(
        "Implement in Session 6: wire HierarchicalTopology + HallucinationInjector"
        " + Simulation + PredictiveIntelTask"
    )


if __name__ == "__main__":
    main()
