"""Run the complete experiment suite across all conditions.

Orchestrates:
  1. Flat baseline × each seed document
  2. Hierarchical × each seed document
  3. Whistleblower (low-rank) × each seed document
  4. Whistleblower (high-rank) × each seed document

Exports all results to ``data/`` as structured JSON, then runs the analysis
and visualisation pipeline.

CLI usage::

    python -m experiments.run_full_suite
    python -m experiments.run_full_suite --turns 3  # smoke test
"""
from __future__ import annotations

import argparse

SEED_DOCS = ["tech_earnings", "policy_draft", "geopolitical_event"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full experiment suite.")
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Override max_turns for all runs (e.g. 3 for a smoke test).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Directory for all trace and result JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: iterate over all conditions and seed documents."""
    args = parse_args()
    raise NotImplementedError(
        "Implement in Phase 3: call run_flat_baseline, run_hierarchical,"
        " run_whistleblower for each seed_doc, then run analysis pipeline."
    )


if __name__ == "__main__":
    main()
