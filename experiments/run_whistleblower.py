"""Run whistleblower intervention variants (RQ3 and RQ4).

CLI usage::

    python -m experiments.run_whistleblower --rank low --seed-doc tech_earnings
    python -m experiments.run_whistleblower --rank high --seed-doc tech_earnings
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical MAS with a Whistleblower agent injected."
    )
    parser.add_argument(
        "--rank",
        required=True,
        choices=["low", "high"],
        help="Whistleblower rank: 'low' = Level 5 analyst, 'high' = Level 2 director.",
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
    """Entry point: build hierarchical topology + whistleblower, run simulation."""
    args = parse_args()
    _rank_level = 5 if args.rank == "low" else 2  # noqa: F841 — used in implementation
    raise NotImplementedError(
        "Implement in Session 7: wire WhistleblowerAgent into HierarchicalTopology"
    )


if __name__ == "__main__":
    main()
