"""Run hierarchical topology trials with hallucination injection at L1.

Usage:
  python -m experiments.run_hierarchical --seed-doc tech_earnings --n-trials 30

  # Dry run:
  python -m experiments.run_hierarchical --seed-doc tech_earnings --mock
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEED_DOC_CHOICES = [
    "finance_earnings_alphabet_ai_capex_2026_v1",
    "geopolitics_sanctions_oil_supplyshock_2025_v1",
    "tech_earnings",
    "policy_draft",
    "geopolitical_event",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical MAS topology trials."
    )
    parser.add_argument(
        "--seed-doc",
        required=True,
        choices=SEED_DOC_CHOICES,
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--enable-db",
        action="store_true",
        help="Also persist seeds, runs, and messages to local Postgres.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection URL. Defaults to DATABASE_URL.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from src.config import Condition, ExperimentConfig, SeedDocument
    from src.tasks.predictive_intel import PredictiveIntelligenceTask

    config = ExperimentConfig(
        condition=Condition.HIERARCHICAL_HALLUCINATION,
        seed_doc=SeedDocument(args.seed_doc),
        n_trials=args.n_trials,
        gcp_project=("mock-project" if args.mock else os.getenv("GCP_PROJECT", "")),
        output_dir=Path(args.output_dir),
        enable_db_persistence=args.enable_db,
        database_url=args.database_url,
    )
    config.validate()

    if args.mock:
        from concordia.testing.mock_model import MockModel
        import json
        mock_json = json.dumps({
            "prediction_direction": "POSITIVE",
            "confidence": 0.90,
            "prediction_summary": "Mock orchestrator directive — hallucinated.",
            "key_factors": ["mock hallucination factor"],
        })
        model = MockModel(response=mock_json)
        logger.info("Using MockModel — no API calls.")
    else:
        from src.language_model import VertexAILanguageModel
        model = VertexAILanguageModel(
            project=config.gcp_project,
            location=config.gcp_location,
            temperature=config.temperature,
            requests_per_minute=config.rate_limit_rpm,
        )

    task = PredictiveIntelligenceTask(args.seed_doc)

    from src.game_master.simulation import SimulationRunner
    runner = SimulationRunner(model=model, config=config)

    logger.info(
        "Starting %d hierarchical trials | seed_doc=%s",
        args.n_trials,
        args.seed_doc,
    )

    for trial_id in range(args.n_trials):
        logger.info("Trial %d/%d ...", trial_id + 1, args.n_trials)
        path = runner.run_hierarchical_trial(task=task, trial_id=trial_id)
        logger.info("  → %s", path)

    logger.info("Done.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
