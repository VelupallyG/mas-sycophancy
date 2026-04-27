"""Run flat topology trials (baseline or with hallucination injection).

Usage:
  # Flat baseline (no hallucination — establishes A₀):
  python -m experiments.run_flat_baseline --seed-doc finance_earnings_alphabet_ai_capex_2026_v1 --n-trials 30

  # Flat with hallucination injection (structural control):
  python -m experiments.run_flat_baseline --seed-doc finance_earnings_alphabet_ai_capex_2026_v1 \\
      --inject-hallucination --n-trials 30

  # Dry run with MockModel (no API calls):
  python -m experiments.run_flat_baseline --seed-doc finance_earnings_alphabet_ai_capex_2026_v1 --mock
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEED_DOC_CHOICES = [
    "finance_earnings_alphabet_ai_capex_2026_v1",
    "iran_oil_sanctions_tightening_march_2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flat MAS topology trials.")
    parser.add_argument(
        "--seed-doc",
        required=True,
        choices=SEED_DOC_CHOICES,
        help="Seed document to use.",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--inject-hallucination",
        action="store_true",
        help="Inject hallucination into one random peer (flat structural control).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockModel — no API calls, for pipeline validation only.",
    )
    parser.add_argument("--output-dir", default="data", help="Root output directory.")
    parser.add_argument(
        "--enable-db",
        action="store_true",
        help="Also persist seeds, runs, and messages to local Postgres.",
    )
    parser.add_argument(
        "--enable-local-evidence",
        action="store_true",
        help="Retrieve local evidence from Postgres and inject it on turn 1.",
    )
    parser.add_argument(
        "--local-evidence-limit",
        type=int,
        default=5,
        help="Maximum local evidence documents to inject per trial.",
    )
    parser.add_argument(
        "--enable-file-evidence",
        action="store_true",
        help="Load evidence from local_evidence/ files and inject per-agent.",
    )
    parser.add_argument(
        "--evidence-docs-per-agent",
        type=int,
        default=5,
        help="Number of evidence documents allocated to each agent.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete trials instead of overwriting them.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from src.config import Condition, ExperimentConfig, SeedDocument
    from src.tasks.predictive_intel import PredictiveIntelligenceTask

    condition = (
        Condition.FLAT_HALLUCINATION
        if args.inject_hallucination
        else Condition.FLAT_BASELINE
    )

    config = ExperimentConfig(
        condition=condition,
        seed_doc=SeedDocument(args.seed_doc),
        n_trials=args.n_trials,
        gcp_project=("mock-project" if args.mock else os.getenv("GCP_PROJECT", "")),
        output_dir=Path(args.output_dir),
        enable_db_persistence=args.enable_db or args.enable_local_evidence,
        database_url=args.database_url,
        enable_local_evidence=args.enable_local_evidence,
        local_evidence_limit=args.local_evidence_limit,
        enable_file_evidence=args.enable_file_evidence,
        evidence_docs_per_agent=args.evidence_docs_per_agent,
    )
    config.validate()

    if args.mock:
        from concordia.testing.mock_model import MockModel
        import json

        mock_json = json.dumps(
            {
                "prediction_direction": "NEGATIVE",
                "predicted_magnitude": "HIGH",
                "predicted_price_change_pct": -8.5,
                "prediction_summary": "Mock prediction for pipeline validation.",
                "key_factors": ["mock factor 1", "mock factor 2"],
            }
        )
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
        "Starting %s trials | seed_doc=%s | inject_hallucination=%s",
        args.n_trials,
        args.seed_doc,
        args.inject_hallucination,
    )

    for trial_id in range(args.n_trials):
        if args.inject_hallucination:
            # K=3 reruns with different randomly selected injector agents.
            for k in range(config.n_flat_injection_reruns):
                logger.info(
                    "Trial %d/%d rerun %d/%d ...",
                    trial_id + 1,
                    args.n_trials,
                    k + 1,
                    config.n_flat_injection_reruns,
                )
                path = runner.run_flat_trial(
                    task=task,
                    trial_id=trial_id,
                    inject_hallucination=True,
                    injection_agent_seed=config.random_seed + trial_id * 100 + k,
                    rerun_id=k,
                    resume=args.resume,
                )
                logger.info("  → %s", path)
        else:
            logger.info("Trial %d/%d ...", trial_id + 1, args.n_trials)
            path = runner.run_flat_trial(
                task=task,
                trial_id=trial_id,
                inject_hallucination=False,
                resume=args.resume,
            )
            logger.info("  → %s", path)

    logger.info("Done. Condition: %s", condition.value)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
