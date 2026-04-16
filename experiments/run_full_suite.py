"""Orchestrate the complete experiment suite.

Runs all three conditions × the current benchmark seed documents × N trials each.
Conditions:
  1. Flat baseline (no hallucination)     — establishes A₀
  2. Flat with hallucination injection    — structural control
  3. Hierarchical with hallucination      — experimental

Parallelism:
  - Trials within a condition are independent and run sequentially here
    (concurrency is the Phase 2 optimisation once the pipeline is validated).
  - Conditions and seed documents run sequentially for simplicity.

Usage:
  python -m experiments.run_full_suite --n-trials 30
  python -m experiments.run_full_suite --mock  # pipeline smoke-test
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEED_DOCS = [
    "finance_earnings_alphabet_ai_capex_2026_v1",
    "geopolitics_sanctions_oil_supplyshock_2025_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full experiment suite.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", default="data")
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
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres connection URL. Defaults to DATABASE_URL.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from src.config import Condition, ExperimentConfig, SeedDocument
    from src.tasks.predictive_intel import PredictiveIntelligenceTask
    from src.game_master.simulation import SimulationRunner

    base_config = ExperimentConfig(
        condition=Condition.FLAT_BASELINE,
        seed_doc=SeedDocument.FINANCE_EARNINGS_ALPHABET_AI_CAPEX_2026_V1,
        n_trials=args.n_trials,
        gcp_project=("mock-project" if args.mock else os.getenv("GCP_PROJECT", "")),
        output_dir=Path(args.output_dir),
        enable_db_persistence=args.enable_db or args.enable_local_evidence,
        database_url=args.database_url,
        enable_local_evidence=args.enable_local_evidence,
        local_evidence_limit=args.local_evidence_limit,
    )
    base_config.validate()

    if args.mock:
        from concordia.testing.mock_model import MockModel
        import json
        model = MockModel(response=json.dumps({
            "prediction_direction": "NEGATIVE",
            "confidence": 0.75,
            "prediction_summary": "Mock.",
            "key_factors": ["mock"],
        }))
    else:
        from src.language_model import VertexAILanguageModel
        model = VertexAILanguageModel(
            project=base_config.gcp_project,
            location=base_config.gcp_location,
            temperature=base_config.temperature,
            requests_per_minute=base_config.rate_limit_rpm,
        )

    total_trials = len(SEED_DOCS) * (
        args.n_trials
        + (args.n_trials * base_config.n_flat_injection_reruns)
        + args.n_trials
    )
    logger.info(
        "Full suite: %d total trials across 3 conditions × %d seeds.",
        total_trials,
        len(SEED_DOCS),
    )

    for seed_doc_name in SEED_DOCS:
        task = PredictiveIntelligenceTask(seed_doc_name)
        seed_doc_enum = SeedDocument(seed_doc_name)

        logger.info("=== Seed document: %s ===", seed_doc_name)

        # Condition 1: Flat baseline
        config = ExperimentConfig(
            condition=Condition.FLAT_BASELINE,
            seed_doc=seed_doc_enum,
            n_trials=args.n_trials,
            gcp_project=base_config.gcp_project,
            output_dir=base_config.output_dir,
            enable_db_persistence=base_config.enable_db_persistence,
            database_url=base_config.database_url,
            enable_local_evidence=base_config.enable_local_evidence,
            local_evidence_limit=base_config.local_evidence_limit,
        )
        config.validate()
        runner = SimulationRunner(model=model, config=config)
        logger.info("--- Flat baseline ---")
        for trial_id in range(args.n_trials):
            runner.run_flat_trial(task=task, trial_id=trial_id, inject_hallucination=False)

        # Condition 2: Flat with hallucination (K=3 reruns per trial)
        config = ExperimentConfig(
            condition=Condition.FLAT_HALLUCINATION,
            seed_doc=seed_doc_enum,
            n_trials=args.n_trials,
            gcp_project=base_config.gcp_project,
            output_dir=base_config.output_dir,
            enable_db_persistence=base_config.enable_db_persistence,
            database_url=base_config.database_url,
            enable_local_evidence=base_config.enable_local_evidence,
            local_evidence_limit=base_config.local_evidence_limit,
        )
        config.validate()
        runner = SimulationRunner(model=model, config=config)
        logger.info("--- Flat with hallucination ---")
        for trial_id in range(args.n_trials):
            for k in range(config.n_flat_injection_reruns):
                runner.run_flat_trial(
                    task=task,
                    trial_id=trial_id,
                    inject_hallucination=True,
                    injection_agent_seed=config.random_seed + trial_id * 100 + k,
                    rerun_id=k,
                )

        # Condition 3: Hierarchical with hallucination
        config = ExperimentConfig(
            condition=Condition.HIERARCHICAL_HALLUCINATION,
            seed_doc=seed_doc_enum,
            n_trials=args.n_trials,
            gcp_project=base_config.gcp_project,
            output_dir=base_config.output_dir,
            enable_db_persistence=base_config.enable_db_persistence,
            database_url=base_config.database_url,
            enable_local_evidence=base_config.enable_local_evidence,
            local_evidence_limit=base_config.local_evidence_limit,
        )
        config.validate()
        runner = SimulationRunner(model=model, config=config)
        logger.info("--- Hierarchical with hallucination ---")
        for trial_id in range(args.n_trials):
            runner.run_hierarchical_trial(task=task, trial_id=trial_id)

    logger.info("Full suite complete.")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
