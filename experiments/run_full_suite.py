"""Orchestrate the complete experiment suite.

Runs all three conditions × all three seed documents × N trials each.
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
import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEED_DOCS = ["tech_earnings", "policy_draft", "geopolitical_event"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full experiment suite.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", default="data")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    from src.config import Condition, ExperimentConfig, SeedDocument
    from src.tasks.predictive_intel import PredictiveIntelligenceTask
    from src.game_master.simulation import SimulationRunner

    base_config = ExperimentConfig(
        condition=Condition.FLAT_BASELINE,
        seed_doc=SeedDocument.TECH_EARNINGS,
        n_trials=args.n_trials,
        gcp_project=("mock-project" if args.mock else os.getenv("GCP_PROJECT", "")),
        output_dir=Path(args.output_dir),
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
    logger.info("Full suite: %d total trials across 3 conditions × 3 seeds.", total_trials)

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
        )
        config.validate()
        runner = SimulationRunner(model=model, config=config)
        logger.info("--- Hierarchical with hallucination ---")
        for trial_id in range(args.n_trials):
            runner.run_hierarchical_trial(task=task, trial_id=trial_id)

    logger.info("Full suite complete.")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
