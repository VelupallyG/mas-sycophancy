"""Run a hierarchical-topology experiment with hallucination injection.

CLI usage::

    python -m experiments.run_hierarchical --seed-doc tech_earnings
    python -m experiments.run_hierarchical --seed-doc geopolitical_event --turns 3
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from src.config import ExperimentConfig, load_config_from_env
from src.game_master.simulation import GameMasterConfig, Simulation
from src.hallucination.injector import HallucinationInjector
from src.model import build_gemini_model
from src.tasks.predictive_intel import PredictiveIntelTask
from src.topologies.hierarchical import HierarchicalTopology


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

    config: ExperimentConfig = load_config_from_env()
    config.seed_doc = args.seed_doc
    config.output_dir = args.output_dir
    config.max_turns = min(args.turns, 10) if args.turns is not None else min(config.max_turns, 10)
    config.experiment_id = (
        f"hier_{args.seed_doc}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    model = build_gemini_model(config.agent)
    task = PredictiveIntelTask()
    seed = task.load_seed(config.seed_doc)
    injector = HallucinationInjector(config.hallucination)
    hallucinated_signal = injector.inject(seed)

    topology_agents = HierarchicalTopology().build(
        config,
        hallucinated_premise=hallucinated_signal.fabricated_claim,
    )

    simulation = Simulation(
        GameMasterConfig(
            experiment=config,
            enforce_approval_chain=True,
            log_dir=config.output_dir,
            verbose=True,
        ),
        model=model,
    )
    result = simulation.run(topology_agents=topology_agents, task=task)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{config.experiment_id}_result.json"
    payload = {
        "experiment_id": result.experiment_id,
        "condition": "influenced",
        "seed_doc": config.seed_doc,
        "hallucinated_claim": hallucinated_signal.fabricated_claim,
        "accuracy": result.accuracy,
        "consensus_prediction": result.consensus_prediction,
        "trace_path": result.trace_path,
        "agent_turn_records": result.agent_turn_records,
        "metadata": result.metadata,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    print(f"Result JSON: {result_path.resolve()}")
    print(f"Trace JSON: {result.trace_path}")
    print(f"Accuracy: {result.accuracy:.3f}")


if __name__ == "__main__":
    main()
