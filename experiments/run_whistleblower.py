"""Run whistleblower intervention variants (RQ3 and RQ4).

CLI usage::

    python -m experiments.run_whistleblower --rank low --seed-doc tech_earnings
    python -m experiments.run_whistleblower --rank high --seed-doc tech_earnings
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from src.agents.whistleblower import rank_from_variant
from src.agents.whistleblower_prefab import WhistleblowerPrefab
from src.config import ExperimentConfig, load_config_from_env
from src.game_master.simulation import GameMasterConfig, Simulation
from src.hallucination.injector import HallucinationInjector
from src.model import build_gemini_model
from src.tasks.predictive_intel import PredictiveIntelTask
from src.topologies.hierarchical import HierarchicalTopology, HierarchyLevels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def inject_whistleblower(
    levels: HierarchyLevels,
    *,
    rank: int,
    name: str = "Whistleblower",
) -> HierarchyLevels:
    """Inject a whistleblower prefab by replacing one agent at target level.

    The replacement preserves total level counts while varying only the
    whistleblower's rank authority.
    """
    if rank not in {2, 5}:
        raise ValueError("whistleblower rank must be 2 or 5 for experiment variants")

    if rank not in levels or not levels[rank]:
        raise ValueError(f"cannot inject whistleblower at missing level {rank}")

    replaced = levels[rank][0]
    replaced_goal = replaced.params.get("goal", "")
    levels[rank][0] = WhistleblowerPrefab(
        params={
            "name": name,
            "rank": rank,
            "goal": replaced_goal,
        }
    )
    return levels


def main() -> None:
    """Entry point: build hierarchical topology + whistleblower, run simulation."""
    args = parse_args()

    config: ExperimentConfig = load_config_from_env()
    config.seed_doc = args.seed_doc
    config.output_dir = args.output_dir
    config.max_turns = min(args.turns, 10) if args.turns is not None else min(config.max_turns, 10)
    config.experiment_id = (
        f"whistle_{args.rank}_{args.seed_doc}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    rank_level = rank_from_variant(args.rank)
    model = build_gemini_model(config.agent)

    task = PredictiveIntelTask()
    seed = task.load_seed(config.seed_doc)
    injector = HallucinationInjector(config.hallucination)
    hallucinated_signal = injector.inject(seed)

    levels = HierarchicalTopology().build(
        config,
        hallucinated_premise=hallucinated_signal.fabricated_claim,
    )
    topology_agents = inject_whistleblower(levels, rank=rank_level)

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
        "condition": "whistleblower",
        "seed_doc": config.seed_doc,
        "whistleblower_rank_variant": args.rank,
        "whistleblower_rank": rank_level,
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
