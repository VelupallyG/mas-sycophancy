"""Concordia Game Master configuration and top-level simulation runner.

IMPORTANT ARCHITECTURAL NOTE (from CLAUDE.md §2):
  The Game Master (GM) is the *objective simulation engine*.  It manages state,
  enforces turn order, maintains ground truth, and exports traces.
  The Orchestrator is a *participating agent* (the CEO) INSIDE the simulation.
  Never confuse these roles.

The GM:
  - Reads the seed document to establish ground truth
  - Enforces topology constraints (hierarchical approval chain / flat round-robin)
  - Caps simulation at T=10 turns
  - Exports structured JSON logs via ``OtelExporter`` after every event

Usage::

    config = GameMasterConfig(experiment=exp_config)
    sim = Simulation(config)
    result = sim.run(topology_agents, task)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import ExperimentConfig


@dataclass
class GameMasterConfig:
    """Configuration for the Concordia Game Master.

    Attributes:
        experiment: Top-level experiment configuration.
        enforce_approval_chain: If ``True`` (hierarchical mode), the GM
            routes upward communications through the reporting chain.
        log_dir: Directory for OTel JSON trace output.
        verbose: If ``True``, print turn summaries to stdout during the run.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    enforce_approval_chain: bool = False
    log_dir: str = "data/"
    verbose: bool = False


@dataclass
class SimulationResult:
    """Output of a completed simulation run.

    Attributes:
        experiment_id: Matches ``GameMasterConfig.experiment.experiment_id``.
        consensus_prediction: The final consensus text produced by the MAS.
        accuracy: Score in [0, 1] from ``PredictiveIntelTask.evaluate``.
        trace_path: Absolute path to the OTel JSON trace file.
        agent_turn_records: Raw per-agent per-turn stance data (used by
            metrics pipeline without re-parsing the trace file).
        metadata: Arbitrary additional fields for future extensibility.
    """

    experiment_id: str
    consensus_prediction: str
    accuracy: float
    trace_path: str
    agent_turn_records: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Simulation:
    """Orchestrates a single MAS simulation run using Concordia.

    Args:
        config: Game Master configuration.
    """

    def __init__(self, config: GameMasterConfig) -> None:
        self.config = config

    def run(
        self,
        topology_agents: list[Any] | dict[int, list[Any]],
        task: Any,
    ) -> SimulationResult:
        """Execute the simulation and return results.

        Runs for at most ``config.experiment.max_turns`` turns.  At each
        turn, agents act in the order specified by the topology (round-robin
        for flat; bottom-up for hierarchical).  The GM records all events via
        ``OtelExporter`` and evaluates the final consensus against ground truth.

        Args:
            topology_agents: Either a flat list of agents (flat topology) or a
                dict mapping level → list of agents (hierarchical topology).
            task: A ``PredictiveIntelTask`` instance loaded with the seed doc.

        Returns:
            ``SimulationResult`` with accuracy, consensus text, and trace path.
        """
        raise NotImplementedError
