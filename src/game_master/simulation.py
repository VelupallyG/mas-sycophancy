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
import random
from typing import Any

from src.config import ExperimentConfig
from src.tasks.predictive_intel import PredictiveIntelTask, SeedDocument
from src.topologies.hierarchical import HierarchicalTopology
from src.tracing.otel_exporter import OtelExporter


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

    @staticmethod
    def _infer_direction_from_text(text: str, fallback: str) -> str:
        candidate = text.lower()
        if any(token in candidate for token in ("crash", "drop", "decline", "down", "bankrupt")):
            return "negative"
        if any(token in candidate for token in ("rally", "surge", "rise", "up", "gain")):
            return "positive"
        if any(token in candidate for token in ("neutral", "flat", "mixed", "sideways")):
            return "neutral"
        return fallback

    @staticmethod
    def _agent_name(agent: Any) -> str:
        params = getattr(agent, "params", None)
        if isinstance(params, dict):
            name = params.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return str(agent)

    @staticmethod
    def _level_for_agent(agent: Any, levels: dict[int, list[Any]] | None) -> int | None:
        if levels is None:
            return None
        for level, members in levels.items():
            for member in members:
                if member is agent:
                    return level
        return None

    @staticmethod
    def _flatten_levels(levels: dict[int, list[Any]]) -> list[Any]:
        ordered: list[Any] = []
        for level in sorted(levels.keys()):
            ordered.extend(levels[level])
        return ordered

    @staticmethod
    def _determine_speaker(
        turn: int,
        agents: list[Any],
        levels: dict[int, list[Any]] | None,
        enforce_approval_chain: bool,
    ) -> Any:
        if enforce_approval_chain and levels is not None:
            chain = [5, 4, 3, 2, 1]
            target_level = chain[(turn - 1) % len(chain)]
            level_agents = levels.get(target_level, [])
            if level_agents:
                return level_agents[(turn - 1) % len(level_agents)]
        return agents[(turn - 1) % len(agents)]

    def _compose_agent_text(
        self,
        speaker_name: str,
        turn: int,
        stance: str,
        seed: SeedDocument,
    ) -> str:
        key_factor = seed.ground_truth_reaction.key_factors[0] if seed.ground_truth_reaction.key_factors else "evidence"
        return (
            f"{speaker_name} turn {turn}: prediction {stance} {seed.ground_truth_reaction.magnitude}; "
            f"supporting factor: {key_factor}."
        )

    def _simulate_stance(
        self,
        *,
        rng: random.Random,
        turn: int,
        expected_stance: str,
        hallucinated_stance: str,
        is_hierarchical: bool,
        hierarchy_level: int | None,
    ) -> str:
        if not is_hierarchical:
            return expected_stance if rng.random() < 0.9 else hallucinated_stance

        # In hierarchical settings, deference pressure grows over turns and is
        # stronger for lower-ranked agents.
        pressure = min(0.9, 0.25 + 0.06 * turn)
        if hierarchy_level is not None:
            if hierarchy_level >= 4:
                pressure += 0.10
            elif hierarchy_level == 1:
                pressure += 0.05
        pressure = min(0.95, pressure)
        return hallucinated_stance if rng.random() < pressure else expected_stance

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
        if not isinstance(task, PredictiveIntelTask):
            raise TypeError("task must be an instance of PredictiveIntelTask")

        seed = task.load_seed(self.config.experiment.seed_doc)
        max_turns = min(self.config.experiment.max_turns, 10)
        rng = random.Random(self.config.experiment.random_seed)

        is_hierarchical = isinstance(topology_agents, dict)
        levels = topology_agents if isinstance(topology_agents, dict) else None
        agents = self._flatten_levels(levels) if levels is not None else list(topology_agents)
        if not agents:
            raise ValueError("topology_agents must be non-empty")

        if self.config.enforce_approval_chain and not is_hierarchical:
            raise ValueError("approval chain can only be enforced for hierarchical topology")

        hallucinated_stance = self._infer_direction_from_text(
            seed.hallucinated_signal.expected_incorrect_prediction,
            fallback="negative" if seed.ground_truth_reaction.direction != "negative" else "positive",
        )

        exporter = OtelExporter(
            service_name=self.config.experiment.experiment_id,
            export_path=self.config.log_dir,
        )

        memory_store: dict[str, list[str]] = {self._agent_name(agent): [] for agent in agents}
        turn_records: list[dict[str, Any]] = []
        last_message = ""

        with exporter.start_span(
            "gm.state_transition",
            {
                "phase": "initialization",
                "seed_id": seed.id,
                "seed_doc": self.config.experiment.seed_doc,
                "max_turns": max_turns,
                "topology": "hierarchical" if is_hierarchical else "flat",
            },
        ):
            pass

        for turn in range(1, max_turns + 1):
            speaker = self._determine_speaker(
                turn=turn,
                agents=agents,
                levels=levels,
                enforce_approval_chain=self.config.enforce_approval_chain,
            )
            speaker_name = self._agent_name(speaker)
            hierarchy_level = self._level_for_agent(speaker, levels)

            stance = self._simulate_stance(
                rng=rng,
                turn=turn,
                expected_stance=seed.ground_truth_reaction.direction,
                hallucinated_stance=hallucinated_stance,
                is_hierarchical=is_hierarchical,
                hierarchy_level=hierarchy_level,
            )
            text = self._compose_agent_text(
                speaker_name=speaker_name,
                turn=turn,
                stance=stance,
                seed=seed,
            )

            with exporter.start_span(
                "agent.action",
                {
                    "agent_name": speaker_name,
                    "turn": turn,
                    "stance": stance,
                    "text": text,
                    "expected_stance": seed.ground_truth_reaction.direction,
                    "hierarchy_level": hierarchy_level,
                },
            ):
                pass

            memory_store[speaker_name].append(text)
            with exporter.start_span(
                "agent.memory_update",
                {
                    "agent_name": speaker_name,
                    "turn": turn,
                    "memory_size": len(memory_store[speaker_name]),
                    "memory_item": text,
                },
            ):
                pass

            if is_hierarchical and hierarchy_level is not None and hierarchy_level > 1 and levels is not None:
                receiver_candidates = levels.get(hierarchy_level - 1, [])
                receiver = receiver_candidates[turn % len(receiver_candidates)] if receiver_candidates else None
                receiver_name = self._agent_name(receiver) if receiver is not None else ""
            else:
                receiver = agents[turn % len(agents)]
                receiver_name = self._agent_name(receiver)

            with exporter.start_span(
                "agent.communication",
                {
                    "agent_name": speaker_name,
                    "turn": turn,
                    "recipient": receiver_name,
                    "message": text,
                    "hierarchy_level": hierarchy_level,
                },
            ):
                pass

            if receiver_name:
                memory_store[receiver_name].append(text)

            turn_records.append(
                {
                    "agent_name": speaker_name,
                    "turn": turn,
                    "stance": stance,
                    "text": text,
                    "expected_stance": seed.ground_truth_reaction.direction,
                    "hierarchy_level": hierarchy_level,
                }
            )
            last_message = text

            with exporter.start_span(
                "gm.state_transition",
                {
                    "phase": "turn_complete",
                    "turn": turn,
                    "speaker": speaker_name,
                },
            ):
                pass

        final_speaker_name = turn_records[-1]["agent_name"] if turn_records else ""
        if is_hierarchical and levels is not None:
            if not HierarchicalTopology().enforce_orchestrator_consensus(final_speaker_name, levels):
                orchestrator = levels[1][0]
                orchestrator_name = self._agent_name(orchestrator)
                final_speaker_name = orchestrator_name
                final_stance = hallucinated_stance
                final_text = self._compose_agent_text(
                    speaker_name=orchestrator_name,
                    turn=max_turns,
                    stance=final_stance,
                    seed=seed,
                )
                with exporter.start_span(
                    "gm.state_transition",
                    {
                        "phase": "consensus_override",
                        "reason": "hierarchical_approval_chain",
                        "consensus_speaker": orchestrator_name,
                    },
                ):
                    pass
                turn_records.append(
                    {
                        "agent_name": orchestrator_name,
                        "turn": max_turns,
                        "stance": final_stance,
                        "text": final_text,
                        "expected_stance": seed.ground_truth_reaction.direction,
                        "hierarchy_level": 1,
                    }
                )
                last_message = final_text

        consensus_prediction = last_message
        accuracy = task.evaluate(consensus_prediction, seed.ground_truth_reaction)

        with exporter.start_span(
            "gm.state_transition",
            {
                "phase": "completed",
                "consensus_speaker": final_speaker_name,
                "accuracy": accuracy,
            },
        ):
            pass

        trace_file = exporter.export_trace(
            f"{self.config.experiment.experiment_id}_trace.json"
        )

        if self.config.verbose:
            print(
                f"[{self.config.experiment.experiment_id}] "
                f"seed={self.config.experiment.seed_doc} accuracy={accuracy:.3f} "
                f"trace={trace_file}"
            )

        return SimulationResult(
            experiment_id=self.config.experiment.experiment_id,
            consensus_prediction=consensus_prediction,
            accuracy=accuracy,
            trace_path=str(trace_file),
            agent_turn_records=turn_records,
            metadata={
                "seed_id": seed.id,
                "seed_doc": self.config.experiment.seed_doc,
                "ground_truth_direction": seed.ground_truth_reaction.direction,
                "ground_truth_magnitude": seed.ground_truth_reaction.magnitude,
                "topology": "hierarchical" if is_hierarchical else "flat",
                "max_turns": max_turns,
            },
        )
