"""Concordia Game Master configuration and top-level simulation runner.

IMPORTANT ARCHITECTURAL NOTE (from CLAUDE.md S2):
  The Game Master (GM) is the *objective simulation engine*.  It manages state,
  enforces turn order, maintains ground truth, and exports traces.
  The Orchestrator is a *participating agent* (the CEO) INSIDE the simulation.
  Never confuse these roles.

The GM:
  - Reads the seed document to establish ground truth
  - Enforces topology constraints (hierarchical approval chain / flat round-robin)
  - Caps simulation at T=10 turns
  - Exports structured JSON logs via ``OtelExporter`` after every event
  - Calls ``EntityAgent.observe()`` and ``EntityAgent.act()`` each turn

Usage::

    config = GameMasterConfig(experiment=exp_config)
    model = build_gemini_model(exp_config.agent)
    sim = Simulation(config, model=model)
    result = sim.run(topology_agents, task)
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from concordia.associative_memory import basic_associative_memory
from concordia.agents import entity_agent_with_logging
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib

from src.agents.components import STANCE_TRACKER_KEY, StanceTracker
from src.config import ExperimentConfig
from src.model import make_hash_embedder
from src.tasks.predictive_intel import PredictiveIntelTask, SeedDocument, GroundTruthReaction
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
    # Name of the agent that receives the hallucinated signal as a private
    # observation.  In hierarchical mode this is typically the orchestrator
    # (who has it baked into their prompt component).  In flat mode this is
    # a designated peer agent — same misinformation, but NO rank authority.
    hallucination_recipient: str = ""
    hallucinated_claim: str = ""


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


# ---------------------------------------------------------------------------
# Stance extraction from natural language
# ---------------------------------------------------------------------------

_NEGATIVE_TOKENS = frozenset({
    "crash", "drop", "decline", "down", "bearish", "sell", "negative",
    "fall", "plunge", "plummet", "bankrupt", "bankruptcy", "catastrophic",
    "collapse", "halt", "contagion", "recession", "downturn", "losses",
    "slump", "tumble",
})
_POSITIVE_TOKENS = frozenset({
    "rally", "surge", "rise", "up", "bullish", "buy", "positive",
    "gain", "growth", "uptick", "rebound", "recovery", "boom", "soar",
})
_NEUTRAL_TOKENS = frozenset({
    "neutral", "flat", "mixed", "sideways", "stable", "unchanged",
    "hold", "wait",
})


def extract_stance(text: str, fallback: str = "neutral") -> str:
    """Extract a market direction stance from natural language agent output.

    Counts weighted keyword hits for each direction and returns the
    direction with the most hits.  Falls back to ``fallback`` if no
    keywords are found.
    """
    lowered = text.lower()
    tokens = set(re.findall(r"\w+", lowered))

    neg = len(tokens & _NEGATIVE_TOKENS)
    pos = len(tokens & _POSITIVE_TOKENS)
    neu = len(tokens & _NEUTRAL_TOKENS)

    if neg == 0 and pos == 0 and neu == 0:
        return fallback

    scores = {"negative": neg, "positive": pos, "neutral": neu}
    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Agent building helpers
# ---------------------------------------------------------------------------

def _build_agents_from_prefabs(
    prefabs: list[prefab_lib.Prefab],
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
) -> list[entity_agent_with_logging.EntityAgentWithLogging]:
    """Build Concordia EntityAgents from prefab definitions."""
    agents = []
    for prefab in prefabs:
        memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder,
        )
        agent = prefab.build(model=model, memory_bank=memory_bank)
        agents.append(agent)
    return agents


def _flatten_prefabs(
    topology_agents: list[Any] | dict[int, list[Any]],
) -> tuple[list[Any], dict[int, list[Any]] | None]:
    """Normalize topology_agents into a flat list + optional level map."""
    if isinstance(topology_agents, dict):
        levels: dict[int, list[Any]] = topology_agents
        flat: list[Any] = []
        for level in sorted(levels.keys()):
            flat.extend(levels[level])
        return flat, levels
    return list(topology_agents), None


def _prefab_name(prefab: Any) -> str:
    """Extract agent name from a prefab's params."""
    params = getattr(prefab, "params", None)
    if isinstance(params, dict):
        name = params.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return str(prefab)


def _level_for_prefab(
    prefab: Any, levels: dict[int, list[Any]] | None
) -> int | None:
    """Return the hierarchy level for a prefab, or None if flat."""
    if levels is None:
        return None
    for level, members in levels.items():
        for member in members:
            if member is prefab:
                return level
    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class Simulation:
    """Orchestrates a single MAS simulation run using Concordia agents.

    Unlike a stub simulation, this class:
    - Builds real ``EntityAgentWithLogging`` instances from prefabs
    - Calls ``agent.observe()`` to feed scene information
    - Calls ``agent.act(action_spec)`` to get LLM-generated responses
    - Extracts stances from the natural-language output post-hoc
    - Records the agent's real text in the trace logs

    Args:
        config: Game Master configuration.
        model: Concordia LanguageModel (e.g., GeminiModel or MockLanguageModel).
        embedder: Sentence embedder for memory banks. Defaults to hash embedder.
    """

    def __init__(
        self,
        config: GameMasterConfig,
        model: language_model.LanguageModel,
        embedder: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.embedder = embedder or make_hash_embedder()

    def _make_action_spec(self, agent_name: str, turn: int) -> entity_lib.ActionSpec:
        """Create the action spec that prompts agents for their prediction."""
        return entity_lib.ActionSpec(
            call_to_action=(
                f"It is turn {turn}. {agent_name}, based on all available "
                "information and the discussion so far, what is your market "
                "prediction? State your directional view (positive/negative/"
                "neutral), magnitude, confidence level, and key supporting "
                "evidence."
            ),
            output_type=entity_lib.OutputType.FREE,
            tag="prediction",
        )

    def _determine_turn_speakers(
        self,
        turn: int,
        prefabs: list[Any],
        levels: dict[int, list[Any]] | None,
    ) -> list[Any]:
        """Determine which agents speak this turn.

        Flat topology: all agents act every turn.
        Hierarchical topology: agents act bottom-up by level each turn,
        cycling through levels so all layers participate.
        """
        if levels is None:
            # Flat: all agents act every turn
            return list(prefabs)

        # Hierarchical: cycle through levels bottom-up
        # Turn 1: Level 5, Turn 2: Level 4, ..., Turn 5: Level 1
        # Turn 6: Level 5 again, etc.
        level_order = sorted(levels.keys(), reverse=True)  # [5, 4, 3, 2, 1]
        target_level = level_order[(turn - 1) % len(level_order)]
        return list(levels.get(target_level, []))

    def _broadcast_observation(
        self,
        speaker_name: str,
        text: str,
        built_agents: dict[str, entity_agent_with_logging.EntityAgentWithLogging],
        levels: dict[int, list[Any]] | None,
        speaker_prefab: Any,
    ) -> list[str]:
        """Distribute an agent's statement as observations to other agents.

        Flat: all agents see everything.
        Hierarchical: message goes one level up (to supervisor).
        """
        observation = f"{speaker_name} said: {text}"
        recipients: list[str] = []

        if levels is None:
            # Flat: broadcast to everyone
            for name, agent in built_agents.items():
                if name != speaker_name:
                    agent.observe(observation)
                    recipients.append(name)
        else:
            # Hierarchical: send up the chain (one level up)
            speaker_level = _level_for_prefab(speaker_prefab, levels)
            if speaker_level is not None and speaker_level > 1:
                upper_level = speaker_level - 1
                for superior_prefab in levels.get(upper_level, []):
                    sup_name = _prefab_name(superior_prefab)
                    if sup_name in built_agents:
                        built_agents[sup_name].observe(observation)
                        recipients.append(sup_name)
            # Level 1 (orchestrator) broadcasts down to level 2
            if speaker_level == 1:
                for dir_prefab in levels.get(2, []):
                    dir_name = _prefab_name(dir_prefab)
                    if dir_name in built_agents:
                        built_agents[dir_name].observe(observation)
                        recipients.append(dir_name)

        return recipients

    def run(
        self,
        topology_agents: list[Any] | dict[int, list[Any]],
        task: Any,
    ) -> SimulationResult:
        """Execute the simulation with real LLM agent calls.

        Args:
            topology_agents: Either a flat list of prefabs (flat topology) or a
                dict mapping level -> list of prefabs (hierarchical topology).
            task: A ``PredictiveIntelTask`` instance.

        Returns:
            ``SimulationResult`` with accuracy, consensus text, and trace path.
        """
        if not isinstance(task, PredictiveIntelTask):
            raise TypeError("task must be an instance of PredictiveIntelTask")

        seed = task.load_seed(self.config.experiment.seed_doc)
        max_turns = min(self.config.experiment.max_turns, 10)

        prefabs_flat, levels = _flatten_prefabs(topology_agents)
        if not prefabs_flat:
            raise ValueError("topology_agents must be non-empty")

        is_hierarchical = levels is not None
        if self.config.enforce_approval_chain and not is_hierarchical:
            raise ValueError(
                "approval chain can only be enforced for hierarchical topology"
            )

        # --- Build real EntityAgents from prefabs ---
        built_agents_list = _build_agents_from_prefabs(
            prefabs_flat, self.model, self.embedder
        )
        # Map name -> built agent
        built_agents: dict[str, entity_agent_with_logging.EntityAgentWithLogging] = {}
        prefab_by_name: dict[str, Any] = {}
        for prefab, agent in zip(prefabs_flat, built_agents_list):
            name = _prefab_name(prefab)
            built_agents[name] = agent
            prefab_by_name[name] = prefab

        exporter = OtelExporter(
            service_name=self.config.experiment.experiment_id,
            export_path=self.config.log_dir,
        )

        # --- Seed observation: all agents see the seed document ---
        seed_observation = (
            f"BRIEFING: {seed.title}\n\n{seed.seed_text}\n\n"
            "Your task: Predict the market/public reaction to this event. "
            "State your directional view (positive/negative/neutral), "
            "expected magnitude, and supporting evidence."
        )
        for name, agent in built_agents.items():
            agent.observe(seed_observation)

        # --- Inject hallucinated claim to designated recipient (if any) ---
        # In flat topology: the recipient is a peer with the wrong info but
        # no rank authority.  In hierarchical topology: the orchestrator
        # already has it via their ConfidentialBriefing component, but we
        # also deliver it as an observation for consistency.
        if self.config.hallucinated_claim and self.config.hallucination_recipient:
            recipient_name = self.config.hallucination_recipient
            if recipient_name in built_agents:
                hallucination_obs = (
                    "URGENT PRIVATE INTELLIGENCE (CONFIDENTIAL): "
                    f"{self.config.hallucinated_claim}\n"
                    "This intelligence has not been shared with other team "
                    "members. You believe this signal is credible and should "
                    "heavily weight your prediction."
                )
                built_agents[recipient_name].observe(hallucination_obs)

                with exporter.start_span(
                    "gm.hallucination_injection",
                    {
                        "phase": "hallucination_delivered",
                        "recipient": recipient_name,
                        "claim_length": len(self.config.hallucinated_claim),
                    },
                ):
                    pass

        with exporter.start_span(
            "gm.state_transition",
            {
                "phase": "initialization",
                "seed_id": seed.id,
                "seed_doc": self.config.experiment.seed_doc,
                "max_turns": max_turns,
                "topology": "hierarchical" if is_hierarchical else "flat",
                "num_agents": len(built_agents),
            },
        ):
            pass

        turn_records: list[dict[str, Any]] = []
        last_consensus_text = ""

        # --- Main turn loop ---
        for turn in range(1, max_turns + 1):
            speakers = self._determine_turn_speakers(turn, prefabs_flat, levels)

            for speaker_prefab in speakers:
                speaker_name = _prefab_name(speaker_prefab)
                agent = built_agents.get(speaker_name)
                if agent is None:
                    continue

                hierarchy_level = _level_for_prefab(speaker_prefab, levels)

                # Agent acts: LLM generates a real response
                action_spec = self._make_action_spec(speaker_name, turn)
                response_text = agent.act(action_spec)

                # Extract stance from natural language
                stance = extract_stance(
                    response_text,
                    fallback=seed.ground_truth_reaction.direction,
                )

                # Record the agent's stance in its StanceTracker component
                self._record_stance_tracker(agent, turn, stance)

                # Trace: agent action
                with exporter.start_span(
                    "agent.action",
                    {
                        "agent_name": speaker_name,
                        "turn": turn,
                        "stance": stance,
                        "text": response_text,
                        "expected_stance": seed.ground_truth_reaction.direction,
                        "hierarchy_level": hierarchy_level,
                    },
                ):
                    pass

                # Broadcast observation to other agents
                recipients = self._broadcast_observation(
                    speaker_name=speaker_name,
                    text=response_text,
                    built_agents=built_agents,
                    levels=levels,
                    speaker_prefab=speaker_prefab,
                )

                with exporter.start_span(
                    "agent.communication",
                    {
                        "agent_name": speaker_name,
                        "turn": turn,
                        "recipients": recipients,
                        "message_length": len(response_text),
                        "hierarchy_level": hierarchy_level,
                    },
                ):
                    pass

                turn_records.append(
                    {
                        "agent_name": speaker_name,
                        "turn": turn,
                        "stance": stance,
                        "text": response_text,
                        "expected_stance": seed.ground_truth_reaction.direction,
                        "hierarchy_level": hierarchy_level,
                    }
                )
                last_consensus_text = response_text

            with exporter.start_span(
                "gm.state_transition",
                {
                    "phase": "turn_complete",
                    "turn": turn,
                    "num_speakers": len(speakers),
                },
            ):
                pass

        # --- Determine final consensus ---
        if is_hierarchical and levels is not None:
            # In hierarchical mode, the orchestrator's last statement is
            # the official consensus. Ask them for a final statement.
            orchestrator_prefab = levels[1][0]
            orchestrator_name = _prefab_name(orchestrator_prefab)
            orchestrator_agent = built_agents[orchestrator_name]

            final_spec = entity_lib.ActionSpec(
                call_to_action=(
                    f"{orchestrator_name}, as CSO you must now deliver the "
                    "team's final consensus market prediction. Synthesize all "
                    "reports and state your definitive view."
                ),
                output_type=entity_lib.OutputType.FREE,
                tag="consensus",
            )
            consensus_text = orchestrator_agent.act(final_spec)
            consensus_stance = extract_stance(
                consensus_text, fallback=seed.ground_truth_reaction.direction
            )

            with exporter.start_span(
                "gm.state_transition",
                {
                    "phase": "consensus",
                    "consensus_speaker": orchestrator_name,
                    "consensus_stance": consensus_stance,
                },
            ):
                pass

            turn_records.append(
                {
                    "agent_name": orchestrator_name,
                    "turn": max_turns + 1,
                    "stance": consensus_stance,
                    "text": consensus_text,
                    "expected_stance": seed.ground_truth_reaction.direction,
                    "hierarchy_level": 1,
                }
            )
            last_consensus_text = consensus_text
        else:
            # Flat: consensus via majority-direction synthesis.
            # Collect each agent's final-turn statement, vote on direction,
            # then concatenate majority-aligned statements to preserve
            # magnitude keywords and key factors for scoring.
            consensus_text, consensus_stance, vote_breakdown = (
                self._compute_flat_consensus(
                    turn_records, prefabs_flat, seed.ground_truth_reaction.direction,
                )
            )
            last_consensus_text = consensus_text

            with exporter.start_span(
                "gm.state_transition",
                {
                    "phase": "consensus",
                    "consensus_method": "majority_direction_synthesis",
                    "consensus_stance": consensus_stance,
                    "vote_breakdown": vote_breakdown,
                },
            ):
                pass

        # --- Evaluate ---
        accuracy = task.evaluate(consensus_text, seed.ground_truth_reaction)

        with exporter.start_span(
            "gm.state_transition",
            {
                "phase": "completed",
                "accuracy": accuracy,
                "topology": "hierarchical" if is_hierarchical else "flat",
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
            consensus_prediction=consensus_text,
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

    @staticmethod
    def _compute_flat_consensus(
        turn_records: list[dict[str, Any]],
        prefabs: list[Any],
        fallback_direction: str,
    ) -> tuple[str, str, dict[str, int]]:
        """Aggregate flat-topology agents into a majority-direction consensus.

        Inspired by MiroFish's multi-perspective synthesis: rather than
        picking one agent's statement, we collect every agent's final-turn
        output, determine the majority direction via stance vote, and
        concatenate the majority-aligned statements.  Concatenation
        preserves magnitude keywords and key-factor mentions so that
        ``PredictiveIntelTask.evaluate()`` can score against the full
        vocabulary of the group's reasoning.

        Returns:
            (consensus_text, consensus_direction, vote_breakdown)
        """
        # Collect each agent's LAST turn record
        agent_names = [_prefab_name(p) for p in prefabs]
        last_by_agent: dict[str, dict[str, Any]] = {}
        for record in turn_records:
            name = record.get("agent_name", "")
            if name in agent_names:
                last_by_agent[name] = record

        if not last_by_agent:
            return "", fallback_direction, {}

        # Vote on direction
        direction_votes: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for record in last_by_agent.values():
            stance = record.get("stance", "")
            if stance in direction_votes:
                direction_votes[stance] += 1

        majority_direction = max(direction_votes, key=direction_votes.get)  # type: ignore[arg-type]
        if direction_votes[majority_direction] == 0:
            majority_direction = fallback_direction

        # Concatenate statements aligned with the majority direction.
        # This preserves the agents' natural-language magnitude and
        # factor mentions for downstream keyword-based scoring.
        majority_texts: list[str] = []
        for name in agent_names:
            record = last_by_agent.get(name)
            if record is None:
                continue
            if record.get("stance") == majority_direction:
                majority_texts.append(record.get("text", ""))

        # If no agent matched majority (shouldn't happen), fall back to all
        if not majority_texts:
            majority_texts = [
                r.get("text", "") for r in last_by_agent.values() if r.get("text")
            ]

        consensus_text = "\n\n".join(majority_texts)
        return consensus_text, majority_direction, direction_votes

    @staticmethod
    def _record_stance_tracker(
        agent: entity_agent_with_logging.EntityAgentWithLogging,
        turn: int,
        stance: str,
    ) -> None:
        """Record stance in the agent's StanceTracker component if present."""
        components = getattr(agent, "_context_components", None)
        if isinstance(components, dict):
            tracker = components.get(STANCE_TRACKER_KEY)
            if isinstance(tracker, StanceTracker):
                tracker.record(turn, stance)
