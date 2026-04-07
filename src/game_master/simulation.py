"""Simulation runner: orchestrates turns, routes observations, records traces.

The Game Master (GM) is the objective simulation engine. It:
  - Instantiates agents for each trial via the prefabs.
  - Routes observations between agents according to the topology.
  - Records structured outputs to JSONL via the exporter.
  - Does NOT participate in the debate — it is not an agent.

Turn execution (per CLAUDE.md):
  Flat condition:
    All agents act simultaneously each turn. Each agent sees all other agents'
    outputs from the previous turn (global shared forum).

  Hierarchical condition:
    Turn 1 (top-down):  L1 → L2 → L3 (pressure establishment)
    Turns 2–N (bottom-up): L3 → L2 → L1, then L1+L2 outputs flow down
                            as starting context for next turn.

This module contains the core SimulationRunner class. The experiment runners
in experiments/ call run_flat_trial() and run_hierarchical_trial().
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from concordia.agents import entity_agent
from concordia.language_model import language_model

from src.agents import prefab_common
from src.agents.analyst_prefab import AnalystPrefab
from src.agents.components import StanceTracker
from src.agents.orchestrator_prefab import OrchestratorPrefab
from src.agents.output_parser import parse_agent_output
from src.config import Condition, ExperimentConfig
from src.hallucination.injector import HallucinationInjector
from src.tasks.predictive_intel import PredictiveIntelligenceTask, TaskContext
from src.tracing.otel_exporter import AgentTurnRecord, JSONLExporter
from src.tracing.raw_trace_exporter import RawTraceExporter, RawTraceRecord

logger = logging.getLogger(__name__)

# Mapping from rank string to integer level for JSONL records.
_RANK_TO_LEVEL = {
    "L1_ORCHESTRATOR": 1,
    "L2_MANAGER": 2,
    "L3_ANALYST": 3,
    "PEER": 0,
}


def _get_tracker(agent: entity_agent.EntityAgent) -> StanceTracker:
    return agent.get_component("stance_tracker", type_=StanceTracker)


def _act_and_record(
    agent: entity_agent.EntityAgent,
    turn: int,
    trial_id: str,
    context: TaskContext,
    condition: str,
    level: int,
    exporter: JSONLExporter,
    raw_exporter: RawTraceExporter,
) -> str:
    """Call agent.act(), parse output, record to JSONL, return raw output."""
    raw = agent.act(prefab_common.ACTION_SPEC)
    parsed = parse_agent_output(raw)
    tracker = _get_tracker(agent)
    previous_direction = tracker.get_current_direction() or "NEUTRAL"

    if parsed is not None:
        record = AgentTurnRecord(
            trial_id=trial_id,
            seed_doc=context.seed_doc_id,
            condition=condition,
            turn=turn,
            agent_id=agent.name,
            level=level,
            prediction_direction=parsed["prediction_direction"],
            confidence=parsed["confidence"],
            prediction_summary=parsed.get("prediction_summary", ""),
            key_factors=parsed.get("key_factors", []),
            parse_success=True,
        )
    else:
        record = AgentTurnRecord.from_parse_failure(
            trial_id=trial_id,
            seed_doc=context.seed_doc_id,
            condition=condition,
            turn=turn,
            agent_id=agent.name,
            level=level,
            previous_direction=previous_direction,
        )

    exporter.record(record)
    raw_exporter.record(
        RawTraceRecord(
            trial_id=trial_id,
            condition=condition,
            turn=turn,
            event_type="agent_output",
            sender=agent.name,
            receiver="game_master",
            content=raw,
            level=level,
        )
    )
    return raw


def _observe_and_record(
    *,
    receiver: entity_agent.EntityAgent,
    content: str,
    trial_id: str,
    condition: str,
    turn: int,
    raw_exporter: RawTraceExporter,
    sender: str,
) -> None:
    receiver.observe(content)
    raw_exporter.record(
        RawTraceRecord(
            trial_id=trial_id,
            condition=condition,
            turn=turn,
            event_type="observation",
            sender=sender,
            receiver=receiver.name,
            content=content,
        )
    )


class SimulationRunner:
    """Runs flat and hierarchical trials for the MAS sycophancy experiment."""

    # Hierarchical topology: 1 orchestrator + 4 managers + 16 analysts = 21 total.
    N_MANAGERS = 4
    N_ANALYSTS_PER_MANAGER = 4  # 4 * 4 = 16 L3 analysts.

    def __init__(
        self,
        model: language_model.LanguageModel,
        config: ExperimentConfig,
    ) -> None:
        self._model = model
        self._config = config

    # -----------------------------------------------------------------------
    # Flat topology
    # -----------------------------------------------------------------------

    def run_flat_trial(
        self,
        task: PredictiveIntelligenceTask,
        trial_id: int,
        inject_hallucination: bool = False,
        injection_agent_seed: int | None = None,
        rerun_id: int | None = None,
    ) -> Path:
        """Run one flat-topology trial and write trace to JSONL.

        Args:
            task: Loaded seed document task.
            trial_id: Trial index (0-based).
            inject_hallucination: If True, one randomly selected peer agent
                receives the hallucination injection.
            injection_agent_seed: Random seed for selecting the injector peer.
                If None, uses config.random_seed + trial_id.

        Returns:
            Path to the written JSONL file.
        """
        context = task.get_context()
        condition = (
            Condition.FLAT_HALLUCINATION.value
            if inject_hallucination
            else Condition.FLAT_BASELINE.value
        )
        tid = f"{condition}_{self._config.seed_doc.value}_trial_{trial_id:03d}"
        if rerun_id is not None:
            tid = f"{tid}_rerun_{rerun_id}"

        cfg = self._config
        out_path = cfg.jsonl_path(trial_id, rerun_id=rerun_id)

        # Determine which peer gets the hallucination (if any).
        injector_idx: int | None = None
        if inject_hallucination:
            rng = random.Random(injection_agent_seed or (cfg.random_seed + trial_id))
            injector_idx = rng.randint(0, 20)  # 21 agents, 0-indexed.

        # Build 21 peer agents.
        persona_base = (
            Path(__file__).parent.parent / "agents" / "prompts" / "financial_analyst.md"
        ).read_text()
        injector = HallucinationInjector(version=cfg.hallucination_prompt_version)
        hallucination_text = injector.render(
            ground_truth_direction=context.ground_truth,
            domain=context.domain,
        )

        agents: list[entity_agent.EntityAgent] = []
        for i in range(21):
            persona = persona_base
            if inject_hallucination and i == injector_idx:
                persona = f"{persona_base}\n\n{hallucination_text}"
            agent = prefab_common.make_agent(
                name=f"peer_{i:02d}",
                model=self._model,
                persona=persona,
                rank="PEER",
            )
            agents.append(agent)

        raw_trace_path = cfg.raw_trace_path(trial_id, rerun_id=rerun_id)

        with (
            JSONLExporter(out_path) as exporter,
            RawTraceExporter(raw_trace_path) as raw_exporter,
        ):
            prev_turn_outputs: dict[str, str] = {}

            # Seed document observation — all agents receive it on Turn 1.
            for agent in agents:
                _observe_and_record(
                    receiver=agent,
                    content=context.formatted_prompt,
                    trial_id=tid,
                    condition=condition,
                    turn=1,
                    raw_exporter=raw_exporter,
                    sender="game_master_seed_doc",
                )

            for turn in range(1, cfg.n_turns + 1):
                current_turn_outputs: dict[str, str] = {}

                for agent in agents:
                    # Inject previous-turn peer outputs (global shared forum).
                    for other_name, other_output in prev_turn_outputs.items():
                        if other_name != agent.name:
                            routed = f"[{other_name}]: {other_output}"
                            _observe_and_record(
                                receiver=agent,
                                content=routed,
                                trial_id=tid,
                                condition=condition,
                                turn=turn,
                                raw_exporter=raw_exporter,
                                sender=other_name,
                            )

                    raw = _act_and_record(
                        agent,
                        turn,
                        tid,
                        context,
                        condition,
                        level=0,
                        exporter=exporter,
                        raw_exporter=raw_exporter,
                    )
                    current_turn_outputs[agent.name] = raw

                prev_turn_outputs = current_turn_outputs

        logger.info("Flat trial %s complete → %s", tid, out_path)
        return out_path

    # -----------------------------------------------------------------------
    # Hierarchical topology
    # -----------------------------------------------------------------------

    def run_hierarchical_trial(
        self,
        task: PredictiveIntelligenceTask,
        trial_id: int,
    ) -> Path:
        """Run one hierarchical trial with hallucination injection at L1.

        Turn 1 (top-down): Orchestrator → Managers → Analysts.
        Turns 2–N (bottom-up): Analysts → Managers → Orchestrator,
                                then Orchestrator/Manager outputs flow back
                                down as starting context for the next turn.

        Args:
            task: Loaded seed document task.
            trial_id: Trial index (0-based).

        Returns:
            Path to the written JSONL file.
        """
        context = task.get_context()
        condition = Condition.HIERARCHICAL_HALLUCINATION.value
        tid = f"{condition}_{self._config.seed_doc.value}_trial_{trial_id:03d}"
        cfg = self._config
        out_path = cfg.jsonl_path(trial_id)

        injector = HallucinationInjector(version=cfg.hallucination_prompt_version)
        hallucination_text = injector.render(
            ground_truth_direction=context.ground_truth,
            domain=context.domain,
        )

        # Build orchestrator (L1) with hallucination.
        orchestrator_prefab = OrchestratorPrefab(
            params={
                "name": "orchestrator",
                "hallucination_injection": hallucination_text,
            }
        )
        orchestrator = orchestrator_prefab.build(self._model, memory_bank=None)  # type: ignore[arg-type]

        # Build 4 managers (L2).
        managers: list[entity_agent.EntityAgent] = []
        for m in range(self.N_MANAGERS):
            p = AnalystPrefab(params={"name": f"manager_{m:02d}", "rank": "L2_MANAGER"})
            managers.append(p.build(self._model, memory_bank=None))  # type: ignore[arg-type]

        # Build 16 analysts (L3), 4 per manager.
        analysts_by_manager: list[list[entity_agent.EntityAgent]] = []
        for m in range(self.N_MANAGERS):
            group: list[entity_agent.EntityAgent] = []
            for a in range(self.N_ANALYSTS_PER_MANAGER):
                idx = m * self.N_ANALYSTS_PER_MANAGER + a
                p = AnalystPrefab(
                    params={"name": f"analyst_{idx:02d}", "rank": "L3_ANALYST"}
                )
                group.append(p.build(self._model, memory_bank=None))  # type: ignore[arg-type]
            analysts_by_manager.append(group)

        all_analysts = [a for group in analysts_by_manager for a in group]

        raw_trace_path = cfg.raw_trace_path(trial_id)

        with (
            JSONLExporter(out_path) as exporter,
            RawTraceExporter(raw_trace_path) as raw_exporter,
        ):
            # Everyone receives the seed document on Turn 1.
            for agent in [orchestrator] + managers + all_analysts:
                _observe_and_record(
                    receiver=agent,
                    content=context.formatted_prompt,
                    trial_id=tid,
                    condition=condition,
                    turn=1,
                    raw_exporter=raw_exporter,
                    sender="game_master_seed_doc",
                )

            # ---------- Turn 1: top-down ----------
            orchestrator_output = _act_and_record(
                orchestrator,
                1,
                tid,
                context,
                condition,
                level=1,
                exporter=exporter,
                raw_exporter=raw_exporter,
            )

            manager_outputs: list[str] = []
            for m_idx, manager in enumerate(managers):
                routed = f"[orchestrator directive]: {orchestrator_output}"
                _observe_and_record(
                    receiver=manager,
                    content=routed,
                    trial_id=tid,
                    condition=condition,
                    turn=1,
                    raw_exporter=raw_exporter,
                    sender="orchestrator",
                )
                m_out = _act_and_record(
                    manager,
                    1,
                    tid,
                    context,
                    condition,
                    level=2,
                    exporter=exporter,
                    raw_exporter=raw_exporter,
                )
                manager_outputs.append(m_out)

            for m_idx, (manager, group) in enumerate(
                zip(managers, analysts_by_manager)
            ):
                for analyst in group:
                    routed = (
                        f"[manager_{m_idx:02d} synthesis]: {manager_outputs[m_idx]}"
                    )
                    _observe_and_record(
                        receiver=analyst,
                        content=routed,
                        trial_id=tid,
                        condition=condition,
                        turn=1,
                        raw_exporter=raw_exporter,
                        sender=f"manager_{m_idx:02d}",
                    )
                    _act_and_record(
                        analyst,
                        1,
                        tid,
                        context,
                        condition,
                        level=3,
                        exporter=exporter,
                        raw_exporter=raw_exporter,
                    )

            # ---------- Turns 2–N: bottom-up ----------
            for turn in range(2, cfg.n_turns + 1):
                # L3 Analysts act first.
                new_manager_inputs: list[list[str]] = [
                    [] for _ in range(self.N_MANAGERS)
                ]
                for m_idx, group in enumerate(analysts_by_manager):
                    for analyst in group:
                        raw = _act_and_record(
                            analyst,
                            turn,
                            tid,
                            context,
                            condition,
                            level=3,
                            exporter=exporter,
                            raw_exporter=raw_exporter,
                        )
                        new_manager_inputs[m_idx].append(f"[{analyst.name}]: {raw}")

                # L2 Managers synthesise analyst reports.
                new_orchestrator_inputs: list[str] = []
                for m_idx, manager in enumerate(managers):
                    for analyst_report in new_manager_inputs[m_idx]:
                        _observe_and_record(
                            receiver=manager,
                            content=analyst_report,
                            trial_id=tid,
                            condition=condition,
                            turn=turn,
                            raw_exporter=raw_exporter,
                            sender="analyst_group",
                        )
                    m_out = _act_and_record(
                        manager,
                        turn,
                        tid,
                        context,
                        condition,
                        level=2,
                        exporter=exporter,
                        raw_exporter=raw_exporter,
                    )
                    new_orchestrator_inputs.append(f"[manager_{m_idx:02d}]: {m_out}")
                    manager_outputs[m_idx] = m_out

                # L1 Orchestrator synthesises manager reports.
                for manager_report in new_orchestrator_inputs:
                    _observe_and_record(
                        receiver=orchestrator,
                        content=manager_report,
                        trial_id=tid,
                        condition=condition,
                        turn=turn,
                        raw_exporter=raw_exporter,
                        sender="manager_group",
                    )
                orchestrator_output = _act_and_record(
                    orchestrator,
                    turn,
                    tid,
                    context,
                    condition,
                    level=1,
                    exporter=exporter,
                    raw_exporter=raw_exporter,
                )

                # Propagate L1+L2 outputs downward (visible next turn).
                for m_idx, (manager, group) in enumerate(
                    zip(managers, analysts_by_manager)
                ):
                    for analyst in group:
                        routed_orch = f"[orchestrator update]: {orchestrator_output}"
                        _observe_and_record(
                            receiver=analyst,
                            content=routed_orch,
                            trial_id=tid,
                            condition=condition,
                            turn=turn,
                            raw_exporter=raw_exporter,
                            sender="orchestrator",
                        )
                        routed_mgr = (
                            f"[manager_{m_idx:02d} update]: {manager_outputs[m_idx]}"
                        )
                        _observe_and_record(
                            receiver=analyst,
                            content=routed_mgr,
                            trial_id=tid,
                            condition=condition,
                            turn=turn,
                            raw_exporter=raw_exporter,
                            sender=f"manager_{m_idx:02d}",
                        )

        logger.info("Hierarchical trial %s complete → %s", tid, out_path)
        return out_path
