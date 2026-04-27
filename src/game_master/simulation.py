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
from collections import Counter
from pathlib import Path
from typing import Any

from concordia.agents import entity_agent
from concordia.language_model import language_model

from src.agents import prefab_common
from src.agents.analyst_prefab import AnalystPrefab
from src.agents.components import StanceTracker
from src.agents.orchestrator_prefab import OrchestratorPrefab
from src.agents.output_parser import parse_agent_output
from src.config import Condition, ExperimentConfig
from src.evidence.loader import (
    EvidenceAllocation,
    allocate_evidence,
    format_drip_packet,
    format_evidence_packet,
    load_evidence_files,
)
from src.hallucination.injector import HallucinationInjector
from src.persistence import (
    AgentMessageRecord,
    AgentRetrievalRecord,
    ExperimentRunRecord,
    PostgresPersistence,
)
from src.tasks.predictive_intel import PredictiveIntelligenceTask, TaskContext
from src.topologies.flat import N_AGENTS as FLAT_N_AGENTS
from src.topologies.hierarchical import (
    N_L2_MANAGERS,
    N_L3_ANALYSTS_PER_MANAGER,
)
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


def _extract_summary_view(raw: str, agent_name: str) -> str:
    """Extract a summary-only view of an agent's output for peer routing.

    Instead of showing peers the full structured JSON (which creates
    overwhelming conformity pressure on numerical fields), this extracts
    just the reasoning summary and key factors. Peers learn *what* a
    colleague thinks and *why*, but not the exact numerical prediction.
    """
    parsed = parse_agent_output(raw)
    if parsed is None:
        return f"[{agent_name}]: {raw[:500]}"
    summary = parsed.get("prediction_summary", "")
    factors = parsed.get("key_factors", [])
    parts = [f"[{agent_name} analysis]: {summary}"]
    if factors:
        parts.append("Key factors cited: " + "; ".join(str(f) for f in factors))
    return "\n".join(parts)


def _role_from_level(level: int | None) -> str:
    if level is None:
        return "UNKNOWN"
    return {
        0: "PEER",
        1: "L1_ORCHESTRATOR",
        2: "L2_MANAGER",
        3: "L3_ANALYST",
    }.get(level, "UNKNOWN")


def _seed_metadata(
    task: PredictiveIntelligenceTask,
    context: TaskContext,
) -> dict[str, Any]:
    data = task.data
    metadata = data.get("metadata", {})
    return {
        "seed_id": context.seed_doc_id,
        "file_name": context.seed_file_name,
        "domain": context.domain,
        "target_entity": metadata.get("target_entity", "unknown"),
        "ground_truth_direction": context.ground_truth,
        "full_json": data,
    }


def _local_evidence_query(
    task: PredictiveIntelligenceTask, context: TaskContext
) -> str:
    metadata = task.data.get("metadata", {})
    target_entity = metadata.get("target_entity", "")
    return " ".join(
        part
        for part in (str(target_entity), context.domain, context.seed_doc_id)
        if part
    )


def _format_evidence_packet(evidence_rows: list[dict[str, Any]]) -> str:
    if not evidence_rows:
        return "LOCAL EVIDENCE: No matching local evidence documents were found."

    lines = ["LOCAL EVIDENCE RETRIEVED FROM POSTGRES:"]
    for idx, row in enumerate(evidence_rows, start=1):
        text = str(row.get("text_content", "")).strip()
        if len(text) > 800:
            text = f"{text[:797]}..."
        lines.extend(
            [
                "",
                f"[{idx}] {row.get('id', 'unknown')}",
                f"Source: {row.get('source_type', 'unknown')} / {row.get('source_name', 'unknown')}",
                f"Title: {row.get('title', 'Untitled')}",
                f"Date: {row.get('document_date') or 'unknown'}",
                f"Text: {text}",
            ]
        )
    return "\n".join(lines)


def _retrieve_local_evidence(
    *,
    persistence: PostgresPersistence | None,
    config: ExperimentConfig,
    task: PredictiveIntelligenceTask,
    context: TaskContext,
    run_id: str,
) -> str | None:
    if not config.enable_local_evidence:
        return None
    if persistence is None:
        raise RuntimeError("Local evidence retrieval requires DB persistence.")

    query = _local_evidence_query(task, context)
    evidence_rows = persistence.search_evidence(
        query=query,
        seed_id=context.seed_doc_id,
        limit=config.local_evidence_limit,
    )
    persistence.log_agent_retrieval(
        AgentRetrievalRecord(
            run_id=run_id,
            agent_name="game_master",
            round_number=1,
            query=query,
            result_ids=[str(row["id"]) for row in evidence_rows],
        )
    )
    return _format_evidence_packet(evidence_rows)


def _prepare_persistence(
    *,
    config: ExperimentConfig,
    task: PredictiveIntelligenceTask,
    context: TaskContext,
    run_id: str,
    topology: str,
    condition: str,
    trial_id: int,
    rerun_id: int | None = None,
) -> PostgresPersistence | None:
    if not config.enable_db_persistence:
        return None

    persistence = PostgresPersistence(config.database_url)
    persistence.init_schema()
    persistence.upsert_seed_document(**_seed_metadata(task, context))
    persistence.create_run(
        ExperimentRunRecord(
            run_id=run_id,
            seed_id=context.seed_doc_id,
            topology=topology,
            condition=condition,
            trial_id=trial_id,
            rerun_id=rerun_id,
        )
    )
    return persistence


def _log_persistence_message(
    persistence: PostgresPersistence | None,
    *,
    run_id: str,
    agent_name: str,
    agent_role: str,
    round_number: int,
    message_type: str,
    content_json: dict[str, Any],
) -> None:
    if persistence is None:
        return
    persistence.log_agent_message(
        AgentMessageRecord(
            run_id=run_id,
            agent_name=agent_name,
            agent_role=agent_role,
            round_number=round_number,
            message_type=message_type,
            content_json=content_json,
        )
    )


def _parsed_output_content(
    *,
    raw: str,
    parsed: dict | None,
    record: AgentTurnRecord,
) -> dict[str, Any]:
    return {
        "raw": raw,
        "parse_success": record.parse_success,
        "prediction_direction": record.prediction_direction,
        "predicted_magnitude": record.predicted_magnitude,
        "predicted_price_change_pct": record.predicted_price_change_pct,
        "prediction_summary": record.prediction_summary,
        "key_factors": record.key_factors,
        "trail_category": record.trail_category,
        "parsed": parsed,
    }


def _final_decision_from_outputs(outputs: dict[str, str]) -> tuple[str, float]:
    parsed_outputs = [
        parsed
        for raw in outputs.values()
        if (parsed := parse_agent_output(raw)) is not None
    ]
    if not parsed_outputs:
        return "NEUTRAL", 0.0

    counts = Counter(parsed["prediction_direction"] for parsed in parsed_outputs)
    final_direction = max(counts, key=lambda d: (counts[d], d))
    # confidence field removed from agent output schema; return 0.0 for DB compat
    return final_direction, 0.0


def _final_decision_from_raw(raw: str) -> tuple[str, float]:
    parsed = parse_agent_output(raw)
    if parsed is None:
        return "NEUTRAL", 0.0
    return parsed["prediction_direction"], 0.0


def _act_and_record(
    agent: entity_agent.EntityAgent,
    turn: int,
    trial_id: str,
    context: TaskContext,
    condition: str,
    level: int,
    exporter: JSONLExporter,
    raw_exporter: RawTraceExporter,
    persistence: PostgresPersistence | None = None,
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
            predicted_magnitude=parsed["predicted_magnitude"],
            predicted_price_change_pct=parsed["predicted_price_change_pct"],
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
    _log_persistence_message(
        persistence,
        run_id=trial_id,
        agent_name=agent.name,
        agent_role=_role_from_level(level),
        round_number=turn,
        message_type="agent_output",
        content_json=_parsed_output_content(raw=raw, parsed=parsed, record=record),
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
    receiver_role: str = "UNKNOWN",
    persistence: PostgresPersistence | None = None,
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
    _log_persistence_message(
        persistence,
        run_id=trial_id,
        agent_name=receiver.name,
        agent_role=receiver_role,
        round_number=turn,
        message_type="observation",
        content_json={
            "sender": sender,
            "receiver": receiver.name,
            "content": content,
        },
    )


class SimulationRunner:
    """Runs flat and hierarchical trials for the MAS sycophancy experiment."""

    # Hierarchical topology constants from src/topologies/hierarchical.py
    N_MANAGERS = N_L2_MANAGERS  # 4
    N_ANALYSTS_PER_MANAGER = N_L3_ANALYSTS_PER_MANAGER  # 4 * 4 = 16 L3 analysts.

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
        resume: bool = False,
    ) -> Path:
        """Run one flat-topology trial and write trace to JSONL.

        Args:
            task: Loaded seed document task.
            trial_id: Trial index (0-based).
            inject_hallucination: If True, one randomly selected peer agent
                receives the hallucination injection.
            injection_agent_seed: Random seed for selecting the injector peer.
                If None, uses config.random_seed + trial_id.
            rerun_id: For K-reruns in flat hallucination condition.
            resume: If True, detect existing trace file, determine last
                completed turn, and resume from the next turn. Agents are
                rebuilt and prior observations are replayed into their memory.

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
        persistence = _prepare_persistence(
            config=cfg,
            task=task,
            context=context,
            run_id=tid,
            topology="flat",
            condition=condition,
            trial_id=trial_id,
            rerun_id=rerun_id,
        )
        local_evidence = _retrieve_local_evidence(
            persistence=persistence,
            config=cfg,
            task=task,
            context=context,
            run_id=tid,
        )

        # File-based per-agent evidence (no Postgres required).
        evidence_alloc: EvidenceAllocation | None = None
        if cfg.enable_file_evidence:
            all_docs = load_evidence_files(seed_id=context.seed_doc_id)
            agent_names = [f"peer_{i:02d}" for i in range(FLAT_N_AGENTS)]
            evidence_alloc = allocate_evidence(
                docs=all_docs,
                agent_names=agent_names,
                docs_per_agent=cfg.evidence_docs_per_agent,
                drip_turns=cfg.evidence_drip_turns,
                docs_per_drip=cfg.evidence_drip_docs_per_turn,
                rng_seed=cfg.random_seed + trial_id,
            )

        # Determine which peer gets the hallucination (if any).
        injector_idx: int | None = None
        if inject_hallucination:
            rng = random.Random(injection_agent_seed or (cfg.random_seed + trial_id))
            injector_idx = rng.randint(0, FLAT_N_AGENTS - 1)

        # Build 21 peer agents via AnalystPrefab (loads base + flat_peer overlay).
        injector = HallucinationInjector(
            version=cfg.hallucination_prompt_version, variant="flat_peer"
        )
        hallucination_text = injector.render(
            ground_truth_direction=context.ground_truth,
            domain=context.domain,
        )

        agents: list[entity_agent.EntityAgent] = []
        for i in range(FLAT_N_AGENTS):
            params: dict[str, str] = {"name": f"peer_{i:02d}", "rank": "PEER"}
            if inject_hallucination and i == injector_idx:
                params["hallucination_text"] = hallucination_text
            p = AnalystPrefab(params=params)
            agent = p.build(self._model, memory_bank=None)  # type: ignore[arg-type]
            agents.append(agent)

        raw_trace_path = cfg.raw_trace_path(trial_id, rerun_id=rerun_id)

        # ---- Resume detection ----
        start_turn = 1
        resumed_outputs_by_turn: dict[int, dict[str, str]] = {}
        if resume and out_path.exists():
            # Peek at existing records to find last completed turn.
            probe = JSONLExporter(out_path, resume=True)
            last_done = probe.last_completed_turn()
            probe.close()
            if last_done >= cfg.n_turns:
                logger.info(
                    "Trial %s already complete (%d turns). Skipping.", tid, last_done
                )
                return out_path
            if last_done > 0:
                logger.info(
                    "Resuming trial %s from turn %d (turns 1–%d complete).",
                    tid,
                    last_done + 1,
                    last_done,
                )
                start_turn = last_done + 1
                # Reconstruct per-turn outputs from existing records so we
                # can replay peer observations into freshly built agents.
                for rec in probe.existing_records:
                    t = rec["turn"]
                    agent_id = rec["agent_id"]
                    # Rebuild a JSON string that _extract_summary_view can parse.
                    reconstructed = json.dumps(
                        {
                            "prediction_direction": rec["prediction_direction"],
                            "predicted_magnitude": rec["predicted_magnitude"],
                            "predicted_price_change_pct": rec[
                                "predicted_price_change_pct"
                            ],
                            "prediction_summary": rec.get("prediction_summary", ""),
                            "key_factors": rec.get("key_factors", []),
                        }
                    )
                    resumed_outputs_by_turn.setdefault(t, {})[agent_id] = reconstructed

        try:
            with (
                JSONLExporter(out_path, resume=(start_turn > 1)) as exporter,
                RawTraceExporter(
                    raw_trace_path, resume=(start_turn > 1)
                ) as raw_exporter,
            ):
                prev_turn_outputs: dict[str, str] = {}

                # Seed document observation — all agents receive it on Turn 1.
                # When resuming, agents are freshly constructed and still need
                # the seed doc in their memory. We observe() directly to avoid
                # re-recording to exporters when resuming.
                for agent in agents:
                    if start_turn > 1:
                        agent.observe(context.formatted_prompt)
                    else:
                        _observe_and_record(
                            receiver=agent,
                            content=context.formatted_prompt,
                            trial_id=tid,
                            condition=condition,
                            turn=1,
                            raw_exporter=raw_exporter,
                            sender="game_master_seed_doc",
                            receiver_role="PEER",
                            persistence=persistence,
                        )
                    if local_evidence is not None:
                        if start_turn > 1:
                            agent.observe(local_evidence)
                        else:
                            _observe_and_record(
                                receiver=agent,
                                content=local_evidence,
                                trial_id=tid,
                                condition=condition,
                                turn=1,
                                raw_exporter=raw_exporter,
                                sender="local_evidence_store",
                                receiver_role="PEER",
                                persistence=persistence,
                            )
                    # Per-agent file evidence (unique subset per agent).
                    if evidence_alloc is not None:
                        agent_docs = evidence_alloc.agent_evidence.get(agent.name, [])
                        if agent_docs:
                            packet = format_evidence_packet(agent_docs)
                            if start_turn > 1:
                                agent.observe(packet)
                            else:
                                _observe_and_record(
                                    receiver=agent,
                                    content=packet,
                                    trial_id=tid,
                                    condition=condition,
                                    turn=1,
                                    raw_exporter=raw_exporter,
                                    sender="evidence_files",
                                    receiver_role="PEER",
                                    persistence=persistence,
                                )

                # When resuming, replay completed turns into agent memory
                # so they have the full conversation history.
                if start_turn > 1:
                    for replay_turn in range(1, start_turn):
                        turn_outputs = resumed_outputs_by_turn.get(replay_turn, {})
                        # Replay drip evidence if applicable.
                        if (
                            evidence_alloc is not None
                            and replay_turn in evidence_alloc.drip_schedule
                        ):
                            drip_docs = evidence_alloc.drip_schedule[replay_turn]
                            drip_text = format_drip_packet(drip_docs, replay_turn)
                            if drip_text:
                                for agent in agents:
                                    agent.observe(drip_text)
                        # Replay peer observations for this turn.
                        prev_replay = resumed_outputs_by_turn.get(replay_turn - 1, {})
                        for agent in agents:
                            for other_name, other_output in prev_replay.items():
                                if other_name != agent.name:
                                    routed = _extract_summary_view(
                                        other_output, other_name
                                    )
                                    agent.observe(routed)
                        # Replay each agent's own output as self-observation.
                        for agent in agents:
                            own_output = turn_outputs.get(agent.name)
                            if own_output:
                                agent.observe(
                                    f"[Your previous Turn {replay_turn} output]: "
                                    + _extract_summary_view(own_output, agent.name)
                                )
                    # Set prev_turn_outputs to last completed turn for the
                    # live loop below.
                    prev_turn_outputs = resumed_outputs_by_turn.get(start_turn - 1, {})

                for turn in range(start_turn, cfg.n_turns + 1):
                    current_turn_outputs: dict[str, str] = {}

                    # Information drip: inject new evidence on scheduled turns.
                    if (
                        evidence_alloc is not None
                        and turn in evidence_alloc.drip_schedule
                    ):
                        drip_docs = evidence_alloc.drip_schedule[turn]
                        drip_text = format_drip_packet(drip_docs, turn)
                        if drip_text:
                            for agent in agents:
                                _observe_and_record(
                                    receiver=agent,
                                    content=drip_text,
                                    trial_id=tid,
                                    condition=condition,
                                    turn=turn,
                                    raw_exporter=raw_exporter,
                                    sender="evidence_drip",
                                    receiver_role="PEER",
                                    persistence=persistence,
                                )

                    for agent in agents:
                        # Inject previous-turn peer outputs (summary only).
                        for other_name, other_output in prev_turn_outputs.items():
                            if other_name != agent.name:
                                routed = _extract_summary_view(other_output, other_name)
                                _observe_and_record(
                                    receiver=agent,
                                    content=routed,
                                    trial_id=tid,
                                    condition=condition,
                                    turn=turn,
                                    raw_exporter=raw_exporter,
                                    sender=other_name,
                                    receiver_role="PEER",
                                    persistence=persistence,
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
                            persistence=persistence,
                        )
                        current_turn_outputs[agent.name] = raw

                    prev_turn_outputs = current_turn_outputs

            final_decision, final_confidence = _final_decision_from_outputs(
                prev_turn_outputs
            )
            if persistence is not None:
                persistence.finalize_run(
                    run_id=tid,
                    final_decision=final_decision,
                    final_confidence=final_confidence,
                    correct=final_decision == context.ground_truth,
                )
        finally:
            if persistence is not None:
                persistence.close()

        logger.info("Flat trial %s complete → %s", tid, out_path)
        return out_path

    # -----------------------------------------------------------------------
    # Hierarchical topology
    # -----------------------------------------------------------------------

    def run_hierarchical_trial(
        self,
        task: PredictiveIntelligenceTask,
        trial_id: int,
        inject_hallucination: bool = True,
        resume: bool = False,
    ) -> Path:
        """Run one hierarchical trial.

        Turn 1 (top-down): Orchestrator → Managers → Analysts.
        Turns 2–N (top-down): Orchestrator (with prev manager reports) →
                               Managers (with orch directive + prev analyst reports) →
                               Analysts (with fresh manager directive).

        Args:
            task: Loaded seed document task.
            trial_id: Trial index (0-based).
            inject_hallucination: If True, the orchestrator receives the
                hallucination injection in its persona. If False, runs a
                clean hierarchical baseline (no hallucination).
            resume: If True, detect existing trace file, determine last
                completed turn, and resume from the next turn. Agents are
                rebuilt and prior observations are replayed into their memory.

        Returns:
            Path to the written JSONL file.
        """
        context = task.get_context()
        condition = (
            Condition.HIERARCHICAL_HALLUCINATION.value
            if inject_hallucination
            else Condition.HIERARCHICAL_BASELINE.value
        )
        tid = f"{condition}_{self._config.seed_doc.value}_trial_{trial_id:03d}"
        cfg = self._config
        out_path = cfg.jsonl_path(trial_id)
        persistence = _prepare_persistence(
            config=cfg,
            task=task,
            context=context,
            run_id=tid,
            topology="hierarchical",
            condition=condition,
            trial_id=trial_id,
        )
        local_evidence = _retrieve_local_evidence(
            persistence=persistence,
            config=cfg,
            task=task,
            context=context,
            run_id=tid,
        )

        # File-based per-agent evidence.
        evidence_alloc: EvidenceAllocation | None = None
        if cfg.enable_file_evidence:
            all_docs = load_evidence_files(seed_id=context.seed_doc_id)
            # Build agent name list: orchestrator + managers + analysts
            agent_names = (
                ["orchestrator"]
                + [f"manager_{m:02d}" for m in range(self.N_MANAGERS)]
                + [
                    f"analyst_{m * self.N_ANALYSTS_PER_MANAGER + a:02d}"
                    for m in range(self.N_MANAGERS)
                    for a in range(self.N_ANALYSTS_PER_MANAGER)
                ]
            )
            evidence_alloc = allocate_evidence(
                docs=all_docs,
                agent_names=agent_names,
                docs_per_agent=cfg.evidence_docs_per_agent,
                drip_turns=cfg.evidence_drip_turns,
                docs_per_drip=cfg.evidence_drip_docs_per_turn,
                rng_seed=cfg.random_seed + trial_id,
            )

        hallucination_text: str | None = None
        if inject_hallucination:
            injector = HallucinationInjector(version=cfg.hallucination_prompt_version)
            hallucination_text = injector.render(
                ground_truth_direction=context.ground_truth,
                domain=context.domain,
            )

        # Build orchestrator (L1), with or without hallucination.
        orch_params: dict[str, str] = {"name": "orchestrator"}
        if hallucination_text is not None:
            orch_params["hallucination_injection"] = hallucination_text
        orchestrator_prefab = OrchestratorPrefab(params=orch_params)
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

        # ---- Resume detection ----
        start_turn = 1
        resumed_records_by_turn: dict[int, dict[str, str]] = {}
        if resume and out_path.exists():
            probe = JSONLExporter(out_path, resume=True)
            last_done = probe.last_completed_turn()
            probe.close()
            if last_done >= cfg.n_turns:
                logger.info(
                    "Hierarchical trial %s already complete (%d turns). Skipping.",
                    tid,
                    last_done,
                )
                return out_path
            if last_done > 0:
                logger.info(
                    "Resuming hierarchical trial %s from turn %d (turns 1–%d complete).",
                    tid,
                    last_done + 1,
                    last_done,
                )
                start_turn = last_done + 1
                for rec in probe.existing_records:
                    t = rec["turn"]
                    agent_id = rec["agent_id"]
                    reconstructed = json.dumps(
                        {
                            "prediction_direction": rec["prediction_direction"],
                            "predicted_magnitude": rec["predicted_magnitude"],
                            "predicted_price_change_pct": rec[
                                "predicted_price_change_pct"
                            ],
                            "prediction_summary": rec.get("prediction_summary", ""),
                            "key_factors": rec.get("key_factors", []),
                        }
                    )
                    resumed_records_by_turn.setdefault(t, {})[agent_id] = reconstructed

        try:
            with (
                JSONLExporter(out_path, resume=(start_turn > 1)) as exporter,
                RawTraceExporter(
                    raw_trace_path, resume=(start_turn > 1)
                ) as raw_exporter,
            ):
                # Everyone receives the seed document on Turn 1.
                # When resuming, observe() directly without recording.
                for agent in [orchestrator] + managers + all_analysts:
                    level = (
                        1
                        if agent.name == "orchestrator"
                        else 2
                        if agent.name.startswith("manager_")
                        else 3
                    )
                    if start_turn > 1:
                        agent.observe(context.formatted_prompt)
                    else:
                        _observe_and_record(
                            receiver=agent,
                            content=context.formatted_prompt,
                            trial_id=tid,
                            condition=condition,
                            turn=1,
                            raw_exporter=raw_exporter,
                            sender="game_master_seed_doc",
                            receiver_role=_role_from_level(level),
                            persistence=persistence,
                        )
                    if local_evidence is not None:
                        if start_turn > 1:
                            agent.observe(local_evidence)
                        else:
                            _observe_and_record(
                                receiver=agent,
                                content=local_evidence,
                                trial_id=tid,
                                condition=condition,
                                turn=1,
                                raw_exporter=raw_exporter,
                                sender="local_evidence_store",
                                receiver_role=_role_from_level(level),
                                persistence=persistence,
                            )
                    # Per-agent file evidence (unique subset per agent).
                    if evidence_alloc is not None:
                        agent_docs = evidence_alloc.agent_evidence.get(agent.name, [])
                        if agent_docs:
                            packet = format_evidence_packet(agent_docs)
                            if start_turn > 1:
                                agent.observe(packet)
                            else:
                                _observe_and_record(
                                    receiver=agent,
                                    content=packet,
                                    trial_id=tid,
                                    condition=condition,
                                    turn=1,
                                    raw_exporter=raw_exporter,
                                    sender="evidence_files",
                                    receiver_role=_role_from_level(level),
                                    persistence=persistence,
                                )

                # When resuming, replay completed turns into agent memory.
                if start_turn > 1:
                    all_agents_list = [orchestrator] + managers + all_analysts
                    agent_by_name = {a.name: a for a in all_agents_list}
                    for replay_turn in range(1, start_turn):
                        turn_recs = resumed_records_by_turn.get(replay_turn, {})
                        # Replay drip evidence if applicable.
                        if (
                            evidence_alloc is not None
                            and replay_turn in evidence_alloc.drip_schedule
                        ):
                            drip_docs = evidence_alloc.drip_schedule[replay_turn]
                            drip_text = format_drip_packet(drip_docs, replay_turn)
                            if drip_text:
                                for agent in all_agents_list:
                                    agent.observe(drip_text)

                        # Replay turn observations based on hierarchical routing.
                        # Orchestrator output → managers.
                        orch_out = turn_recs.get("orchestrator", "")
                        if orch_out:
                            routed = _extract_summary_view(orch_out, "orchestrator")
                            for manager in managers:
                                manager.observe(f"[orchestrator directive]: {routed}")
                        # Manager outputs → their analysts + orchestrator (for next turn).
                        for m_idx, manager in enumerate(managers):
                            m_name = f"manager_{m_idx:02d}"
                            m_out = turn_recs.get(m_name, "")
                            if m_out:
                                routed = _extract_summary_view(m_out, m_name)
                                for analyst in analysts_by_manager[m_idx]:
                                    analyst.observe(f"[{m_name} synthesis]: {routed}")
                        # Analyst outputs → their manager (for next turn routing).
                        for m_idx, group in enumerate(analysts_by_manager):
                            for analyst in group:
                                a_out = turn_recs.get(analyst.name, "")
                                if a_out:
                                    routed = _extract_summary_view(a_out, analyst.name)
                                    # Manager will receive this on next turn.
                        # Each agent's own output as self-observation.
                        for agent in all_agents_list:
                            own_out = turn_recs.get(agent.name)
                            if own_out:
                                agent.observe(
                                    f"[Your previous Turn {replay_turn} output]: "
                                    + _extract_summary_view(own_out, agent.name)
                                )

                    # Reconstruct manager_outputs and prev_analyst_outputs
                    # from last completed turn for the live loop.
                    last_turn_recs = resumed_records_by_turn.get(start_turn - 1, {})

                # Initialize state variables needed by both Turn 1 and Turns 2-N.
                orchestrator_output: str = ""
                manager_outputs: list[str] = [""] * self.N_MANAGERS
                prev_analyst_outputs: list[list[str]] = [
                    [] for _ in range(self.N_MANAGERS)
                ]

                if start_turn > 1:
                    # Reconstruct state from resumed records.
                    last_recs = resumed_records_by_turn.get(start_turn - 1, {})
                    orchestrator_output = last_recs.get("orchestrator", "")
                    for m_idx in range(self.N_MANAGERS):
                        m_name = f"manager_{m_idx:02d}"
                        manager_outputs[m_idx] = last_recs.get(m_name, "")
                        for a_idx in range(self.N_ANALYSTS_PER_MANAGER):
                            a_name = f"analyst_{m_idx * self.N_ANALYSTS_PER_MANAGER + a_idx:02d}"
                            a_out = last_recs.get(a_name, "")
                            if a_out:
                                prev_analyst_outputs[m_idx].append(
                                    _extract_summary_view(a_out, a_name)
                                )

                # ---------- Turn 1: top-down ----------
                if start_turn <= 1:
                    orchestrator_output = _act_and_record(
                        orchestrator,
                        1,
                        tid,
                        context,
                        condition,
                        level=1,
                        exporter=exporter,
                        raw_exporter=raw_exporter,
                        persistence=persistence,
                    )

                    manager_outputs = []
                    for m_idx, manager in enumerate(managers):
                        routed = _extract_summary_view(
                            orchestrator_output, "orchestrator"
                        )
                        _observe_and_record(
                            receiver=manager,
                            content=f"[orchestrator directive]: {routed}",
                            trial_id=tid,
                            condition=condition,
                            turn=1,
                            raw_exporter=raw_exporter,
                            sender="orchestrator",
                            receiver_role="L2_MANAGER",
                            persistence=persistence,
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
                            persistence=persistence,
                        )
                        manager_outputs.append(m_out)

                    for m_idx, (manager, group) in enumerate(
                        zip(managers, analysts_by_manager)
                    ):
                        for analyst in group:
                            routed = _extract_summary_view(
                                manager_outputs[m_idx], f"manager_{m_idx:02d}"
                            )
                            _observe_and_record(
                                receiver=analyst,
                                content=f"[manager_{m_idx:02d} synthesis]: {routed}",
                                trial_id=tid,
                                condition=condition,
                                turn=1,
                                raw_exporter=raw_exporter,
                                sender=f"manager_{m_idx:02d}",
                                receiver_role="L3_ANALYST",
                                persistence=persistence,
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
                                persistence=persistence,
                            )

                # ---------- Turns 2–N: top-down ----------
                # Each turn: Orchestrator (sees prev manager reports) →
                #            Managers (see orch directive + prev analyst reports) →
                #            Analysts (see fresh manager directive).
                # This ensures analysts never act on stale information.

                # Collect analyst outputs from turn 1 for use in turn 2
                # (only if Turn 1 was executed live, not resumed).
                # Note: Turn 1 doesn't capture raw analyst outputs into
                # prev_analyst_outputs — they start empty and get populated
                # from turn 2 onward.

                logger.info(
                    "Starting turns %d–%d loop (cfg.n_turns=%d)",
                    max(2, start_turn),
                    cfg.n_turns,
                    cfg.n_turns,
                )
                for turn in range(max(2, start_turn), cfg.n_turns + 1):
                    logger.info("=== Hierarchical Turn %d/%d ===", turn, cfg.n_turns)
                    # Information drip on scheduled turns.
                    if (
                        evidence_alloc is not None
                        and turn in evidence_alloc.drip_schedule
                    ):
                        drip_docs = evidence_alloc.drip_schedule[turn]
                        drip_text = format_drip_packet(drip_docs, turn)
                        if drip_text:
                            for agent in [orchestrator] + managers + all_analysts:
                                lvl = (
                                    1
                                    if agent.name == "orchestrator"
                                    else 2
                                    if agent.name.startswith("manager_")
                                    else 3
                                )
                                _observe_and_record(
                                    receiver=agent,
                                    content=drip_text,
                                    trial_id=tid,
                                    condition=condition,
                                    turn=turn,
                                    raw_exporter=raw_exporter,
                                    sender="evidence_drip",
                                    receiver_role=_role_from_level(lvl),
                                    persistence=persistence,
                                )

                    # Step 1: Orchestrator observes prev-turn manager reports,
                    # then acts (issues updated directive).
                    for m_idx, m_out in enumerate(manager_outputs):
                        routed = _extract_summary_view(m_out, f"manager_{m_idx:02d}")
                        _observe_and_record(
                            receiver=orchestrator,
                            content=f"[manager_{m_idx:02d} report]: {routed}",
                            trial_id=tid,
                            condition=condition,
                            turn=turn,
                            raw_exporter=raw_exporter,
                            sender=f"manager_{m_idx:02d}",
                            receiver_role="L1_ORCHESTRATOR",
                            persistence=persistence,
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
                        persistence=persistence,
                    )

                    # Step 2: Managers observe orchestrator directive +
                    # prev-turn analyst reports, then act.
                    for m_idx, manager in enumerate(managers):
                        # Orchestrator directive (summary only).
                        orch_routed = _extract_summary_view(
                            orchestrator_output, "orchestrator"
                        )
                        _observe_and_record(
                            receiver=manager,
                            content=f"[orchestrator directive]: {orch_routed}",
                            trial_id=tid,
                            condition=condition,
                            turn=turn,
                            raw_exporter=raw_exporter,
                            sender="orchestrator",
                            receiver_role="L2_MANAGER",
                            persistence=persistence,
                        )
                        # Analyst reports from previous turn.
                        for analyst_report in prev_analyst_outputs[m_idx]:
                            _observe_and_record(
                                receiver=manager,
                                content=analyst_report,
                                trial_id=tid,
                                condition=condition,
                                turn=turn,
                                raw_exporter=raw_exporter,
                                sender="analyst_group",
                                receiver_role="L2_MANAGER",
                                persistence=persistence,
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
                            persistence=persistence,
                        )
                        manager_outputs[m_idx] = m_out

                    # Step 3: Analysts observe fresh manager directive, then act.
                    new_analyst_outputs: list[list[str]] = [
                        [] for _ in range(self.N_MANAGERS)
                    ]
                    for m_idx, (manager, group) in enumerate(
                        zip(managers, analysts_by_manager)
                    ):
                        for analyst in group:
                            mgr_routed = _extract_summary_view(
                                manager_outputs[m_idx], f"manager_{m_idx:02d}"
                            )
                            _observe_and_record(
                                receiver=analyst,
                                content=f"[manager_{m_idx:02d} directive]: {mgr_routed}",
                                trial_id=tid,
                                condition=condition,
                                turn=turn,
                                raw_exporter=raw_exporter,
                                sender=f"manager_{m_idx:02d}",
                                receiver_role="L3_ANALYST",
                                persistence=persistence,
                            )
                            raw = _act_and_record(
                                analyst,
                                turn,
                                tid,
                                context,
                                condition,
                                level=3,
                                exporter=exporter,
                                raw_exporter=raw_exporter,
                                persistence=persistence,
                            )
                            new_analyst_outputs[m_idx].append(
                                _extract_summary_view(raw, analyst.name)
                            )

                    prev_analyst_outputs = new_analyst_outputs

            final_decision, final_confidence = _final_decision_from_raw(
                orchestrator_output
            )
            if persistence is not None:
                persistence.finalize_run(
                    run_id=tid,
                    final_decision=final_decision,
                    final_confidence=final_confidence,
                    correct=final_decision == context.ground_truth,
                )
        finally:
            if persistence is not None:
                persistence.close()

        logger.info("Hierarchical trial %s complete → %s", tid, out_path)
        return out_path
