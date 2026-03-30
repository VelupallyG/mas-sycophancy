# Codebase Audit Report: MAS Sycophancy Prototype vs. ACL Proposal & Research Questions

**Date:** 2026-03-29
**Scope:** All implemented code through Phase 2 (agents, topologies, metrics, game master, experiment runners)
**Reference Documents:** ACL-Format-Proposal.pdf, pivot_summary.md, CLAUDE.md

---

## CRITICAL Issues (Would Invalidate Results)

### C1. Simulation Does Not Use LLM Agents — Results Are Fabricated by RNG

**Files:** `src/game_master/simulation.py:148-170`

The `Simulation._simulate_stance()` method generates agent stances using `random.Random`, not by calling the LLM through Concordia's `EntityAgent.act()`. Agent "behavior" is entirely determined by hardcoded probability thresholds:

- Flat: 90% chance of correct stance, 10% hallucinated
- Hierarchical: pressure starts at 25% and grows by 6% per turn, with +10% for Level 4-5 agents

This means **the experiment's conclusion is baked into the code**. The hierarchical topology will always show higher sycophancy because the RNG probabilities are set that way. No LLM reasoning, no RLHF-driven deference, no emergent sycophancy — the core phenomenon the paper claims to study does not exist in this prototype. The prefabs are constructed but their `build()` methods are never called; no `EntityAgent` is ever instantiated or acted through.

**Impact:** Total. Every metric (delta-squared, ToF, NoF, TRAIL) computed from these traces reflects the researcher's assumptions, not empirical observations. This is circular reasoning: the hypothesis is encoded as the data-generating process.

**Status: FIXED.** Simulation rewritten to:
- Build real `EntityAgentWithLogging` instances from prefabs via `prefab.build(model, memory_bank)`
- Call `agent.observe()` to feed seed documents and peer statements
- Call `agent.act(action_spec)` to get LLM-generated natural language responses
- Extract stances post-hoc from agent text via keyword analysis (`extract_stance()`)
- Record the agent's actual text in trace logs
- New `src/model.py` provides `build_gemini_model()` (Vertex AI via Concordia's `GeminiModel`) and `MockLanguageModel` for testing
- All 141 tests pass


---

### C2. Flat Baseline Has No Hallucination Injection — Confounded IV

**Files:** `experiments/run_flat_baseline.py`, `src/game_master/simulation.py:208-211`

The flat baseline runs without hallucination injection (no `hallucinated_premise` is passed). The hierarchical condition runs *with* hallucination injection. This means **two independent variables change simultaneously**: topology (flat vs. hierarchical) AND hallucination injection (absent vs. present).

The ACL proposal (Section 3.1) is clear: both topologies should receive the same hallucinated signal from the orchestrator. The difference is how it propagates. In the flat topology, the hallucinating agent has no rank authority; in the hierarchical topology, it has Level-1 override power. The comparison isolates *structural authority*, not the presence/absence of misinformation.

**Impact:** RQ1 ("Do hierarchical MAS blindly converge to orchestrator hallucinations?") cannot be answered. Any accuracy difference could be attributed to the hallucination injection itself, not the topology. A reviewer would reject this as a basic confound.

**Status: FIXED.** Changes:
- `GameMasterConfig` now has `hallucination_recipient` and `hallucinated_claim` fields
- The simulation delivers the hallucinated claim as a private `agent.observe()` to the designated recipient before the first turn
- `run_flat_baseline.py` now loads the same seed document, extracts the hallucinated signal via `HallucinationInjector`, and delivers it to Agent_01 (a Level-5 peer with no rank authority)
- The only independent variable between conditions is now topology (peer vs. hierarchical authority)

---

### C3. delta-squared Definition Compares Wrong Quantities

**Files:** `src/metrics/sycophancy_effect.py:45-67`

The implementation computes `delta_squared = baseline_acc - influenced_acc` where `baseline_acc` is the flat topology accuracy and `influenced_acc` is the hierarchical topology accuracy.

The ACL proposal (Section 4.1, Equation 1) defines it differently: `A_0` is "the subordinate agent's baseline accuracy... when operating completely independently, devoid of any hierarchical pressure" and `A_i` is "the exact same agent's accuracy when subjected to the orchestrator's incorrect, hallucinated directive."

This is a **per-agent** metric, not a system-level metric. The proposal measures how much each individual subordinate's accuracy degrades when placed under hierarchical pressure. The current code computes a single system-level accuracy diff, which conflates the orchestrator's own accuracy (which should be low by design) with subordinate accuracy.

**Impact:** The reported delta-squared will not match the formal definition in the paper. It masks whether subordinates specifically are the ones degrading vs. the orchestrator dragging down the average. Per the proposal, you need the same agent's accuracy in independent mode vs. under-hierarchy mode.

**Status: FIXED.** `sycophancy_effect.py` rewritten with:
- `compute_per_agent_delta_squared()` — matches agents by name across baseline/influenced conditions, computes per-agent A_0 - A_i
- `AgentDeltaSquared` dataclass with per-agent results including hierarchy_level
- `exclude_orchestrator=True` by default (orchestrator accuracy is low by design)
- `compute_delta_squared_by_level()` — stratified mean for RQ2 (do lower-ranked agents show higher delta-squared?)
- `compute_mean_delta_squared()` — system-level summary
- Legacy `compute_delta_squared_from_logs()` preserved for backward compatibility
- 7 new tests pass covering per-agent matching, orchestrator exclusion, level stratification

---

### C4. Only One Agent Speaks Per Turn — Not a Multi-Agent Debate

**Files:** `src/game_master/simulation.py:234-241`

The simulation selects a single speaker per turn via `_determine_speaker()`. Over 10 turns with 20 agents, most agents never speak at all. The ACL proposal describes agents debating, exchanging views, and recursively feeding responses to each other. The Concordia framework supports multi-agent simultaneous or sequential action where ALL agents act each round.

In a proper Concordia simulation, every agent observes the scene, retrieves relevant memories, reasons about the situation, and produces an action each turn. The current code picks one agent per turn, fabricates a stance string, and moves on.

**Impact:** RQ2 ("Can a lower-ranked agent with the correct answer dynamically shift the group consensus?") requires observing how correct agents influence others over multiple exchanges. With one speaker per turn and no actual agent interaction, this dynamic cannot be studied. The rich multi-agent debate dynamics the paper claims to analyze do not exist.

**Status: FIXED (as part of C1 rewrite).** `_determine_turn_speakers()` now:
- Flat: ALL agents act every turn (peer-to-peer debate)
- Hierarchical: one full level acts per turn, cycling bottom-up (L5, L4, L3, L2, L1, repeat)
- Each speaker's output is broadcast as observations to other agents
- Test verifies: 3 agents * 2 turns = 6 turn records in flat mode

---

### C5. Accuracy Evaluation Is Keyword Matching, Not Prediction Evaluation

**Files:** `src/tasks/predictive_intel.py:132-166`

The `evaluate()` method scores predictions by checking if the ground truth direction string (e.g., "negative") and magnitude string (e.g., "moderate") appear as substrings in the prediction text. The prediction text is fabricated by `_compose_agent_text()` which always includes the stance word.

This means a prediction that says "negative" will always get 0.6 for direction match. A prediction of "catastrophic crash" won't match "negative" even though it's semantically negative. The evaluation is both too coarse (only 3 direction buckets) and too fragile (depends on exact keyword presence).

**Impact:** The accuracy scores that feed delta-squared are unreliable. They don't measure whether the agent's reasoning or prediction quality has degraded — they measure whether a hardcoded keyword appeared. Combined with C1 (stances are RNG-generated), accuracy is effectively `0.6 * P(RNG picks correct direction)`.

**Status: FIXED.** `evaluate()` rewritten with:
- Direction scoring (0.6) now uses `extract_stance()` — the same semantic keyword extraction used in the simulation — instead of literal substring matching. Handles synonyms like "crash"→negative, "rally"→positive, "catastrophic"→negative
- Magnitude scoring (0.2) now checks synonym sets (e.g. "major" matches "severe", "drastic", "massive", "extreme", "catastrophic")
- Key factor overlap (0.2) unchanged (literal factor matching is appropriate here since factors are controlled strings from seed documents)
- All 147 tests pass

---

## MAJOR Issues (Significantly Weaken Results)

### M1. Flat Topology Agents Are All Identical — No Persona Diversity

**Files:** `src/topologies/flat.py:61`

The flat topology creates N identical `AnalystPrefab` instances. The ACL proposal (Section 3.1, Flat MAS) and the pivot summary both specify that flat agents should have "distinct personas (e.g., retail investors, institutional traders, tech commentators)" operating in a "decentralized, peer-to-peer network."

The whole premise of the predictive intelligence pivot is that a diverse swarm of persona-driven agents provides value through parallel synthesis. If all agents are identical Level-5 Analysts with the same prompt, there is no meaningful diversity of perspective. This undermines the argument that flat MAS outperforms single-agent inference on this task.

---

### M2. No Concordia Game Master Is Used — Custom Simulation Instead

**Files:** `src/game_master/simulation.py`

The `Simulation` class is a custom Python class that doesn't use Concordia's actual `GameMaster` at all. Concordia provides `concordia.environment.game_master.GameMaster` which handles scene management, action resolution, state updates, and component coordination. The proposal (Section 3.1) explicitly states the simulation "will be built exclusively using the Concordia GABM library."

The current code reimplements turn-taking, state management, and logging from scratch, meaning none of Concordia's built-in plausibility checking, scene description generation, or component lifecycle management is used.

---

### M3. StanceTracker Component Is Never Called During Simulation

**Files:** `src/agents/components.py:118`, `src/game_master/simulation.py`

The `StanceTracker` component has a `record()` method that must be called externally after each turn. The simulation never calls it — it builds its own `turn_records` list instead. This means the custom component built specifically for this experiment is dead code, and the metrics pipeline can't use the component's `history` property as designed.

---

### M4. TRAIL Categorization Uses Signals That Don't Exist in Traces

**Files:** `src/metrics/trail.py:51-54, 57-108`

The `categorize_failure()` function expects trace entries to contain `planning_signal`, `deferred_to_authority`, `flip_count`, and `error` fields. The simulation's trace entries (written via `OtelExporter`) contain none of these fields — they write `agent_name`, `turn`, `stance`, `text`, `expected_stance`, and `hierarchy_level`.

There is no code anywhere that computes `planning_signal` or `deferred_to_authority` from agent output. The TRAIL classifier cannot distinguish between reasoning and planning errors with the current trace schema.

---

### M5. NoF Stratified Mean Hides Per-Level Sycophancy Patterns

**Files:** `src/metrics/flip_metrics.py:245-297`

The `compute_nof()` function macro-averages reversal counts across hierarchy levels. This means a hierarchy where Level-5 analysts flip 8 times each but Level-2 directors flip 0 times would produce the same NoF as one where all levels flip 4 times. RQ2 specifically asks about lower-ranked agents — the per-level breakdown is the scientifically interesting data, not the macro-average.

The proposal's NoF metric (Section 4.2) counts total stance reversals. The stratified mean is an invention that isn't in the paper's definition.

---

### M6. Hierarchical Consensus Override Forces the Orchestrator's Hallucinated Stance

**Files:** `src/game_master/simulation.py:330-361`

When the simulation ends and the last speaker wasn't Level-1, the code appends an extra record with `final_stance = hallucinated_stance`. This always produces the wrong answer for the hierarchical condition, regardless of what agents actually said. Combined with C2, this guarantees hierarchical accuracy < flat accuracy.

This isn't measuring whether agents converged to the hallucination — it's the GM forcibly appending the wrong answer at the end.

---

### M7. Semantic Compression Metric Uses Hash-Based Pseudo-Embeddings

**Files:** `src/metrics/linguistic.py:82-93`

The `_hashed_embedding()` function creates fake embeddings by hashing individual tokens into a fixed-size vector. This has no semantic content — "bank" and "financial institution" will produce completely unrelated vectors. The proposal (Section 4.4) describes "advanced embedding geometry analysis" measuring "intrinsic dimensionality of output embeddings."

Hash embeddings will produce random-looking compression ratios that have no relationship to actual semantic convergence. This metric will be meaningless noise.

---

## MODERATE Issues (Should Be Addressed Before Publication)

### m1. Hallucination Injection Is Static, Not LLM-Generated

The `HallucinationInjector.inject()` simply returns the pre-authored `hallucinated_signal` from the seed document JSON. The ACL proposal (Section 3.2) describes using Farm-style persuasive strategies (logical appeals, credibility appeals, emotional appeals) and MedHallu's structured generation pipeline. The current approach uses a single static fabricated claim. This is acceptable for Phase 1 prototyping but must evolve for publication.

### m2. Only 3 Seed Documents

The experiment has only 3 seed documents. For statistical validity, results over 3 scenarios provide very low statistical power. The proposal implies a much larger evaluation corpus. Even for a prototype, the results section will need many more scenarios to support any claims.

### m3. No Statistical Significance Testing

No code exists for computing confidence intervals, p-values, or effect sizes beyond the raw delta-squared. With stochastic simulation (or stochastic LLM outputs when C1 is fixed), multiple runs per condition are needed. The experiment runners execute one run per condition.

### m4. Missing `deference_lexicon.json`

The `linguistic.py` module references `src/metrics/deference_lexicon.json` but this file doesn't exist. The code falls back to a hardcoded 5-phrase lexicon. The proposal describes a "customized lexicon" — this needs to be authored properly.

### m5. OTel Exporter Is Not Actually OpenTelemetry

The `OtelExporter` class doesn't use the OpenTelemetry SDK at all — it's a custom JSON writer. The proposal and CLAUDE.md repeatedly reference "standardized open-telemetry format" and "open-telemetry execution tracing." This is a naming mismatch that would confuse reviewers. Either use the real OTel SDK or rename it.

### m6. `run_full_suite.py` Is Not Implemented

The full experiment orchestrator raises `NotImplementedError`. This is expected for Phase 2 but noted for completeness.

### m7. Config References `.txt` Extension for Hallucination Prompt

`src/config.py:40` sets `prompt_path` to `orchestrator_hallucination_v1.txt` but the actual file is `orchestrator_hallucination_v1.md`. This would cause a `FileNotFoundError` at runtime.

---

## Deviations From ACL Proposal (Document for Paper)

These are intentional design changes from the original proposal that should be recorded as methodology modifications:

| # | Proposal Says | Implementation Does | Rationale (Pivot Summary) |
|---|---|---|---|
| D1 | Farm + MedHallu datasets | Predictive Intelligence seed documents | Task validity — single LLM outperforms MAS on QA tasks; pivot to parallel synthesis where MAS excels |
| D2 | Medical diagnostics + persuasion tasks | Financial market prediction | Same rationale — domain where MAS adds genuine value |
| D3 | Multiple frontier models (GPT-4o, Claude 3.5, Gemini 1.5 Pro) | Single model (Gemini 2.5 Flash) | Cost constraint; isolates topology as IV |
| D4 | 20-40 persona-driven agents (flat) | 20 identical analysts | Simplification for prototype (but see M1 — this needs fixing) |
| D5 | TRAIL uses LLM-as-a-judge at temp=0.0 | TRAIL is rule-based heuristic | Consistent with "no LLM-as-a-judge" design principle in CLAUDE.md, but diverges from proposal Section 5.1 |

---

## Priority Summary

| Priority | Count | Action |
|----------|-------|--------|
| CRITICAL (invalidates results) | 5 | Must fix before any experiment run |
| MAJOR (significantly weakens) | 7 | Must fix before publication |
| MODERATE (should address) | 7 | Fix before submission |
| Documented deviations | 5 | Record in methodology section |

**The single most important fix is C1**: the simulation must actually run LLM agents through Concordia's EntityAgent lifecycle. Without this, there is no experiment — only a random number generator producing predetermined results shaped by hardcoded assumptions about sycophancy dynamics.
