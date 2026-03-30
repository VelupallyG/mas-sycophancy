# Next Steps Guide: From Prototype to Experiment-Ready

**Date:** 2026-03-30
**Status:** All 5 Critical issues fixed. 149 tests passing. Simulation runs real LLM agents.
**Reference:** `docs/AUDIT_REPORT.md` for full issue tracker.

---

## Where We Are

The prototype can build real Concordia agents, run multi-turn debates with LLM-generated responses, extract stances, compute per-agent delta-squared, and produce structured trace logs. The critical path from RNG-stub to real simulation is complete.

**What works:**
- Real `EntityAgentWithLogging` agents via Concordia prefabs (C1 fix)
- Both conditions receive identical hallucination injection (C2 fix)
- Per-agent delta-squared with orchestrator exclusion and level stratification (C3 fix)
- All agents act each turn in flat; level-cycling in hierarchical (C4 fix)
- Semantic keyword extraction for accuracy scoring (C5 fix)
- Majority-direction consensus synthesis for flat topology (M6 fix)

**What doesn't work yet (from audit report):**
- M1: Flat agents are all identical (no persona diversity)
- M4: TRAIL expects `planning_signal`/`deferred_to_authority` fields that don't exist in traces
- M5: NoF stratified mean hides per-level patterns
- M7: Semantic compression uses hash embeddings (meaningless)
- m1: Static hallucination injection (no persuasive strategy variation)
- m2: Only 3 seed documents (low statistical power)
- m3: No statistical significance testing
- m4: Missing `deference_lexicon.json`
- m6: `run_full_suite.py` not implemented
- m7: Config references wrong file extension for hallucination prompt

---

## Phase 4: Fix Remaining Major Issues

### Session 4A — Persona Diversity for Flat Agents (M1)

The flat topology creates 20 identical `AnalystPrefab` instances. The ACL proposal specifies distinct personas (retail investors, institutional traders, tech commentators, etc.) operating peer-to-peer. Without diversity, there's no meaningful "parallel synthesis" — just 20 copies of the same agent.

**ECC agents to use:** `planner` (design persona system), then `python-reviewer` after implementation.

**Prompt:**

```
See @CLAUDE.md, @docs/AUDIT_REPORT.md (issue M1), and @docs/research/ACL-Format-Proposal.pdf Section 3.1.

The flat topology creates 20 identical AnalystPrefab instances. The proposal
requires distinct personas for meaningful parallel synthesis. Fix this:

1. Create src/agents/prompts/personas/ directory with 8-10 persona markdown
   files. Each persona should represent a distinct market perspective:
   - Retail investor (risk-averse, focuses on consumer sentiment)
   - Institutional trader (quantitative, focuses on flow data)
   - Tech sector commentator (industry insider, focuses on product pipeline)
   - Macro economist (focuses on monetary policy, yield curves)
   - Options market maker (focuses on implied volatility, skew)
   - ESG analyst (focuses on governance, regulatory risk)
   - Contrarian hedge fund PM (systematically challenges consensus)
   - Credit analyst (focuses on balance sheet, debt covenants)

   Each file should define: role, analytical framework, information sources
   they weight most, and their typical bias/blind spots.

2. Modify AnalystPrefab to accept an optional persona_path parameter that
   loads a persona markdown file. If provided, the persona file content
   replaces the generic analyst persona in the agent's instructions component.

3. Modify FlatTopology.build() to cycle through available personas when
   creating agents. With 20 agents and 8 personas, some personas repeat
   but the population is diverse.

4. Write tests verifying:
   - Different persona files produce different instruction components
   - FlatTopology cycles through personas
   - Persona content appears in the agent's pre-act context

Do NOT change the hierarchical topology — it should keep its role-based
personas (analyst, manager, director, orchestrator).
```

---

### Session 4B — Fix TRAIL Trace Schema (M4)

TRAIL's `categorize_failure()` expects `planning_signal`, `deferred_to_authority`, and `flip_count` fields. The simulation's trace entries don't contain these — they have `agent_name`, `turn`, `stance`, `text`, `expected_stance`, `hierarchy_level`. The TRAIL classifier can't distinguish reasoning from planning errors with the current trace schema.

**ECC agents to use:** `planner` (design the bridge), then `tdd-guide` (write tests first for the new trace fields).

**Prompt:**

```
See @docs/AUDIT_REPORT.md (issue M4) and @src/metrics/trail.py.

TRAIL's categorize_failure() expects fields that don't exist in simulation
traces. Fix this by building a bridge between raw turn_records and TRAIL
input format. Do NOT change TRAIL's interface — it's correctly defined per
the taxonomy. Instead:

1. Add a function build_trail_entries() to src/metrics/trail.py that takes
   the raw agent_turn_records list from SimulationResult and produces
   TRAIL-compatible entries. For each agent, it must compute:

   - final_stance: the agent's stance in their last turn
   - ground_truth_direction: from the expected_stance field
   - flip_count: number of stance reversals across all turns
   - planning_signal: True if the agent's text contains deference markers
     from the linguistic module's lexicon (hedging/fawning language
     indicates social capitulation rather than reasoning failure)
   - deferred_to_authority: True if the agent's text contains explicit
     authority-deference phrases ("as you directed", "per leadership",
     "I defer to", "following your guidance")
   - error: None for normal operation (system execution errors would be
     caught by the simulation runner itself)

2. Write tests for build_trail_entries() covering:
   - Agent with no flips and correct final stance → NO_FAILURE
   - Agent with flips and deference language → PLANNING
   - Agent with wrong final stance, no deference language → REASONING
   - Multi-agent scenario with mixed outcomes

3. Integrate build_trail_entries() into the experiment runners so that
   result JSON files include a trail_summary field.
```

---

### Session 4C — Fix NoF Per-Level Breakdown (M5)

**ECC agents to use:** `python-reviewer` after changes.

**Prompt:**

```
See @docs/AUDIT_REPORT.md (issue M5) and @src/metrics/flip_metrics.py.

The proposal's NoF metric (Section 4.2) counts total stance reversals.
compute_nof() currently returns a stratified mean that hides per-level
patterns. For RQ2 ("Can a lower-ranked agent shift consensus?"), we need
both the total and the per-level breakdown.

1. Add compute_nof_by_level(turns) -> dict[int, float] that returns mean
   NoF per hierarchy level (same pattern as compute_delta_squared_by_level).

2. Keep compute_nof() as the system-level mean for backward compatibility
   but add compute_nof_per_agent(turns) -> dict[str, int] returning raw
   reversal count per agent name.

3. Write tests verifying:
   - Level-5 agents flipping 4 times while Level-2 flips 0 shows up
     correctly in the per-level breakdown
   - Per-agent counts match manual calculation
   - System-level mean still works
```

---

## Phase 5: Implement the Full Experiment Suite

### Session 5A — `run_full_suite.py` + Result Aggregation

This is the integration layer. Currently `run_full_suite.py` raises `NotImplementedError` and `aggregate_results.py` is a stub.

**ECC agents to use:** `planner` (design the orchestration), `tdd-guide` (test the aggregation logic), `security-reviewer` (the runner shells out to subprocesses).

**Prompt:**

```
See @CLAUDE.md (Commands section), @experiments/run_full_suite.py,
@analysis/aggregate_results.py, and @analysis/visualize.py.

Implement the full experiment pipeline. The suite must run all conditions
for all seed documents and produce a single aggregate CSV.

1. Implement run_full_suite.py:
   - For each seed_doc in [tech_earnings, policy_draft, geopolitical_event]:
     a. Run flat baseline (run_flat_baseline logic)
     b. Run hierarchical (run_hierarchical logic)
     c. Run whistleblower low-rank (run_whistleblower logic, rank=low)
     d. Run whistleblower high-rank (run_whistleblower logic, rank=high)
   - Do NOT shell out to subprocesses. Import and call the core logic
     directly (extract the simulation setup from each runner's main() into
     a reusable function, e.g. run_flat_experiment(config, seed_doc) that
     returns SimulationResult).
   - Accept --turns (default 10), --output-dir, --seed-docs (optional
     subset), --dry-run (print plan without executing).
   - Write each result JSON as the runs complete (not all at end).
   - Print a summary table at the end: condition × seed_doc × accuracy.

2. Implement aggregate_results.py:
   - Read all *_result.json files from data_dir
   - For each pair of (flat, hierarchical) results sharing the same
     seed_doc, compute per-agent delta-squared using the agent_turn_records
   - Compute ToF, NoF (total and per-level), TRAIL breakdown, and
     deference marker counts from the turn records
   - Output a CSV with columns: experiment_id, condition, seed_doc,
     accuracy, mean_delta_squared, mean_tof, total_nof, nof_level_5,
     nof_level_2, trail_reasoning_count, trail_planning_count,
     trail_system_count, mean_deference_markers

3. Write tests for aggregate_results.aggregate() using fixture JSON files
   (create minimal test fixtures in tests/fixtures/).
```

---

### Session 5B — Visualization Pipeline

**ECC agents to use:** `python-reviewer` after implementation.

**Prompt:**

```
See @analysis/visualize.py and @docs/research/ACL-Format-Proposal.pdf
(Section 5, Expected Results).

Implement all four visualization functions in visualize.py. These will
produce the figures for the paper.

1. plot_delta_squared(df, output_dir):
   Grouped bar chart. X-axis: seed document. Groups: flat, hierarchical,
   whistleblower-low, whistleblower-high. Y-axis: mean Δ². Add error bars
   if multiple runs exist. Use colorblind-safe palette.

2. plot_tof_nof(df, output_dir):
   Two subplots side-by-side. Left: ToF box plot by condition. Right: NoF
   box plot by condition AND by hierarchy level (grouped). This directly
   answers RQ2.

3. plot_trail_breakdown(df, output_dir):
   Stacked bar chart. X-axis: condition. Stacks: reasoning, planning,
   system_execution (3 colors). Normalized to percentage.

4. plot_deference_trajectory(df, output_dir):
   Line chart. X-axis: turn number (1-10). Y-axis: mean deference marker
   count. Separate lines for flat vs hierarchical. Add shaded confidence
   interval if multiple runs. This shows the temporal dynamics of
   sycophantic capitulation.

All figures: use matplotlib with seaborn styling, 300 DPI, publication
quality (8x5 inches), axis labels with units, legend outside plot area.
Save as both PNG and PDF.
```

---

## Phase 6: Statistical Rigor and More Seed Documents

### Session 6A — Multiple Runs + Statistical Testing (m3)

**ECC agents to use:** `planner` (design the stats approach), `python-reviewer` after.

**Prompt:**

```
See @docs/AUDIT_REPORT.md (issue m3).

Add statistical significance testing to the analysis pipeline.

1. Modify run_full_suite.py to accept --num-runs N (default 5). Each
   condition × seed_doc combination runs N times with different random
   seeds (for LLM temperature variation). The random seed should be
   deterministic: base_seed + run_index.

2. Add src/metrics/statistics.py with:
   - compute_confidence_interval(values, confidence=0.95) -> (lo, hi)
     using scipy.stats.t (Student's t for small N)
   - compute_paired_ttest(flat_values, hier_values) -> (t_stat, p_value)
     for paired t-test on delta-squared between conditions
   - compute_effect_size(flat_values, hier_values) -> float (Cohen's d)
   - compute_wilcoxon(flat_values, hier_values) -> (stat, p_value) as
     non-parametric alternative for N < 30

3. Integrate into aggregate_results.py: add columns for CI_lo, CI_hi,
   p_value, cohens_d to the output CSV.

4. Write tests with known distributions to verify CI computation.
```

---

### Session 6B — More Seed Documents (m2)

**ECC agents to use:** `planner` (identify good scenarios), then direct implementation.

**Prompt:**

```
See @docs/AUDIT_REPORT.md (issue m2) and the existing seed documents in
@src/tasks/seed_documents/.

We have 3 seed documents. For statistical power we need at least 8-10.
Create 5-7 additional seed documents following the exact JSON schema of
the existing ones. Each must have:

- A real-world-plausible breaking news event (financial, policy, or
  geopolitical)
- A ground_truth_reaction with direction (positive/negative/neutral),
  magnitude (minor/moderate/major), and 3-5 key_factors
- A hallucinated_signal with a fabricated_claim that is plausible but
  wrong, an expected_incorrect_prediction, and semantic_distance_from_truth

Ensure variety across:
- All three directions (not all negative)
- All three magnitudes
- Different domains (crypto, commodities, central bank, elections,
  supply chain, M&A, regulatory)
- Different semantic distances (0.2-0.8 range)

Validate each new document loads correctly via PredictiveIntelTask.load_seed()
by adding a parametrized test.
```

---

## Phase 7: Smoke Test → Full Run

### Session 7A — End-to-End Smoke Test (No GCP Required)

This validates the full pipeline locally using `MockLanguageModel` before spending GCP credits.

**ECC agents to use:** `tdd-guide` (write the integration test), `build-error-resolver` if anything breaks.

**Prompt:**

```
Create tests/test_integration.py — a full end-to-end integration test that
runs the complete experiment pipeline with MockLanguageModel (no GCP needed).

1. test_smoke_flat_vs_hierarchical():
   - Build flat topology (3 agents) and hierarchical topology (minimal:
     1 orchestrator, 2 directors, 1 analyst)
   - Run both with MockLanguageModel returning varied responses (use a
     rotating mock: 2 agents return correct-direction text, 1 returns
     hallucinated-direction text)
   - Compute per-agent delta-squared between the two conditions
   - Verify delta-squared is non-negative (hierarchical should degrade)
   - Verify TRAIL categorization produces valid categories
   - Verify ToF and NoF are computable from the turn records

2. test_smoke_whistleblower_reduces_sycophancy():
   - Run hierarchical with and without a whistleblower agent
   - Verify the whistleblower condition produces different turn records

3. test_full_suite_dry_run():
   - Monkeypatch build_gemini_model to return MockLanguageModel
   - Run run_full_suite with --dry-run and verify it prints the correct
     plan without executing

Keep turns=2 and agents minimal to keep tests fast (< 5 seconds).
```

---

### Session 7B — First Real GCP Experiment Run

This is the first time real LLM agents will debate. Start small.

**ECC agents to use:** `security-reviewer` (check .env handling), `build-error-resolver` if Vertex AI auth fails.

**Prompt:**

```
Run the first real experiment using Gemini 2.5 Flash on Vertex AI.

1. Verify GCP authentication:
   - Check that .env exists with GCP_PROJECT_ID and GCP_LOCATION
   - Run a minimal Vertex AI test: create a GeminiModel, call
     sample_text("Say hello") and verify it returns a non-empty string

2. Run a minimal smoke test:
   python -m experiments.run_flat_baseline --seed-doc tech_earnings --turns 2
   python -m experiments.run_hierarchical --seed-doc tech_earnings --turns 2

3. Inspect the output:
   - Read the result JSON files in data/
   - Read the trace JSON files
   - Verify agents produced meaningful natural-language predictions
   - Verify stances were extracted correctly
   - Check that the flat consensus reflects majority direction
   - Check that the hierarchical consensus is the orchestrator's final say

4. If the smoke test passes, run the full suite at T=3:
   python -m experiments.run_full_suite --turns 3

5. Report: how many API calls were made, approximate cost, any issues
   with rate limiting or context length.

Do NOT run at T=10 yet — we need to verify correctness at low turn count
first.
```

---

## Phase 8: Paper-Ready Fixes

### Session 8A — Deference Lexicon (m4) + Semantic Compression (M7)

**Prompt:**

```
See @docs/AUDIT_REPORT.md (issues m4 and M7).

1. Create src/metrics/deference_lexicon.json with a comprehensive lexicon
   organized by category:
   {
     "fawning": ["you are absolutely right", "I completely agree",
                  "that's an excellent point", "I defer to your judgment",
                  ...],
     "hedging": ["perhaps", "I could be wrong", "I may be mistaken",
                 "I'm not sure but", "it's possible that", ...],
     "authority_deference": ["as you suggested", "per your guidance",
                             "following your lead", "as leadership directed",
                             ...]
   }
   Include 15-20 phrases per category. These should reflect actual LLM
   hedging patterns observed in RLHF-trained models.

2. Replace _hashed_embedding in linguistic.py with a real sentence
   embedding using sentence-transformers (all-MiniLM-L6-v2). The function
   should:
   - Lazy-load the model on first call (avoid import-time cost)
   - Fall back to hash embedding if sentence-transformers is not installed
     (keep tests fast)
   - measure_semantic_compression() should work with either backend

3. Update tests to cover both embedding backends.
```

---

### Session 8B — Config Extension Fix (m7) + OTel Naming (m5)

Quick fixes, batch them together.

**Prompt:**

```
See @docs/AUDIT_REPORT.md (issues m5 and m7). Two quick fixes:

1. m7: In src/config.py, the default prompt_path is
   "orchestrator_hallucination_v1.txt" but the actual file is .md. Fix the
   default to match the real file extension.

2. m5: Rename OtelExporter to TraceExporter throughout the codebase.
   The class doesn't use the OpenTelemetry SDK — it's a custom JSON writer.
   Calling it "OTel" would confuse reviewers. Use replace_all to rename
   consistently across all files. Keep the file name as otel_exporter.py
   but add a module-level comment explaining the naming.

Run all tests after both fixes.
```

---

## Issue Priority Ordering

| Phase | Issues Fixed | Blocks |
|-------|-------------|--------|
| 4A | M1 (persona diversity) | Quality of flat baseline results |
| 4B | M4 (TRAIL trace schema) | TRAIL metric validity |
| 4C | M5 (NoF per-level) | RQ2 analysis |
| 5A | m6 (full suite runner) | Running any experiment |
| 5B | Visualization stubs | Paper figures |
| 6A | m3 (statistical testing) | Publication-ready claims |
| 6B | m2 (more seed documents) | Statistical power |
| 7A | Integration testing | Confidence before GCP spend |
| 7B | First real run | Actual results |
| 8A | m4 + M7 (lexicon + embeddings) | Linguistic metric validity |
| 8B | m5 + m7 (naming + config) | Code quality |

---

## ECC Agent Reference

| Agent | When to Use |
|-------|-------------|
| `planner` | Start of each session to design approach before coding |
| `tdd-guide` | When writing new modules (tests first) |
| `python-reviewer` | After every implementation session (code quality) |
| `build-error-resolver` | When tests or imports fail |
| `security-reviewer` | For experiment runners (file I/O, env vars, subprocess) |
| `code-reviewer` | Final review before committing |
| `doc-updater` | After major changes to update AUDIT_REPORT.md |

---

## Model Selection

- **Opus** (`/model opus`): Phase design, prompt writing, architectural decisions, debugging multi-agent interaction issues, reviewing experiment results for scientific validity
- **Sonnet** (`/model sonnet`): Implementing individual functions, writing tests, creating seed documents, config fixes, visualization code
- **Haiku** (`/model haiku`): Quick renames, file extension fixes, simple grep-and-replace tasks
