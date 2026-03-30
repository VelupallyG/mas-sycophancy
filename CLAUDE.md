# MAS Sycophancy & Hallucination Propagation Research Prototype

This project investigates whether hierarchical Multi-Agent System (MAS) topologies systematically amplify hallucination propagation compared to flat topologies — the "Yes-Man collapse" phenomenon (regressive sycophancy).

## Tech Stack
- **Language:** Python 3.11+
- **Simulation Framework:** Concordia GABM (Google DeepMind) — `pip install gdm-concordia`
- **LLM Backend:** Google Vertex AI Python SDK (`google-cloud-aiplatform`) — Gemini 2.5 Flash via `vertexai.generative_models`
- **Evaluation:** OpenTelemetry tracing, programmatic metrics (no LLM-as-a-judge for core metrics)
- **Data:** JSON-based seed documents for the Predictive Intelligence task
- **Analysis:** pandas, matplotlib, seaborn for metric visualization
- **Testing:** pytest

## Project Structure

```
mas-sycophancy/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview for humans
├── pyproject.toml               # Dependencies and project metadata
├── .env.example                 # GCP project config template
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Experiment configuration dataclasses
│   │
│   ├── agents/                  # Concordia Prefab definitions (one per role)
│   │   ├── __init__.py
│   │   ├── components.py        # Custom reusable components (HierarchicalRank, StanceTracker)
│   │   ├── prefab_common.py     # Shared prefab construction helpers (composition utilities)
│   │   ├── analyst_prefab.py    # Level 5 Entry Analyst — @dataclass Prefab with build()
│   │   ├── manager_prefab.py    # Levels 3-4 Manager — @dataclass Prefab with build()
│   │   ├── director_prefab.py   # Level 2 Senior Director — @dataclass Prefab with build()
│   │   ├── orchestrator_prefab.py # Level 1 CEO/CSO — receives hallucinated signal via premise
│   │   ├── whistleblower_prefab.py # Anti-deference agent — rank parameterized in build()
│   │   └── prompts/             # Persona and instruction text files (not hardcoded)
│   │       ├── analyst.md
│   │       ├── orchestrator.md
│   │       ├── orchestrator_hallucination_v1.md
│   │       └── whistleblower.md
│   │
│   ├── topologies/              # MAS topology constructors
│   │   ├── __init__.py
│   │   ├── flat.py              # Peer-to-peer flat MAS (control group)
│   │   └── hierarchical.py      # 5-level stratified MAS (experimental group)
│   │
│   ├── tasks/                   # Task definitions and seed documents
│   │   ├── __init__.py
│   │   ├── predictive_intel.py  # MiroFish-style predictive intelligence task
│   │   └── seed_documents/      # JSON seed documents for scenarios
│   │       ├── tech_earnings.json
│   │       ├── policy_draft.json
│   │       └── geopolitical_event.json
│   │
│   ├── hallucination/           # Hallucination injection engine
│   │   ├── __init__.py
│   │   └── injector.py          # Controlled hallucination generation
│   │
│   ├── metrics/                 # Programmatic evaluation
│   │   ├── __init__.py
│   │   ├── sycophancy_effect.py # Δ² calculation
│   │   ├── flip_metrics.py      # ToF and NoF calculation
│   │   ├── trail.py             # TRAIL framework error categorization
│   │   └── linguistic.py        # Deference markers and semantic compression
│   │
│   ├── tracing/                 # OpenTelemetry instrumentation
│   │   ├── __init__.py
│   │   └── otel_exporter.py     # Trace export and parsing
│   │
│   └── game_master/             # Concordia Game Master configuration
│       ├── __init__.py
│       └── simulation.py        # GM setup, ground truth, state management
│
├── experiments/                 # Experiment runners
│   ├── __init__.py
│   ├── run_flat_baseline.py     # Run flat topology baseline
│   ├── run_hierarchical.py      # Run hierarchical topology experiment
│   ├── run_whistleblower.py     # Run whistleblower intervention variants
│   └── run_full_suite.py        # Orchestrate complete experiment suite
│
├── analysis/                    # Post-experiment analysis
│   ├── __init__.py
│   ├── aggregate_results.py     # Combine and analyze experiment outputs
│   └── visualize.py             # Generate charts and figures
│
├── data/                        # Experiment output data (gitignored)
│   └── .gitkeep
│
├── tests/
│   ├── test_agents.py
│   ├── test_topologies.py
│   ├── test_metrics.py
│   ├── test_hallucination_injector.py
│   └── test_game_master.py
│
└── docs/
    ├── ARCHITECTURE.md          # System architecture documentation
    ├── METRICS.md               # Detailed metric definitions
    └── EXPERIMENT_PROTOCOL.md   # Step-by-step experiment protocol
```

## Key Architecture Decisions

1. **Concordia v2.0 Prefab + Component composition (NOT class inheritance).** Each agent role is a `@dataclasses.dataclass` class inheriting from `prefab_lib.Prefab`. It implements a `build(self, model, memory_bank)` method that composes Concordia Components (memory, observation, instructions, QuestionOfRecentMemories, concat_act_component, etc.) and returns an `entity_agent.EntityAgent`. There is NO base agent class. Key imports:
   ```python
   import dataclasses
   from concordia.agents import entity_agent
   from concordia.components.agent import (
       memory, observation, instructions,
       question_of_recent_memories, concat_act_component,
   )
   from concordia.typing import prefab as prefab_lib
   ```
   Reference the actual Concordia repo for patterns:
   - `examples/concordia_cheat_sheet.md` (full API walkthrough)
   - `concordia/components/README.md` (component catalog)
   - `concordia/prefabs/entity/basic.py` (reference Prefab implementation)

2. **Game Master ≠ Orchestrator.** The Concordia Game Master is the objective simulation engine (manages state, logs, ground truth). The Orchestrator is a *participating agent* (the CEO) inside the simulation. Never confuse these roles.

3. **Single model: Gemini 2.5 Flash via Vertex AI.** All agents use the same model (`gemini-2.5-flash-002` via the `vertexai` Python SDK). This isolates topology as the independent variable. We use Vertex AI (not the Gemini Developer API) because GCP credits apply. The SDK reference is at https://docs.cloud.google.com/python/docs/reference/vertexai/latest. Initialize with:
   ```python
   import vertexai
   from vertexai.generative_models import GenerativeModel

   vertexai.init(project="YOUR_PROJECT_ID", location="us-central1")
   model = GenerativeModel("gemini-2.5-flash-002")
   ```

4. **Programmatic metrics only.** No LLM-as-a-judge for core evaluation. All metrics (Δ², ToF, NoF, TRAIL) are calculated from execution traces exported by the Game Master's OpenTelemetry logs.

5. **T=10 turn horizon.** Each experimental instance is capped at 10 conversational turns to manage costs and match literature standards for capturing sycophancy dynamics.

## The Task: Predictive Intelligence Simulation

Instead of medical QA (where a single LLM outperforms MAS), the task is a **parallel synthesis and prediction** problem where MAS genuinely excels:

- **Seed Document:** Real-world breaking news (financial, policy, geopolitical)
- **Objective:** Predict market/public reaction
- **Flat MAS:** 20-40 persona-driven agents debate peer-to-peer to reach consensus prediction
- **Hierarchical MAS:** Same agents forced into 5-level corporate reporting chain
- **Hallucination Injection:** The Orchestrator (CSO) receives the seed document but is prompt-engineered to fabricate a wildly incorrect market signal
- **Ground Truth:** The actual market/public reaction (sourced post-hoc or from historical data)

## Metrics (Formal Definitions)

- **Δ² (Sycophancy Effect Size):** `A₀ - Aᵢ` where A₀ = baseline accuracy (independent), Aᵢ = accuracy under orchestrator pressure
- **ToF (Turn of Flip):** `E[min t | y_i^(t) ≠ y_expected_i]` — the first turn where the agent abandons its correct stance
- **NoF (Number of Flip):** Total stance reversals across all T turns
- **TRAIL:** Categorize each failure as Reasoning Error, Planning Error, or System Execution Error

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Authenticate with GCP
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Run tests
pytest tests/ -v

# Run a single flat baseline experiment
python -m experiments.run_flat_baseline --seed-doc tech_earnings

# Run hierarchical experiment with hallucination injection
python -m experiments.run_hierarchical --seed-doc tech_earnings

# Run whistleblower variant
python -m experiments.run_whistleblower --rank low --seed-doc tech_earnings

# Run full experiment suite
python -m experiments.run_full_suite

# Analyze results
python -m analysis.aggregate_results --data-dir data/
python -m analysis.visualize --data-dir data/ --output-dir figures/
```

## Development Rules

- Always write tests before implementing new components
- Type-hint everything (use dataclasses for config, Pydantic for validation)
- Keep agent system prompts in separate .txt or .md files, not hardcoded in Python
- Log everything — every agent action, every state transition, every GM decision
- Use structured logging (JSON format) for machine-parseable experiment outputs
- Never modify Concordia library internals — extend via components
- Keep hallucination injection prompts versioned and reproducible
- Pin all random seeds for reproducibility

## Research Questions Mapping

| RQ | What it tests | Implementation |
|----|--------------|----------------|
| RQ1 | Do hierarchical MAS blindly converge to orchestrator hallucinations? | Compare Flat vs Hierarchical Δ² |
| RQ2 | Can a lower-ranked correct agent shift consensus? | Track ToF and NoF for Level 5 analysts |
| RQ3 | Does a Whistleblower agent disrupt forced convergence? | Inject Whistleblower, compare Δ² to baseline |
| RQ4 | Does Whistleblower rank matter? | Compare Low-rank (L5) vs High-rank (L2) Whistleblower ToF and Δ² |
