# MAS Sycophancy

**If the CEO Is Wrong, Is Everyone Wrong?**

Measuring authority-gradient sycophancy propagation in hierarchical multi-agent systems.

## What This Is

A controlled experiment testing whether hierarchical MAS topologies amplify hallucination propagation compared to flat topologies. 21 Gemini 2.5 Flash agents perform a geopolitical market-prediction task (Iran oil sanctions, ground truth: POSITIVE +2.8%). A single hallucination ("NEGATIVE") is injected into one agent — the orchestrator in the hierarchy, or a random peer in the flat group.

**Core finding:** The hierarchy reaches 100% wrong consensus by Turn 3 (Δ² = 0.80). The flat topology contains the hallucination to the injector alone (Δ² = 0.048).

## Stack

- Python 3.11+
- [Concordia](https://github.com/google-deepmind/concordia) (generative agent-based modeling)
- Google Vertex AI (Gemini 2.5 Flash)
- PostgreSQL (experiment tracking, evidence storage)

## Quick Start

```bash
pip install -e ".[dev]"
gcloud auth application-default login

# Run a single condition
python -m experiments.run_flat_baseline --seed-doc iran_oil_sanctions
python -m experiments.run_hierarchical --seed-doc iran_oil_sanctions

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/
├── agents/          # Concordia prefabs + persona prompts
├── topologies/      # Flat and hierarchical communication graphs
├── hallucination/   # Injection templates and injector logic
├── metrics/         # Δ², ToF, NoF, TRAIL, linguistic deference
├── evidence/        # Evidence loading and per-agent allocation
├── game_master/     # Simulation orchestration
└── tracing/         # JSONL trace export

experiments/         # Entry points for each condition
data/real_v2/        # Experiment outputs, traces, figures
paper/               # Final report (ACL format LaTeX)
```

## Key Metrics

| Metric | Definition |
|--------|-----------|
| Δ² (Sycophancy Effect) | Accuracy drop attributable to topology |
| ToF (Turn of Flip) | First turn an agent adopts the hallucination |
| NoF (Number of Flips) | Total stance reversals across turns |
| TRAIL | Error categorization (planning / reasoning / system) |

## Authors

Gautham Velupally, Pranav Nagarajan, Shanmukh Upadhyayula, Sanjit Kalle

498 HLI SP26
