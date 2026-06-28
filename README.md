# MAS Sycophancy

Controlled experiments on how hallucinations spread through teams of LLM agents, and which communication structures make those errors harder to contain.

**Paper:** [Sycophancy in Hierarchical versus Flat Multi-Agent Systems](https://drive.google.com/file/d/16sKAof3bSq-tglBCidpEvbPG1wiejZDK/view?usp=sharing)

## Summary

This project compares two multi-agent system designs:

- **Flat topology:** 21 peer agents deliberate together.
- **Hierarchical topology:** 21 agents are organized into a 3-level manager/analyst tree.

Each condition uses Gemini 2.5 Flash agents on a geopolitical market-prediction task. A single false claim is injected into either the hierarchy's top-level orchestrator or a randomly selected flat peer. The goal is to measure whether the system corrects the false claim or amplifies it.

## Key Result

The hierarchical system fully adopted the hallucinated answer by Turn 3, while the flat system largely isolated the error.

| Condition | Direction Accuracy | Sycophancy Effect |
| --- | ---: | ---: |
| Flat baseline | 100.0% | 0.000 |
| Flat hallucination | 95.2% | 0.048 |
| Hierarchical baseline | 100.0% | 0.000 |
| Hierarchical hallucination | 20.0% | 0.800 |

Interpretation: the hierarchy introduced an 80 percentage-point accuracy drop attributable to topology, while the flat structure limited spread to the injected peer.

## Implementation

- Agent prefabs and personas for analyst, manager, and orchestrator roles.
- Flat and hierarchical communication topologies.
- Hallucination injection logic for controlled intervention experiments.
- Experiment runners for flat baseline, flat hallucination, hierarchical baseline, and hierarchical hallucination conditions.
- Metrics for error spread, turn of flip, number of flips, and TRAIL-style error categorization.
- JSONL tracing and optional PostgreSQL persistence for experiment outputs.

## Tech Stack

- Python 3.13+
- Google DeepMind Concordia
- Google Vertex AI / Gemini 2.5 Flash
- PostgreSQL
- pandas, matplotlib, seaborn
- pytest

## Repository Layout

```text
src/agents/          Concordia agent prefabs and output parsing
src/topologies/      Flat and hierarchical communication graphs
src/hallucination/   Injection templates and injector logic
src/metrics/         Delta-squared, turn-of-flip, TRAIL-style metrics
src/evidence/        Evidence loading and per-agent allocation
src/game_master/     Simulation orchestration
src/tracing/         JSONL trace export
experiments/         Entrypoints for each experimental condition
analysis/            Post-processing and visualization utilities
tests/               Unit and smoke tests
```

## Quick Start

```bash
pip install -e ".[dev]"
gcloud auth application-default login

python -m experiments.run_flat_baseline --seed-doc iran_oil_sanctions
python -m experiments.run_hierarchical --seed-doc iran_oil_sanctions

pytest tests/ -v
```

## Notes

Results in `FINDINGS.md` are from the prototype experimental run and should be interpreted as directional evidence. The paper discusses limitations and proposed extensions such as more seeds, model sweeps, and intervention agents.

## Authors

Gautham Velupally, Pranav Nagarajan, Shanmukh Upadhyayula, Sanjit Kalle
