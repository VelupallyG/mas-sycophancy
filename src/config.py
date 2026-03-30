"""Experiment configuration dataclasses.

All experiment parameters are centralised here. Load from environment
variables via ``load_config_from_env()`` or construct directly in tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Parameters that govern individual agent behaviour."""

    model_id: str = "gemini-2.5-flash-002"
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    temperature: float = 0.7
    max_output_tokens: int = 1024


@dataclass
class TopologyConfig:
    """Parameters for topology construction."""

    num_agents: int = 20
    # Hierarchical level counts: keys are levels 1-5
    level_counts: dict[int, int] = field(
        default_factory=lambda: {1: 1, 2: 2, 3: 4, 4: 4, 5: 9}
    )


@dataclass
class HallucinationConfig:
    """Parameters for the hallucination injection engine."""

    prompt_version: str = "v1"
    # Path to the versioned prompt template (relative to repo root)
    prompt_path: str = "src/agents/prompts/orchestrator_hallucination_v1.txt"
    random_seed: int = 42


@dataclass
class MetricsConfig:
    """Parameters for the metrics pipeline."""

    # Minimum cosine distance to classify a claim as a flip
    flip_threshold: float = 0.5
    # Path to the deference-marker lexicon (JSON array of strings)
    deference_lexicon_path: str = "src/metrics/deference_lexicon.json"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_id: str = "exp_001"
    seed_doc: str = "tech_earnings"
    max_turns: int = 10
    random_seed: int = 42
    output_dir: str = "data/"

    agent: AgentConfig = field(default_factory=AgentConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def load_config_from_env() -> ExperimentConfig:
    """Construct an ``ExperimentConfig`` from environment variables.

    Reads GCP_PROJECT_ID, GCP_LOCATION, GEMINI_MODEL, EXPERIMENT_SEED,
    MAX_TURNS, and NUM_AGENTS from the process environment (populated from
    ``.env`` at runtime).  Falls back to dataclass defaults for any variable
    that is absent.
    """
    agent = AgentConfig(
        model_id=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-002"),
        gcp_project=os.getenv("GCP_PROJECT_ID", ""),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
    )
    topology = TopologyConfig(
        num_agents=int(os.getenv("NUM_AGENTS", "20")),
    )
    seed = int(os.getenv("EXPERIMENT_SEED", "42"))
    return ExperimentConfig(
        max_turns=int(os.getenv("MAX_TURNS", "10")),
        random_seed=seed,
        agent=agent,
        topology=topology,
        hallucination=HallucinationConfig(random_seed=seed),
    )
