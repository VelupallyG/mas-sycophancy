"""Flat (peer-to-peer) MAS topology — control group.

In the flat topology all agents share the same rank (Level 5) and communicate
via round-robin turn-taking enforced by the Concordia Game Master.  No agent
has override authority over any other.  This is the baseline against which the
hierarchical topology's hallucination amplification is measured.

Usage::

    topology = FlatTopology()
    agents = topology.build(config)
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.config import ExperimentConfig


class FlatTopology:
    """Constructs a flat peer-to-peer agent collective.

    All ``num_agents`` agents are instantiated as ``AnalystAgent`` instances
    with identical rank components.  Persona names are auto-generated as
    ``Agent_01``, ``Agent_02``, … unless an explicit name list is provided.
    """

    def build(
        self,
        config: ExperimentConfig,
        names: list[str] | None = None,
    ) -> list[BaseAgent]:
        """Instantiate and return the flat agent list.

        Args:
            config: Experiment configuration driving agent count and LLM params.
            names: Optional explicit list of persona names.  If provided it
                must have exactly ``config.topology.num_agents`` elements.

        Returns:
            List of ``BaseAgent`` instances, all at Level 5.

        Raises:
            ValueError: If ``names`` length does not match ``num_agents``.
        """
        raise NotImplementedError
