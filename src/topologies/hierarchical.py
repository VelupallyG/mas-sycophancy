"""Hierarchical (5-level stratified) MAS topology — experimental group.

Agents are distributed across five levels mirroring a corporate reporting chain:

  Level 1 — Orchestrator (1 agent, CEO/CSO)
  Level 2 — Senior Directors (2–3 agents)
  Level 3 — Senior Managers (3–5 agents)
  Level 4 — Junior Managers (3–5 agents)
  Level 5 — Entry Analysts (remaining agents)

The Concordia Game Master enforces that the final consensus must pass through
the Orchestrator.  Lower-level agents can only influence consensus by
convincing their immediate superiors — they cannot bypass the chain.

Usage::

    topology = HierarchicalTopology()
    levels = topology.build(config)
    # levels[1] = [OrchestratorAgent], levels[5] = [AnalystAgent, ...]
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.config import ExperimentConfig


class HierarchicalTopology:
    """Constructs the 5-level stratified corporate reporting chain.

    Respects the level counts defined in ``config.topology.level_counts``.
    The Orchestrator is always initialised with the hallucinated signal from
    ``config`` (if present).
    """

    def build(
        self,
        config: ExperimentConfig,
        names: dict[int, list[str]] | None = None,
    ) -> dict[int, list[BaseAgent]]:
        """Instantiate and return agents keyed by hierarchical level.

        Args:
            config: Experiment configuration (drives level counts, LLM params,
                and hallucination signal).
            names: Optional mapping from level → list of persona names.  If
                provided, each sub-list must match the corresponding count in
                ``config.topology.level_counts``.

        Returns:
            Dict mapping level integers (1–5) to lists of ``BaseAgent``
            instances.

        Raises:
            ValueError: If ``names`` lengths conflict with ``level_counts``.
        """
        raise NotImplementedError

    def get_reporting_chain(
        self,
        agent: BaseAgent,
        all_agents: dict[int, list[BaseAgent]],
    ) -> list[BaseAgent]:
        """Return the chain of superiors above ``agent`` up to Level 1.

        Used by the Game Master to route upward communications and enforce
        approval chains.

        Args:
            agent: The agent whose superiors are requested.
            all_agents: Full level → agents mapping from :meth:`build`.

        Returns:
            List of agents from the immediate superior up to the Orchestrator.
        """
        raise NotImplementedError
