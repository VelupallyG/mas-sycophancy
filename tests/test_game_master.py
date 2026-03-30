"""Tests for the Game Master and Simulation runner.

Implement in Session 6.  Tests cover:
- Simulation.run raises NotImplementedError until implemented (guards stub)
- GameMasterConfig default values are correct
- SimulationResult fields are accessible
- Simulation respects max_turns cap
- Trace file is written to output directory after run
"""
import pytest

from src.config import ExperimentConfig
from src.game_master.simulation import GameMasterConfig, Simulation, SimulationResult


@pytest.mark.skip(reason="Not implemented yet — Session 6")
def test_game_master_config_defaults() -> None:
    config = GameMasterConfig()
    assert config.enforce_approval_chain is False
    assert config.verbose is False


@pytest.mark.skip(reason="Not implemented yet — Session 6")
def test_simulation_result_fields() -> None:
    result = SimulationResult(
        experiment_id="test_001",
        consensus_prediction="negative moderate",
        accuracy=0.8,
        trace_path="data/test_001.json",
    )
    assert result.experiment_id == "test_001"
    assert result.accuracy == pytest.approx(0.8)
    assert result.agent_turn_records == []


@pytest.mark.skip(reason="Not implemented yet — Session 6")
def test_simulation_run_not_implemented() -> None:
    """Verify the stub raises NotImplementedError (expected at scaffold stage)."""
    config = GameMasterConfig(experiment=ExperimentConfig())
    sim = Simulation(config)
    with pytest.raises(NotImplementedError):
        sim.run(topology_agents=[], task=None)
