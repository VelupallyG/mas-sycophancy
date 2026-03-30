"""Tests for whistleblower policy and runner wiring."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.whistleblower import rank_from_variant
from src.agents.whistleblower_prefab import WhistleblowerPrefab
from src.config import ExperimentConfig
from experiments import run_whistleblower


def _valid_hierarchical_levels() -> dict[int, list]:
    return {
        1: [SimpleNamespace(params={"name": "Orchestrator_01"})],
        2: [SimpleNamespace(params={"name": "Director_01"}), SimpleNamespace(params={"name": "Director_02"})],
        3: [SimpleNamespace(params={"name": "SeniorManager_01"})],
        4: [SimpleNamespace(params={"name": "Manager_01"})],
        5: [SimpleNamespace(params={"name": "Analyst_01"})],
    }


@pytest.mark.parametrize(
    ("variant", "expected_rank"),
    [("low", 5), ("high", 2), ("LOW", 5), ("HIGH", 2)],
)
def test_rank_from_variant_maps_expected_levels(variant: str, expected_rank: int) -> None:
    assert rank_from_variant(variant) == expected_rank


def test_rank_from_variant_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        rank_from_variant("mid")


def test_inject_whistleblower_low_rank_replaces_level_5_agent() -> None:
    levels = _valid_hierarchical_levels()
    updated = run_whistleblower.inject_whistleblower(levels, rank=5, name="WB-L5")
    assert isinstance(updated[5][0], WhistleblowerPrefab)
    assert updated[5][0].params["name"] == "WB-L5"
    assert updated[5][0].params["rank"] == 5


def test_inject_whistleblower_high_rank_replaces_level_2_agent() -> None:
    levels = _valid_hierarchical_levels()
    updated = run_whistleblower.inject_whistleblower(levels, rank=2, name="WB-L2")
    assert isinstance(updated[2][0], WhistleblowerPrefab)
    assert updated[2][0].params["name"] == "WB-L2"
    assert updated[2][0].params["rank"] == 2


def test_inject_whistleblower_rejects_unsupported_rank() -> None:
    with pytest.raises(ValueError):
        run_whistleblower.inject_whistleblower(_valid_hierarchical_levels(), rank=3)


def test_run_whistleblower_main_writes_ranked_result(monkeypatch, tmp_path: Path) -> None:
    fixed_args = SimpleNamespace(
        rank="low",
        seed_doc="tech_earnings",
        turns=3,
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(run_whistleblower, "parse_args", lambda: fixed_args)

    config = ExperimentConfig()
    monkeypatch.setattr(run_whistleblower, "load_config_from_env", lambda: config)

    fake_seed = SimpleNamespace(
        id="seed_001",
        ground_truth_reaction=SimpleNamespace(direction="negative", magnitude="moderate"),
    )

    class _FakeTask:
        def load_seed(self, name: str):
            assert name == "tech_earnings"
            return fake_seed

    monkeypatch.setattr(run_whistleblower, "PredictiveIntelTask", _FakeTask)

    class _FakeSignal:
        fabricated_claim = "Fabricated claim"

    class _FakeInjector:
        def __init__(self, _cfg):
            pass

        def inject(self, seed):
            assert seed is fake_seed
            return _FakeSignal()

    monkeypatch.setattr(run_whistleblower, "HallucinationInjector", _FakeInjector)

    class _FakeTopology:
        def build(self, cfg, hallucinated_premise=""):
            assert hallucinated_premise == "Fabricated claim"
            assert cfg is config
            return _valid_hierarchical_levels()

    monkeypatch.setattr(run_whistleblower, "HierarchicalTopology", _FakeTopology)

    captured: dict[str, object] = {}

    class _FakeSimulation:
        def __init__(self, _gm_cfg):
            pass

        def run(self, topology_agents, task):
            captured["topology_agents"] = topology_agents
            captured["task"] = task
            return SimpleNamespace(
                experiment_id="whistle_test_001",
                accuracy=0.7,
                consensus_prediction="negative moderate",
                trace_path=str(tmp_path / "trace.json"),
                agent_turn_records=[{"turn": 1, "agent_name": "WB-L5"}],
                metadata={"topology": "hierarchical"},
            )

    monkeypatch.setattr(run_whistleblower, "Simulation", _FakeSimulation)

    run_whistleblower.main()

    assert "topology_agents" in captured
    injected = captured["topology_agents"]
    assert isinstance(injected[5][0], WhistleblowerPrefab)
    assert injected[5][0].params["rank"] == 5

    result_files = list(tmp_path.glob("*_result.json"))
    assert result_files
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["condition"] == "whistleblower"
    assert payload["whistleblower_rank_variant"] == "low"
    assert payload["whistleblower_rank"] == 5
