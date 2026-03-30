"""Tests for the HallucinationInjector.

Implement in Session 4.  Tests cover:
- inject() returns the signal from the seed document
- format_prompt() substitutes the fabricated claim into the template
- Results are reproducible with the same random seed
- Returned HallucinatedSignal has correct type and non-empty fabricated_claim
"""
import pytest

from src.config import HallucinationConfig
from src.hallucination.injector import HallucinationInjector
from src.tasks.predictive_intel import PredictiveIntelTask


@pytest.fixture
def tech_earnings_seed():
    task = PredictiveIntelTask()
    return task.load_seed("tech_earnings")


@pytest.mark.skip(reason="Not implemented yet — Session 4")
def test_inject_returns_signal(tech_earnings_seed) -> None:
    config = HallucinationConfig(random_seed=42)
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)
    assert signal.fabricated_claim != ""
    assert 0.0 <= signal.semantic_distance_from_truth <= 1.0


@pytest.mark.skip(reason="Not implemented yet — Session 4")
def test_inject_reproducible(tech_earnings_seed) -> None:
    config = HallucinationConfig(random_seed=42)
    injector1 = HallucinationInjector(config)
    injector2 = HallucinationInjector(config)
    assert injector1.inject(tech_earnings_seed).fabricated_claim == \
           injector2.inject(tech_earnings_seed).fabricated_claim


@pytest.mark.skip(reason="Not implemented yet — Session 4")
def test_format_prompt_contains_claim(tech_earnings_seed) -> None:
    config = HallucinationConfig(random_seed=42)
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)
    prompt = injector.format_prompt(signal)
    assert signal.fabricated_claim in prompt
