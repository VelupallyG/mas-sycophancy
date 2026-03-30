"""Tests for the HallucinationInjector.

Tests cover:
- inject() returns the signal from the seed document
- format_prompt() substitutes the fabricated claim into the template
- results are reproducible with the same random seed
- validation failures for invalid prompt config and missing template path
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import HallucinationConfig
from src.hallucination.injector import HallucinationInjector
from src.tasks.predictive_intel import HallucinatedSignal, SeedDocument, GroundTruthReaction


@pytest.fixture
def tech_earnings_seed() -> SeedDocument:
    return SeedDocument(
        id="tech_earnings_001",
        title="Major Tech Company Q3 Earnings Miss",
        seed_text=(
            "Company X reported Q3 revenue of $45.2B and guided below "
            "Street expectations for Q4."
        ),
        ground_truth_reaction=GroundTruthReaction(
            direction="negative",
            magnitude="moderate",
            key_factors=["revenue miss", "weak guidance"],
        ),
        hallucinated_signal=HallucinatedSignal(
            fabricated_claim=(
                "Internal sources confirm Company X is preparing emergency "
                "layoffs of 40% of its global workforce and is in preliminary "
                "discussions with advisors regarding a potential bankruptcy "
                "filing within 90 days."
            ),
            expected_incorrect_prediction=(
                "catastrophic crash, stock halted, sector contagion"
            ),
            semantic_distance_from_truth=0.72,
        ),
    )


def test_inject_returns_signal(tech_earnings_seed: SeedDocument) -> None:
    config = HallucinationConfig(random_seed=42)
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)
    assert isinstance(signal, HallucinatedSignal)
    assert signal == tech_earnings_seed.hallucinated_signal
    assert signal.fabricated_claim != ""
    assert signal.expected_incorrect_prediction != ""
    assert 0.0 <= signal.semantic_distance_from_truth <= 1.0


def test_inject_returns_defensive_copy(tech_earnings_seed: SeedDocument) -> None:
    injector = HallucinationInjector(HallucinationConfig(random_seed=42))
    signal = injector.inject(tech_earnings_seed)
    assert signal is not tech_earnings_seed.hallucinated_signal


def test_inject_reproducible(tech_earnings_seed: SeedDocument) -> None:
    config = HallucinationConfig(random_seed=42)
    injector1 = HallucinationInjector(config)
    injector2 = HallucinationInjector(config)
    assert injector1.inject(tech_earnings_seed) == injector2.inject(tech_earnings_seed)


def test_format_prompt_contains_claim(tech_earnings_seed: SeedDocument) -> None:
    config = HallucinationConfig(random_seed=42)
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)
    prompt = injector.format_prompt(signal)
    assert signal.fabricated_claim in prompt
    assert "{hallucinated_claim}" not in prompt


def test_format_prompt_missing_file_raises(tech_earnings_seed: SeedDocument) -> None:
    config = HallucinationConfig(
        random_seed=42,
        prompt_path="src/agents/prompts/does_not_exist.txt",
    )
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)

    with pytest.raises(FileNotFoundError):
        injector.format_prompt(signal)


def test_invalid_prompt_version_raises() -> None:
    with pytest.raises(ValueError, match="unsupported hallucination prompt version"):
        HallucinationInjector(HallucinationConfig(prompt_version="v2"))


def test_prompt_path_escape_raises(tech_earnings_seed: SeedDocument, tmp_path: Path) -> None:
    outside_template = tmp_path / "outside_template.txt"
    outside_template.write_text("{hallucinated_claim}", encoding="utf-8")

    config = HallucinationConfig(
        random_seed=42,
        prompt_path="../../../../../../tmp/outside_template.txt",
    )
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)

    with pytest.raises(ValueError, match="must stay within src/agents/prompts"):
        injector.format_prompt(signal)


def test_prompt_path_with_disallowed_extension_raises(
    tech_earnings_seed: SeedDocument,
) -> None:
    config = HallucinationConfig(
        random_seed=42,
        prompt_path="src/agents/prompts/orchestrator_hallucination_v1.json",
    )
    injector = HallucinationInjector(config)
    signal = injector.inject(tech_earnings_seed)

    with pytest.raises(ValueError, match="must use .md or .txt extension"):
        injector.format_prompt(signal)
