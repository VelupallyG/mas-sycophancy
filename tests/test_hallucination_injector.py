"""Tests for the HallucinationInjector."""

from __future__ import annotations

import pytest

from src.hallucination.injector import HallucinationInjector, _OPPOSITE


def test_opposite_directions():
    assert _OPPOSITE["POSITIVE"] == "NEGATIVE"
    assert _OPPOSITE["NEGATIVE"] == "POSITIVE"
    assert _OPPOSITE["NEUTRAL"] == "POSITIVE"


def test_injector_renders_direction():
    injector = HallucinationInjector(version="v1")
    result = injector.render(ground_truth_direction="NEGATIVE", domain="finance")
    assert "POSITIVE" in result
    assert "{{HALLUCINATED_DIRECTION}}" not in result
    assert "{{HALLUCINATION_RATIONALE}}" not in result


def test_injector_opposite_of_positive():
    injector = HallucinationInjector(version="v1")
    result = injector.render(ground_truth_direction="POSITIVE", domain="finance")
    assert "NEGATIVE" in result


def test_injector_custom_rationale():
    injector = HallucinationInjector(version="v1")
    custom = "Our proprietary model says crash imminent."
    result = injector.render(
        ground_truth_direction="POSITIVE",
        domain="finance",
        custom_rationale=custom,
    )
    assert custom in result


def test_injector_missing_template():
    with pytest.raises(FileNotFoundError):
        HallucinationInjector(version="v999")
