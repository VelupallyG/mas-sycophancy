"""Tests for src.evidence.loader — file-based evidence loading and allocation."""

import json
import pytest
from pathlib import Path

from src.evidence.loader import (
    EvidenceDoc,
    EvidenceAllocation,
    load_evidence_files,
    allocate_evidence,
    format_evidence_packet,
    format_drip_packet,
)


# ---------------------------------------------------------------------------
# EvidenceDoc
# ---------------------------------------------------------------------------


class TestEvidenceDoc:
    def test_format_basic(self):
        doc = EvidenceDoc(
            id="d1",
            source_type="gdelt",
            source_name="Reuters",
            title="Oil prices rise",
            text_content="Brent crude jumped 3%.",
            document_date="2025-03-18",
        )
        out = doc.format(index=2)
        assert "[Evidence 2] Oil prices rise" in out
        assert "Source: gdelt / Reuters" in out
        assert "Date: 2025-03-18" in out
        assert "Brent crude jumped 3%." in out

    def test_format_truncates_long_text(self):
        doc = EvidenceDoc(
            id="d2",
            source_type="eia",
            source_name="EIA",
            title="Long doc",
            text_content="x" * 2000,
        )
        out = doc.format()
        # Should be truncated to 1200 chars (1197 + "...")
        assert out.endswith("...")
        assert len(out.split("\n")[-1]) == 1200

    def test_format_no_date(self):
        doc = EvidenceDoc(
            id="d3",
            source_type="context",
            source_name="manual",
            title="No date",
            text_content="Some content",
        )
        out = doc.format()
        assert "Date:" not in out


# ---------------------------------------------------------------------------
# load_evidence_files
# ---------------------------------------------------------------------------


class TestLoadEvidenceFiles:
    def test_loads_matching_files(self, tmp_path: Path):
        (tmp_path / "a.json").write_text(
            json.dumps(
                {
                    "seed_id": "test_seed",
                    "title": "Doc A",
                    "text_content": "Content A",
                    "source_type": "gdelt",
                    "source_name": "AP",
                }
            )
        )
        (tmp_path / "b.json").write_text(
            json.dumps(
                {
                    "seed_id": "other_seed",
                    "title": "Doc B",
                    "text_content": "Content B",
                }
            )
        )
        docs = load_evidence_files("test_seed", evidence_dir=tmp_path)
        assert len(docs) == 1
        assert docs[0].title == "Doc A"

    def test_skips_empty_text(self, tmp_path: Path):
        (tmp_path / "empty.json").write_text(
            json.dumps({"seed_id": "s", "title": "Empty", "text_content": ""})
        )
        assert load_evidence_files("s", evidence_dir=tmp_path) == []

    def test_skips_malformed_json(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("not json {{{")
        assert load_evidence_files("s", evidence_dir=tmp_path) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert load_evidence_files("s", evidence_dir=tmp_path / "nope") == []

    def test_searches_subdirectories(self, tmp_path: Path):
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "c.json").write_text(
            json.dumps(
                {
                    "seed_id": "s",
                    "title": "Nested",
                    "text_content": "Found it",
                    "source_type": "eia",
                    "source_name": "EIA",
                }
            )
        )
        docs = load_evidence_files("s", evidence_dir=tmp_path)
        assert len(docs) == 1
        assert docs[0].title == "Nested"


# ---------------------------------------------------------------------------
# allocate_evidence
# ---------------------------------------------------------------------------


def _make_docs(n: int) -> list[EvidenceDoc]:
    return [
        EvidenceDoc(
            id=f"doc_{i}",
            source_type="test",
            source_name="test",
            title=f"Doc {i}",
            text_content=f"Content {i}",
        )
        for i in range(n)
    ]


class TestAllocateEvidence:
    def test_basic_allocation(self):
        docs = _make_docs(30)
        agents = ["a1", "a2", "a3"]
        alloc = allocate_evidence(
            docs, agents, docs_per_agent=3, drip_turns=(5,), docs_per_drip=2
        )
        for name in agents:
            assert len(alloc.agent_evidence[name]) == 3
        assert 5 in alloc.drip_schedule
        assert len(alloc.drip_schedule[5]) == 2

    def test_deterministic_with_same_seed(self):
        docs = _make_docs(20)
        agents = ["a", "b"]
        a1 = allocate_evidence(docs, agents, rng_seed=99)
        a2 = allocate_evidence(docs, agents, rng_seed=99)
        assert a1.agent_evidence == a2.agent_evidence

    def test_different_seeds_give_different_allocations(self):
        docs = _make_docs(20)
        agents = ["a", "b"]
        a1 = allocate_evidence(docs, agents, rng_seed=1)
        a2 = allocate_evidence(docs, agents, rng_seed=2)
        # Very unlikely to be identical
        assert a1.agent_evidence != a2.agent_evidence

    def test_empty_docs(self):
        alloc = allocate_evidence([], ["a", "b"], docs_per_agent=3)
        assert alloc.agent_evidence["a"] == []
        assert alloc.agent_evidence["b"] == []

    def test_small_pool_cycles(self):
        """When fewer docs than needed, allocation still works (with replacement)."""
        docs = _make_docs(3)
        agents = ["a", "b", "c"]
        alloc = allocate_evidence(
            docs, agents, docs_per_agent=5, drip_turns=(), docs_per_drip=0
        )
        for name in agents:
            assert len(alloc.agent_evidence[name]) == 5

    def test_no_drip_when_insufficient_docs(self):
        docs = _make_docs(2)
        alloc = allocate_evidence(
            docs, ["a"], docs_per_agent=2, drip_turns=(3, 5), docs_per_drip=3
        )
        # 2 docs < 6 needed for drip, so drip is skipped
        assert alloc.drip_pool == []
        assert alloc.drip_schedule == {}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_evidence_packet(self):
        docs = _make_docs(2)
        out = format_evidence_packet(docs)
        assert "SUPPLEMENTARY EVIDENCE" in out
        assert "[Evidence 1]" in out
        assert "[Evidence 2]" in out

    def test_format_evidence_packet_empty(self):
        assert format_evidence_packet([]) == ""

    def test_format_drip_packet(self):
        docs = _make_docs(1)
        out = format_drip_packet(docs, turn=5)
        assert "NEW INTELLIGENCE UPDATE (Turn 5)" in out
        assert "[Evidence 1]" in out

    def test_format_drip_packet_empty(self):
        assert format_drip_packet([], turn=3) == ""
