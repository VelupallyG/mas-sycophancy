"""Language model wrappers and embedder factories for Concordia integration.

Production usage: ``build_gemini_model()`` wraps Concordia's GeminiModel with
Vertex AI credentials from the experiment config.

Testing: ``MockLanguageModel`` returns deterministic canned responses so that
the full simulation pipeline can be exercised without API calls.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np
from concordia.language_model import language_model

from src.config import AgentConfig


# ---------------------------------------------------------------------------
# Production model
# ---------------------------------------------------------------------------

def build_gemini_model(agent_config: AgentConfig) -> language_model.LanguageModel:
    """Build a Concordia ``LanguageModel`` backed by Vertex AI Gemini.

    Uses ``concordia.contrib.language_models.google.gemini_model.GeminiModel``
    which supports Vertex AI natively when ``project`` and ``location`` are set.
    """
    from concordia.contrib.language_models.google import gemini_model

    if not agent_config.gcp_project:
        raise ValueError(
            "agent_config.gcp_project must be set (or set GCP_PROJECT_ID env var)"
        )

    return gemini_model.GeminiModel(
        model_name=agent_config.model_id,
        project=agent_config.gcp_project,
        location=agent_config.gcp_location,
    )


# ---------------------------------------------------------------------------
# Mock model for testing
# ---------------------------------------------------------------------------

class MockLanguageModel(language_model.LanguageModel):
    """Deterministic mock that cycles through canned responses.

    If no responses are supplied, generates ``"Mock response N"`` strings.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        terminators: Collection[str] = (),
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        timeout: float = 60,
        seed: int | None = None,
    ) -> str:
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
        else:
            response = f"Mock response {self._call_count}"
        self._call_count += 1
        return response

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        return 0, responses[0], {}


# ---------------------------------------------------------------------------
# Embedder factories
# ---------------------------------------------------------------------------

def make_hash_embedder(dimensions: int = 64) -> Callable[[str], np.ndarray]:
    """Deterministic hash-based sentence embedder for testing.

    Not semantically meaningful — use only for tests and prototyping where
    memory retrieval quality is not under evaluation.
    """
    def embed(text: str) -> np.ndarray:
        vector = np.zeros(dimensions)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).hexdigest()
            idx = int(digest[:8], 16) % dimensions
            sign = -1.0 if (int(digest[8:10], 16) % 2) else 1.0
            vector[idx] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector
    return embed


def make_sentence_transformer_embedder(
    model_name: str = "all-MiniLM-L6-v2",
) -> Callable[[str], np.ndarray]:
    """Production embedder using sentence-transformers.

    Requires ``pip install sentence-transformers``.
    """
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_name)

    def embed(text: str) -> np.ndarray:
        return st_model.encode(text)  # type: ignore[return-value]

    return embed
