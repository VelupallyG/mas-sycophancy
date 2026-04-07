"""VertexAILanguageModel: Concordia LanguageModel backed by Gemini on Vertex AI.

Concordia's LanguageModel interface requires two methods:
  - sample_text(prompt, ...) -> str
  - sample_choice(prompt, responses, ...) -> tuple[int, str, dict]

This adapter forwards sample_text() calls to Vertex AI with JSON-constrained
decoding enabled (response_mime_type="application/json"). This is the only
Vertex AI adapter for this project; it is used by all agent prefabs.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Sequence
from typing import Any, override

from concordia.language_model import language_model

from src.rate_limiter import call_with_retry, get_shared_rate_limiter


class VertexAILanguageModel(language_model.LanguageModel):
    """Concordia LanguageModel wrapper around Gemini via Vertex AI."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.2,
        requests_per_minute: int = 60,
    ) -> None:
        """Initialise and authenticate with Vertex AI.

        Args:
            model_id: Vertex AI model ID (e.g., "gemini-2.5-flash").
            project: GCP project ID. Falls back to GCP_PROJECT env var.
            location: GCP region. Defaults to "us-central1".
            temperature: Sampling temperature for all agent calls (default 0.2).
                Use 0.0 only for deterministic TRAIL classification calls.
            requests_per_minute: Global Vertex AI request ceiling.
        """
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        project = project or os.environ.get("GCP_PROJECT")
        if not project:
            raise ValueError(
                "GCP project required. Set GCP_PROJECT env var or pass project=."
            )

        vertexai.init(project=project, location=location)
        self._vertex_model = GenerativeModel(model_id)
        self._temperature = temperature
        self._rate_limiter = get_shared_rate_limiter(
            requests_per_minute=requests_per_minute
        )

        # JSON-constrained decoding for all agent prediction calls.
        self._json_config = GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        )
        # Plain config for sample_choice (free-text, not JSON).
        self._plain_config = GenerationConfig(temperature=temperature)

    def _generate_content_with_guardrails(
        self, prompt: str, generation_config: Any
    ) -> str:
        """Rate-limit and retry Vertex calls to avoid quota flakiness."""

        def _call_vertex() -> Any:
            self._rate_limiter.acquire()
            return self._vertex_model.generate_content(
                prompt,
                generation_config=generation_config,
            )

        response = call_with_retry(
            _call_vertex,
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
        )
        return response.text.strip()

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        """Call Vertex AI with JSON-constrained decoding and return response text.

        Note: Concordia passes its own temperature/top_p/top_k defaults, but
        we use the adapter's configured temperature for consistency across all
        turns. The response_mime_type="application/json" config is always applied.
        """
        # Unused args are intentionally ignored — Concordia's interface requires
        # them, but JSON-constrained decoding handles output format.
        del max_tokens, terminators, temperature, top_p, top_k, timeout, seed

        return self._generate_content_with_guardrails(
            prompt=prompt,
            generation_config=self._json_config,
        )

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, Any]]:
        """Ask the model to pick one of the provided responses.

        Constructs a free-text prompt and scans the response for which option
        was chosen. Falls back to the first option if none can be identified.
        """
        del seed

        choices_str = "\n".join(f"  {i}. {r}" for i, r in enumerate(responses))
        full_prompt = (
            f"{prompt}\n\nChoose exactly one of the following options by "
            f"stating it verbatim:\n{choices_str}"
        )
        text = self._generate_content_with_guardrails(
            prompt=full_prompt,
            generation_config=self._plain_config,
        )

        for i, option in enumerate(responses):
            if option.lower() in text.lower():
                return i, option, {"raw_response": text}

        # Fallback: return first option.
        return 0, responses[0], {"raw_response": text, "fallback": True}
