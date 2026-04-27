"""VertexAILanguageModel: Concordia LanguageModel backed by Gemini on Vertex AI.

Concordia's LanguageModel interface requires two methods:
  - sample_text(prompt, ...) -> str
  - sample_choice(prompt, responses, ...) -> tuple[int, str, dict]

This adapter uses the google-genai SDK (unified Gemini SDK) with vertexai=True
backend. All agent prediction calls use JSON-constrained decoding.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Sequence
from typing import Any, override

from concordia.language_model import language_model

from src.rate_limiter import call_with_retry, get_shared_rate_limiter


class VertexAILanguageModel(language_model.LanguageModel):
    """Concordia LanguageModel wrapper around Gemini via google-genai SDK."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.2,
        requests_per_minute: int = 60,
    ) -> None:
        """Initialise and authenticate with Vertex AI via google-genai.

        Args:
            model_id: Gemini model ID (e.g., "gemini-2.5-flash").
            project: GCP project ID. Falls back to GCP_PROJECT env var.
            location: GCP region. Defaults to "us-central1".
            temperature: Sampling temperature for all agent calls (default 0.2).
            requests_per_minute: Global Vertex AI request ceiling.
        """
        from google import genai

        project = project or os.environ.get("GCP_PROJECT")
        if not project:
            raise ValueError(
                "GCP project required. Set GCP_PROJECT env var or pass project=."
            )

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model_id = model_id
        self._temperature = temperature
        self._rate_limiter = get_shared_rate_limiter(
            requests_per_minute=requests_per_minute
        )

    def _generate(self, prompt: str, *, json_mode: bool = True) -> str:
        """Rate-limit and retry Vertex calls."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        def _call() -> Any:
            self._rate_limiter.acquire()
            return self._client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            )

        response = call_with_retry(
            _call,
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
        """Call Vertex AI with JSON-constrained decoding and return response text."""
        del max_tokens, terminators, temperature, top_p, top_k, timeout, seed
        return self._generate(prompt, json_mode=True)

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, Any]]:
        """Ask the model to pick one of the provided responses."""
        del seed

        choices_str = "\n".join(f"  {i}. {r}" for i, r in enumerate(responses))
        full_prompt = (
            f"{prompt}\n\nChoose exactly one of the following options by "
            f"stating it verbatim:\n{choices_str}"
        )
        text = self._generate(full_prompt, json_mode=False)

        for i, option in enumerate(responses):
            if option.lower() in text.lower():
                return i, option, {"raw_response": text}

        return 0, responses[0], {"raw_response": text, "fallback": True}
