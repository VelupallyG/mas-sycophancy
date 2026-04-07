"""LLM-as-judge adapter for TRAIL categorisation calls.

This module performs deterministic (temperature=0.0) single-turn LLM calls that
return strict JSON and are parsed by src.metrics.trail.parse_trail_judge_output().
"""

from __future__ import annotations

import os

from src.rate_limiter import call_with_retry, get_shared_rate_limiter


class VertexAITrailJudge:
    """Small Vertex AI wrapper dedicated to TRAIL classification prompts."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str = "us-central1",
        requests_per_minute: int = 60,
        temperature: float = 0.0,
    ) -> None:
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        if temperature != 0.0:
            raise ValueError("TRAIL judge temperature must be 0.0")

        project = project or os.environ.get("GCP_PROJECT")
        if not project:
            raise ValueError(
                "GCP project required. Set GCP_PROJECT env var or pass project=."
            )

        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(model_id)
        self._config = GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
        )
        self._rate_limiter = get_shared_rate_limiter(
            requests_per_minute=requests_per_minute
        )

    def judge(self, prompt: str) -> str:
        """Run one deterministic TRAIL categorisation call and return raw text."""

        def _call_vertex():
            self._rate_limiter.acquire()
            return self._model.generate_content(prompt, generation_config=self._config)

        response = call_with_retry(
            _call_vertex,
            max_attempts=5,
            base_delay=1.0,
            max_delay=60.0,
        )
        return response.text.strip()
