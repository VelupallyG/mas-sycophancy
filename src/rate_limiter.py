"""Synchronous rate limiter with exponential-backoff retry for Vertex AI calls."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = ()

try:
    from google.api_core import exceptions as google_exceptions

    _RETRY_EXCEPTIONS = (
        google_exceptions.ResourceExhausted,  # 429
        google_exceptions.ServiceUnavailable,  # 503
    )
except ImportError:
    pass


class SyncRateLimiter:
    """Token-bucket rate limiter for synchronous Vertex AI calls.

    Enforces a ceiling of `requests_per_minute` calls per 60-second window.
    All callers share the same bucket — instantiate once and inject everywhere.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be >= 1")
        self._interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0

    def acquire(self) -> None:
        """Block until the next request slot is available."""
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_request_time)
            if wait > 0:
                time.sleep(wait)
            self._last_request_time = time.monotonic()


def with_retry(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorator: retry on Vertex AI rate-limit or server errors.

    Strategy: exponential backoff starting at `base_delay`, doubling each
    retry, capped at `max_delay`. Only retries on 429 and 503.

    Args:
        max_attempts: Maximum number of total attempts (first call + retries).
        base_delay: Initial sleep duration in seconds.
        max_delay: Maximum sleep duration in seconds.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except _RETRY_EXCEPTIONS as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "Vertex AI error on attempt %d/%d: %s — retrying in %.1fs",
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)

        return wrapper  # type: ignore[return-value]

    return decorator


def call_with_retry(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Call a zero-argument function with retry-on-429/503 behavior."""

    @with_retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    def _wrapped() -> Any:
        return fn()

    return _wrapped()
