"""Error resilience patterns for multi-API pipelines.

Provides:
- CircuitBreaker: stops calling a failing API until it recovers
- safe_call: wraps async calls with timeout + retry + fallback
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitBreaker:
    """Simple circuit breaker: opens after N failures, resets after cooldown.

    Usage:
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        if cb.allow():
            try:
                result = await do_api_call()
                cb.record_success()
            except Exception:
                cb.record_failure()
                raise
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._last_failure: float = 0
        self._state = "closed"  # closed = normal, open = blocked, half_open = testing

    def allow(self) -> bool:
        """Check if we should attempt the call."""
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._last_failure >= self.cooldown_seconds:
                self._state = "half_open"
                return True
            return False
        # half_open: allow one attempt
        return True

    def record_success(self):
        """Record a successful call."""
        self._failures = 0
        self._state = "closed"

    def record_failure(self):
        """Record a failed call."""
        self._failures += 1
        self._last_failure = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = "open"

    @property
    def is_open(self) -> bool:
        return self._state == "open"


async def safe_call(
    fn: Callable[..., Any],
    *args,
    fallback: Any = None,
    timeout: float = 30,
    retries: int = 1,
    **kwargs,
) -> Any:
    """Call an async function with timeout and retry, returning fallback on failure.

    Args:
        fn: Async function to call
        *args: Positional args for fn
        fallback: Value to return on any failure
        timeout: Max seconds per attempt
        retries: Number of retry attempts (0 = no retry)
        **kwargs: Keyword args for fn

    Returns:
        Result of fn(*args, **kwargs) or fallback on failure
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            last_err = f"timeout after {timeout}s"
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    return fallback
