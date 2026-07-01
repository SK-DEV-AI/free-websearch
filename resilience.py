"""Error resilience patterns for multi-API pipelines.

Provides:
- CircuitBreaker: stops calling a failing API until it recovers
"""

from __future__ import annotations

import time
from typing import Any


class CircuitBreaker:
    """Simple circuit breaker: opens after N failures, resets after cooldown."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._last_failure: float = 0
        self._state = "closed"

    def allow(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._last_failure >= self.cooldown_seconds:
                self._state = "half_open"
                return True
            return False
        return True

    def record_success(self):
        self._failures = 0
        self._state = "closed"

    def record_failure(self):
        self._failures += 1
        self._last_failure = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = "open"

    @property
    def is_open(self) -> bool:
        return self._state == "open"
