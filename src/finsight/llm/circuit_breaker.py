"""Circuit breaker for LLM and ML service calls.

States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (probe) → CLOSED

Uses a simple in-process counter; for multi-process deployments, back with Redis.
"""

from __future__ import annotations

import time
from enum import Enum, auto

import structlog

from finsight.errors import CircuitOpenError

logger = structlog.get_logger()


class _State(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

        self._state = _State.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def is_open(self) -> bool:
        if self._state == _State.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                self._state = _State.HALF_OPEN
                logger.info("circuit_breaker.half_open", name=self._name)
                return False
            return True
        return False

    def record_success(self) -> None:
        if self._state in (_State.HALF_OPEN, _State.OPEN):
            logger.info("circuit_breaker.closed", name=self._name)
        self._state = _State.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self._failure_threshold:
            if self._state != _State.OPEN:
                logger.warning(
                    "circuit_breaker.open",
                    name=self._name,
                    failures=self._failure_count,
                )
            self._state = _State.OPEN

    def check(self) -> None:
        """Raise CircuitOpenError if circuit is open."""
        if self.is_open:
            raise CircuitOpenError(f"Circuit breaker '{self._name}' is OPEN — service unavailable")


# Module-level breakers for key services
_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name)
    return _breakers[name]
