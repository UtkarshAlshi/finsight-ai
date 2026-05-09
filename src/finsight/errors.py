"""Typed exception hierarchy for FinSight AI."""

from http import HTTPStatus


class FinSightError(Exception):
    """Base exception. Subclasses map to HTTP status codes."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ── Client errors ─────────────────────────────────────────────────────────────


class ValidationError(FinSightError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "validation_error"


class NotFoundError(FinSightError):
    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "not_found"


class RateLimitError(FinSightError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS.value
    error_code = "rate_limit_exceeded"


class AuthError(FinSightError):
    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "unauthorized"


class GuardrailError(FinSightError):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "guardrail_triggered"


# ── Upstream / integration errors ─────────────────────────────────────────────


class LLMError(FinSightError):
    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "llm_error"


class LLMRateLimitError(LLMError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS.value
    error_code = "llm_rate_limit"


class EmbeddingError(FinSightError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "embedding_error"


class RetrievalError(FinSightError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "retrieval_error"


class IngestionError(FinSightError):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "ingestion_error"


class CircuitOpenError(FinSightError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE.value
    error_code = "circuit_open"


class MLServiceError(FinSightError):
    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "ml_service_error"
