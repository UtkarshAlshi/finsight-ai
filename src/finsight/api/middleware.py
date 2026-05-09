"""FastAPI middleware: request-id, structured logging, error mapping."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from finsight.errors import FinSightError
from finsight.obs.metrics import http_request_duration_seconds, http_requests_total

logger = structlog.get_logger()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects a unique request-id into structlog context for every request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        response.headers["X-Request-ID"] = request_id

        http_requests_total.labels(
            method=request.method,
            path=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            path=request.url.path,
        ).observe(duration)

        logger.info(
            "http.request",
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response


def add_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(FinSightError)
    async def finsight_error_handler(request: Request, exc: FinSightError) -> JSONResponse:
        logger.warning(
            "error.handled",
            error_code=exc.error_code,
            message=exc.message,
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )
