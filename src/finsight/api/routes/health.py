"""Health and ping endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from finsight.obs.tracing import get_tracer

logger = structlog.get_logger()
tracer = get_tracer()
router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


class PingResponse(BaseModel):
    message: str
    traced: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ping", response_model=PingResponse)
async def ping() -> PingResponse:
    with tracer.start_as_current_span("ping") as span:
        span.set_attribute("custom.test", "true")
        logger.info("ping.received")
        return PingResponse(message="pong", traced=True)
