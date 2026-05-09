"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app

from finsight.api.middleware import add_middleware
from finsight.api.routes import admin, health, query
from finsight.config import get_settings
from finsight.obs.logging import configure_logging
from finsight.obs.tracing import setup_tracing

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)
    setup_tracing(
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key.get_secret_value(),
        langfuse_host=settings.langfuse_host,
        use_console=not settings.is_production,
    )
    logger.info("app.started", env=settings.app_env)
    yield
    logger.info("app.stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="FinSight AI",
        description="Multi-agent financial intelligence platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    add_middleware(app)
    FastAPIInstrumentor.instrument_app(app)

    app.include_router(health.router, tags=["health"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()
