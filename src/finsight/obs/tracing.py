"""OpenTelemetry tracer + Langfuse integration setup."""

from __future__ import annotations

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = structlog.get_logger()

_tracer_provider: TracerProvider | None = None


def setup_tracing(
    service_name: str,
    otlp_endpoint: str,
    *,
    langfuse_public_key: str = "",
    langfuse_secret_key: str = "",
    langfuse_host: str = "",
    use_console: bool = False,
) -> TracerProvider:
    """Initialise OTel TracerProvider and optionally Langfuse client.

    Should be called once at application startup before any spans are created.
    """
    global _tracer_provider  # noqa: PLW0603

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if use_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    if langfuse_public_key and langfuse_secret_key:
        _setup_langfuse(langfuse_public_key, langfuse_secret_key, langfuse_host)

    logger.info(
        "tracing.initialised",
        service=service_name,
        otlp_endpoint=otlp_endpoint,
        langfuse_enabled=bool(langfuse_public_key),
    )
    return provider


def _setup_langfuse(public_key: str, secret_key: str, host: str) -> None:
    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host or "https://cloud.langfuse.com",
        )
        client.auth_check()
        logger.info("langfuse.connected", host=host)
    except Exception as exc:
        logger.warning("langfuse.init_failed", error=str(exc))


def get_tracer(name: str = "finsight") -> trace.Tracer:
    return trace.get_tracer(name)
