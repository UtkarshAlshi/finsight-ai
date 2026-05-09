"""Tiered LLM router: cheap model for low-complexity, premium for synthesis."""

from __future__ import annotations

import enum

import structlog

from finsight.config import Settings
from finsight.llm.client import AnthropicClient, LLMClient, OpenAIClient
from finsight.obs.metrics import llm_requests_total

logger = structlog.get_logger()


class TaskComplexity(enum.Enum):
    """Routing tier for LLM calls."""

    LOW = "low"  # classification, extraction, lookup
    HIGH = "high"  # synthesis, multi-document analysis, generation


# Union of concrete clients satisfying the LLMClient protocol
_AnyClient = AnthropicClient | OpenAIClient


class LLMRouter:
    """Routes LLM calls to cheap or premium model based on TaskComplexity."""

    def __init__(self, settings: Settings) -> None:
        anthropic_key = settings.anthropic_api_key.get_secret_value()
        openai_key = settings.openai_api_key.get_secret_value()

        self._cheap: _AnyClient = AnthropicClient(
            api_key=anthropic_key,
            model=settings.llm_cheap_model,
        )
        self._premium: _AnyClient = AnthropicClient(
            api_key=anthropic_key,
            model=settings.llm_premium_model,
        )
        self._openai_fallback: _AnyClient = OpenAIClient(
            api_key=openai_key,
            model=settings.llm_openai_model,
        )

    def get_client(
        self,
        complexity: TaskComplexity,
        *,
        use_openai: bool = False,
    ) -> _AnyClient:
        """Return the appropriate LLM client for the given task complexity."""
        if use_openai:
            client: _AnyClient = self._openai_fallback
        elif complexity == TaskComplexity.LOW:
            client = self._cheap
        else:
            client = self._premium

        llm_requests_total.labels(
            provider=client.provider,
            model=client.model,
            task_complexity=complexity.value,
        ).inc()

        logger.debug(
            "llm.routed",
            complexity=complexity.value,
            provider=client.provider,
            model=client.model,
        )
        return client


def is_llm_client(obj: object) -> bool:
    """Runtime check that an object satisfies the LLMClient protocol."""
    return isinstance(obj, LLMClient)
