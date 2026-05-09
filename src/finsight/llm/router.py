"""Tiered LLM router: cheap model for low-complexity, premium for synthesis.

OpenAI is the primary provider. Anthropic is an optional fallback, used only
when ANTHROPIC_API_KEY is set and the caller explicitly requests it.
"""

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
    """Routes LLM calls to cheap or premium model based on TaskComplexity.

    Primary provider is always OpenAI. Anthropic clients are created only when
    ANTHROPIC_API_KEY is set; requests for Anthropic without a key fall back to
    the OpenAI client with a warning.
    """

    def __init__(self, settings: Settings) -> None:
        openai_key = settings.openai_api_key.get_secret_value()

        self._cheap: _AnyClient = OpenAIClient(
            api_key=openai_key,
            model=settings.llm_cheap_model,
        )
        self._premium: _AnyClient = OpenAIClient(
            api_key=openai_key,
            model=settings.llm_premium_model,
        )

        # Anthropic clients — instantiated only when key is present
        self._anthropic_cheap: _AnyClient | None = None
        self._anthropic_premium: _AnyClient | None = None
        if settings.anthropic_api_key is not None:
            anthropic_key = settings.anthropic_api_key.get_secret_value()
            self._anthropic_cheap = AnthropicClient(
                api_key=anthropic_key,
                model=settings.llm_anthropic_cheap_model,
            )
            self._anthropic_premium = AnthropicClient(
                api_key=anthropic_key,
                model=settings.llm_anthropic_premium_model,
            )

    def get_client(
        self,
        complexity: TaskComplexity,
        *,
        use_anthropic: bool = False,
    ) -> _AnyClient:
        """Return the appropriate LLM client for the given task complexity.

        When use_anthropic=True and ANTHROPIC_API_KEY is set, returns the
        Anthropic client. Otherwise returns the primary OpenAI client and logs
        a warning if Anthropic was requested but unavailable.
        """
        if use_anthropic:
            anthropic = (
                self._anthropic_cheap
                if complexity == TaskComplexity.LOW
                else self._anthropic_premium
            )
            if anthropic is not None:
                client: _AnyClient = anthropic
            else:
                logger.warning(
                    "llm.anthropic_unavailable",
                    reason="ANTHROPIC_API_KEY not set — falling back to OpenAI",
                )
                client = self._cheap if complexity == TaskComplexity.LOW else self._premium
        else:
            client = self._cheap if complexity == TaskComplexity.LOW else self._premium

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
