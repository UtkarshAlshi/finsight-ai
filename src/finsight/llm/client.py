"""LLMClient protocol + Anthropic and OpenAI implementations."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Protocol, cast, runtime_checkable

import structlog
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from finsight.errors import LLMError
from finsight.obs.metrics import (
    llm_cost_usd_total,
    llm_latency_seconds,
    llm_requests_total,
    llm_tokens_total,
)

logger = structlog.get_logger()

# ── Cost table (USD per 1M tokens) ────────────────────────────────────────────
_INPUT_COST: dict[str, float] = {
    "claude-haiku-4-5-20251001": 0.80,
    "claude-sonnet-4-6": 3.00,
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
}
_OUTPUT_COST: dict[str, float] = {
    "claude-haiku-4-5-20251001": 4.00,
    "claude-sonnet-4-6": 15.00,
    "gpt-4o-mini": 0.60,
    "gpt-4o": 10.00,
}


@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol all LLM clients must satisfy."""

    provider: str
    model: str

    async def generate(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str: ...

    def stream(
        self, prompt: str, *, system: str = "", max_tokens: int = 1024
    ) -> AsyncIterator[str]: ...

    async def embed(self, text: str) -> list[float]: ...


class AnthropicClient:
    """Anthropic Claude implementation of LLMClient."""

    provider = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        start = time.perf_counter()
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=messages,
                        system=system or None,  # type: ignore[arg-type]
                    )
                    content = response.content[0]
                    text = content.text if hasattr(content, "text") else str(content)
                    self._record_metrics(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                        time.perf_counter() - start,
                    )
                    return text
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        return ""  # unreachable, satisfies mypy

    async def stream(
        self, prompt: str, *, system: str = "", max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                system=system or None,  # type: ignore[arg-type]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise LLMError(f"Anthropic stream error: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Use sentence-transformers for embeddings, not the Anthropic API")

    def _record_metrics(self, input_tokens: int, output_tokens: int, latency: float) -> None:
        llm_requests_total.labels(
            provider=self.provider, model=self.model, task_complexity="unknown"
        ).inc()
        llm_tokens_total.labels(provider=self.provider, model=self.model, token_type="input").inc(
            input_tokens
        )
        llm_tokens_total.labels(provider=self.provider, model=self.model, token_type="output").inc(
            output_tokens
        )
        input_cost = input_tokens * _INPUT_COST.get(self.model, 0) / 1_000_000
        output_cost = output_tokens * _OUTPUT_COST.get(self.model, 0) / 1_000_000
        llm_cost_usd_total.labels(provider=self.provider, model=self.model).inc(
            input_cost + output_cost
        )
        llm_latency_seconds.labels(provider=self.provider, model=self.model).observe(latency)


class OpenAIClient:
    """OpenAI implementation of LLMClient."""

    provider = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        start = time.perf_counter()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(Exception),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=max_tokens,
                    )
                    text = response.choices[0].message.content or ""
                    usage = response.usage
                    if usage:
                        self._record_metrics(
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            time.perf_counter() - start,
                        )
                    return text
        except Exception as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc
        return ""

    async def stream(
        self, prompt: str, *, system: str = "", max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            raw = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                stream=True,
            )
            stream = cast(AsyncStream[ChatCompletionChunk], raw)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise LLMError(f"OpenAI stream error: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Use sentence-transformers for embeddings")

    def _record_metrics(self, input_tokens: int, output_tokens: int, latency: float) -> None:
        llm_requests_total.labels(
            provider=self.provider, model=self.model, task_complexity="unknown"
        ).inc()
        llm_tokens_total.labels(provider=self.provider, model=self.model, token_type="input").inc(
            input_tokens
        )
        llm_tokens_total.labels(provider=self.provider, model=self.model, token_type="output").inc(
            output_tokens
        )
        input_cost = input_tokens * _INPUT_COST.get(self.model, 0) / 1_000_000
        output_cost = output_tokens * _OUTPUT_COST.get(self.model, 0) / 1_000_000
        llm_cost_usd_total.labels(provider=self.provider, model=self.model).inc(
            input_cost + output_cost
        )
        llm_latency_seconds.labels(provider=self.provider, model=self.model).observe(latency)


class GroqClient(OpenAIClient):
    """Groq implementation — OpenAI-compatible API, different base URL."""

    provider = "groq"

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
