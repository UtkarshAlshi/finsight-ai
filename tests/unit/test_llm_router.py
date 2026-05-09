"""Unit tests for LLM tiered router."""

import pytest

from finsight.config import Settings
from finsight.llm.client import AnthropicClient, OpenAIClient
from finsight.llm.router import LLMRouter, TaskComplexity


@pytest.fixture
def settings_openai_only() -> Settings:
    """No Anthropic key — OpenAI only."""
    return Settings(
        openai_api_key="sk-test",
        llm_cheap_model="gpt-4o-mini",
        llm_premium_model="gpt-4o",
    )


@pytest.fixture
def settings_with_anthropic() -> Settings:
    """Both providers available."""
    return Settings(
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-test",
        llm_cheap_model="gpt-4o-mini",
        llm_premium_model="gpt-4o",
        llm_anthropic_cheap_model="claude-haiku-4-5-20251001",
        llm_anthropic_premium_model="claude-sonnet-4-6",
    )


# ── Primary (OpenAI) routing ──────────────────────────────────────────────────


def test_low_complexity_routes_to_openai_cheap(settings_openai_only: Settings) -> None:
    router = LLMRouter(settings_openai_only)
    client = router.get_client(TaskComplexity.LOW)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"


def test_high_complexity_routes_to_openai_premium(settings_openai_only: Settings) -> None:
    router = LLMRouter(settings_openai_only)
    client = router.get_client(TaskComplexity.HIGH)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"


# ── Anthropic fallback routing ────────────────────────────────────────────────


def test_use_anthropic_low_returns_anthropic_cheap(settings_with_anthropic: Settings) -> None:
    router = LLMRouter(settings_with_anthropic)
    client = router.get_client(TaskComplexity.LOW, use_anthropic=True)
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-haiku-4-5-20251001"


def test_use_anthropic_high_returns_anthropic_premium(settings_with_anthropic: Settings) -> None:
    router = LLMRouter(settings_with_anthropic)
    client = router.get_client(TaskComplexity.HIGH, use_anthropic=True)
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-4-6"


def test_use_anthropic_without_key_falls_back_to_openai(settings_openai_only: Settings) -> None:
    """When ANTHROPIC_API_KEY is absent, use_anthropic=True silently falls back to OpenAI."""
    router = LLMRouter(settings_openai_only)
    client = router.get_client(TaskComplexity.HIGH, use_anthropic=True)
    assert isinstance(client, OpenAIClient)


def test_anthropic_clients_not_created_without_key(settings_openai_only: Settings) -> None:
    router = LLMRouter(settings_openai_only)
    assert router._anthropic_cheap is None
    assert router._anthropic_premium is None


def test_anthropic_clients_created_with_key(settings_with_anthropic: Settings) -> None:
    router = LLMRouter(settings_with_anthropic)
    assert isinstance(router._anthropic_cheap, AnthropicClient)
    assert isinstance(router._anthropic_premium, AnthropicClient)
