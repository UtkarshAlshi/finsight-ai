"""Unit tests for LLM tiered router."""

import pytest

from finsight.config import Settings
from finsight.llm.client import AnthropicClient, OpenAIClient
from finsight.llm.router import LLMRouter, TaskComplexity


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-test",
        llm_cheap_model="claude-haiku-4-5-20251001",
        llm_premium_model="claude-sonnet-4-6",
        llm_openai_model="gpt-4o-mini",
    )


def test_low_complexity_routes_to_cheap_model(settings: Settings) -> None:
    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.LOW)
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-haiku-4-5-20251001"


def test_high_complexity_routes_to_premium_model(settings: Settings) -> None:
    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.HIGH)
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-4-6"


def test_openai_flag_overrides_complexity(settings: Settings) -> None:
    router = LLMRouter(settings)
    client = router.get_client(TaskComplexity.HIGH, use_openai=True)
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"
