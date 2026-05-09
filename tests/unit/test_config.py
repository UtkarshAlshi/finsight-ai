"""Unit tests for config module."""

import pytest

from finsight.config import Settings, get_settings


def test_default_settings_are_valid() -> None:
    s = Settings()
    assert s.app_env == "development"
    assert s.api_port == 8000
    assert s.retrieval_top_k == 20
    assert s.rerank_top_k == 5


def test_ticker_list_parsed_correctly() -> None:
    s = Settings(sec_tickers="AAPL,MSFT, GOOGL")
    assert s.ticker_list == ["AAPL", "MSFT", "GOOGL"]


def test_is_production_flag() -> None:
    s = Settings(app_env="production")
    assert s.is_production is True

    s_dev = Settings(app_env="development")
    assert s_dev.is_production is False


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9999")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings()
    assert s.api_port == 9999
    assert s.log_level == "DEBUG"
