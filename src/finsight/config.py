"""Central configuration via pydantic-settings — all values read from env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))

    # ── LLM ───────────────────────────────────────────────────
    # OpenAI is primary; Anthropic is optional fallback.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    groq_api_key: SecretStr | None = None
    llm_cheap_model: str = "gpt-4o-mini"
    llm_premium_model: str = "gpt-4o"
    llm_anthropic_cheap_model: str = "claude-haiku-4-5-20251001"
    llm_anthropic_premium_model: str = "claude-sonnet-4-6"

    # ── Database ──────────────────────────────────────────────
    postgres_dsn: str = "postgresql+asyncpg://finsight:finsight@localhost:5432/finsight"
    postgres_sync_dsn: str = "postgresql://finsight:finsight@localhost:5432/finsight"

    # ── Vector DB ─────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "finsight_docs"

    # ── Redis ─────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.92

    # ── Kafka ─────────────────────────────────────────────────
    kafka_brokers: str = "localhost:9092"
    kafka_news_topic: str = "financial_news"
    kafka_group_id: str = "finsight_consumer"

    # ── Observability ─────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))
    langfuse_host: str = "http://localhost:3000"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "finsight-api"

    # ── SEC EDGAR ─────────────────────────────────────────────
    sec_edgar_user_agent: str = "FinSightAI contact@example.com"
    sec_edgar_base_url: str = "https://data.sec.gov"
    sec_tickers: str = "AAPL,MSFT,GOOGL,META,AMZN,NVDA,TSLA,JPM,V,WMT"

    # ── RAG ───────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Rate Limiting ─────────────────────────────────────────
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        """Treat an empty or whitespace-only ANTHROPIC_API_KEY as absent (None)."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _empty_groq_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def ticker_list(self) -> list[str]:
        return [t.strip() for t in self.sec_tickers.split(",") if t.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
