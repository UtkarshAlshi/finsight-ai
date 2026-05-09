"""Tests for health and metrics endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from finsight.api.main import app


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


async def test_ping_returns_pong(client: AsyncClient) -> None:
    response = await client.get("/ping")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "pong"


async def test_metrics_endpoint_returns_prometheus_format(client: AsyncClient) -> None:
    response = await client.get("/metrics", follow_redirects=True)
    assert response.status_code == 200
    assert b"finsight_build_info" in response.content


async def test_request_id_header_present(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "x-request-id" in response.headers
    # UUID format: 8-4-4-4-12 hex chars
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 36
