"""Unit tests for ML service endpoints (no real model loading)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Sentiment service
# ---------------------------------------------------------------------------


def test_sentiment_health() -> None:
    from finsight.ml.sentiment_service import create_sentiment_app

    client = TestClient(create_sentiment_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_sentiment_predict_graceful_degradation() -> None:
    """When model fails to load, neutral scores are returned."""
    from finsight.ml.sentiment_service import create_sentiment_app

    with patch("finsight.ml.sentiment_service._load_model", return_value=None):
        client = TestClient(create_sentiment_app())
        resp = client.post("/predict", json={"texts": ["Revenue grew 20% YoY."]})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["label"] == "neutral"


def test_sentiment_predict_with_mock_model() -> None:
    """Verify pipeline output is mapped to SentimentScore correctly."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        [{"label": "positive", "score": 0.91}, {"label": "negative", "score": 0.05}]
    ]

    from finsight.ml.sentiment_service import create_sentiment_app

    with patch("finsight.ml.sentiment_service._load_model", return_value=mock_pipeline):
        client = TestClient(create_sentiment_app())
        resp = client.post("/predict", json={"texts": ["Strong earnings beat expectations."]})

    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert preds[0]["label"] == "positive"
    assert preds[0]["score"] == pytest.approx(0.91, abs=1e-3)


# ---------------------------------------------------------------------------
# Forecast service
# ---------------------------------------------------------------------------


def test_forecast_health() -> None:
    from finsight.ml.forecast_service import create_forecast_app

    client = TestClient(create_forecast_app())
    resp = client.get("/health")
    assert resp.status_code == 200


def test_forecast_unknown_ticker_returns_404() -> None:
    from finsight.ml.forecast_service import create_forecast_app

    client = TestClient(create_forecast_app())
    resp = client.post("/forecast", json={"ticker": "ZZZZ", "periods": 4})
    assert resp.status_code == 404


def test_forecast_known_ticker_with_mock_prophet() -> None:
    from datetime import date

    mock_rows = [
        {
            "ds": MagicMock(date=lambda: date(2025, 9, 30)),
            "yhat": 420.0,
            "yhat_lower": 400.0,
            "yhat_upper": 440.0,
        },
        {
            "ds": MagicMock(date=lambda: date(2026, 9, 30)),
            "yhat": 450.0,
            "yhat_lower": 430.0,
            "yhat_upper": 470.0,
        },
    ]

    with patch("finsight.ml.forecast_service._run_prophet", return_value=mock_rows):
        from finsight.ml.forecast_service import create_forecast_app

        client = TestClient(create_forecast_app())
        resp = client.post("/forecast", json={"ticker": "AAPL", "periods": 2})

    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert len(data["forecast"]) == 2


# ---------------------------------------------------------------------------
# Anomaly service
# ---------------------------------------------------------------------------


def test_anomaly_health() -> None:
    from finsight.ml.anomaly_service import create_anomaly_app

    client = TestClient(create_anomaly_app())
    resp = client.get("/health")
    assert resp.status_code == 200


def test_anomaly_detect_returns_results() -> None:
    """With sklearn available, detect should label observations."""
    features = [[float(i), float(i * 2)] for i in range(10)]
    # Inject an outlier
    features.append([999.0, 999.0])

    from finsight.ml.anomaly_service import create_anomaly_app

    client = TestClient(create_anomaly_app())
    resp = client.post("/detect", json={"ticker": "AAPL", "features": features})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "AAPL"
    assert len(data["results"]) == len(features)
    assert "anomaly_count" in data


def test_anomaly_detect_graceful_degradation() -> None:
    """When sklearn fails, no anomalies are flagged."""
    with patch("finsight.ml.anomaly_service._run_isolation_forest") as mock_iforest:
        mock_iforest.return_value = [
            {"index": 0, "score": 0.0, "is_anomaly": False},
            {"index": 1, "score": 0.0, "is_anomaly": False},
        ]
        from finsight.ml.anomaly_service import create_anomaly_app

        features = [[1.0, 2.0]] * 10
        client = TestClient(create_anomaly_app())
        resp = client.post("/detect", json={"ticker": "MSFT", "features": features})

    assert resp.status_code == 200
    assert resp.json()["anomaly_count"] == 0
