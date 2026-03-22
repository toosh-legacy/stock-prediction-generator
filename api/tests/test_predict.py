import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_requires_auth(client: AsyncClient):
    r = await client.post("/v1/predict/", json={"ticker": "AAPL", "model": "xgboost", "horizon": 1})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_predict_invalid_ticker(client: AsyncClient, valid_api_key: str):
    r = await client.post(
        "/v1/predict/",
        json={"ticker": "INVALIDTICKER123XYZ", "model": "xgboost", "horizon": 1},
        headers={"X-API-Key": valid_api_key},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_api_key(client: AsyncClient):
    r = await client.post("/v1/auth/keys", json={"name": "Integration Test Key"})
    assert r.status_code == 200
    data = r.json()
    assert "api_key" in data
    assert data["api_key"].startswith("sk-")
    assert "key_id" in data


@pytest.mark.asyncio
async def test_predict_validates_horizon(client: AsyncClient, valid_api_key: str):
    r = await client.post(
        "/v1/predict/",
        json={"ticker": "AAPL", "model": "xgboost", "horizon": 100},
        headers={"X-API-Key": valid_api_key},
    )
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_predict_validates_model(client: AsyncClient, valid_api_key: str):
    r = await client.post(
        "/v1/predict/",
        json={"ticker": "AAPL", "model": "invalid_model", "horizon": 5},
        headers={"X-API-Key": valid_api_key},
    )
    assert r.status_code == 422
