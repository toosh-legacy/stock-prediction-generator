import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_backtest_requires_auth(client: AsyncClient):
    r = await client.post(
        "/v1/backtest/",
        json={
            "ticker": "AAPL",
            "model": "xgboost",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
        },
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_backtest_invalid_ticker(client: AsyncClient, valid_api_key: str):
    r = await client.post(
        "/v1/backtest/",
        json={
            "ticker": "INVALIDTICKER123XYZ",
            "model": "xgboost",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
        },
        headers={"X-API-Key": valid_api_key},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_backtest_validates_capital(client: AsyncClient, valid_api_key: str):
    r = await client.post(
        "/v1/backtest/",
        json={
            "ticker": "AAPL",
            "model": "xgboost",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 10,  # below minimum of 1000
        },
        headers={"X-API-Key": valid_api_key},
    )
    assert r.status_code == 422
