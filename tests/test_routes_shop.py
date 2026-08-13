"""Tests for public shop/product endpoints (chat frontend)."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def shop_client(monkeypatch: pytest.MonkeyPatch):
    """Client with mocked DB session returning canned shops/products."""
    from api.main import app

    mock_session = MagicMock()
    mock_shop = MagicMock(id=1, shop_name="测试店铺", shop_logo=None, description="测试描述")
    mock_product = MagicMock(goods_id=57430876, goods_name="iPhone 17 Pro Max", price="9999", thumb_url="http://x/y.jpg")

    # list_shops uses session.query(Shop).all()
    mock_session.query.return_value.all.return_value = [mock_shop]
    # search_products uses session.query(ProductKnowledge).filter().filter().limit().all()
    mock_session.query.return_value.filter.return_value.filter.return_value.limit.return_value.all.return_value = [mock_product]

    monkeypatch.setattr("api.controllers.shop.create_session", lambda: mock_session)
    return TestClient(app)


def test_list_shops(shop_client: TestClient) -> None:
    r = shop_client.get("/v1/shops")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["shop_name"] == "测试店铺"


def test_search_products(shop_client: TestClient) -> None:
    r = shop_client.get("/v1/products", params={"shop_id": 1, "q": "iPhone"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["goods_id"] == 57430876
    assert "iPhone" in data[0]["goods_name"]


def test_search_products_missing_shop_id(shop_client: TestClient) -> None:
    r = shop_client.get("/v1/products", params={"q": "iPhone"})
    assert r.status_code == 422  # FastAPI returns 422 for missing required query param
