"""Tests for the /v1/admin/* endpoints (TurnContext dry-run, validate, scene cache, context-types)."""

import pytest
from fastapi.testclient import TestClient

from api.controllers.admin import router as admin_router
from api.core import scene_classifier
from api.core.scene_classifier import SceneClassifier
from fastapi import FastAPI


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(admin_router)
    return TestClient(app)


# ── dry-run ──────────────────────────────────────────────────────────


def test_dry_run_pure_text(client: TestClient) -> None:
    r = client.post("/v1/admin/turn-context/dry-run", json={"raw_query": "iPhone 17 Pro 参数"})
    assert r.status_code == 200
    data = r.json()
    assert data["customer_text"] == "iPhone 17 Pro 参数"
    assert data["turn_type"]["has_text"] is True


def test_dry_run_with_product_card(client: TestClient) -> None:
    raw = "客户消息：在吗？\n商品ID：12345\n商品：【iPhone 17 Pro】\n价格：9999"
    r = client.post("/v1/admin/turn-context/dry-run", json={"raw_query": raw})
    assert r.status_code == 200
    data = r.json()
    assert data["customer_text"] == "在吗？"
    assert data["product_card"]["goods_id"] == "12345"
    assert data["product_card"]["goods_name"] == "iPhone 17 Pro"


# ── validate ─────────────────────────────────────────────────────────


def test_validate_goods_card_missing_fields(client: TestClient) -> None:
    r = client.post(
        "/v1/admin/context/validate",
        json={"context": {"type": "goods_card", "kwargs": {}}},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert "kwargs.goods_id" in data["missing"]
    assert "kwargs.goods_name" in data["missing"]


def test_validate_goods_card_with_fields(client: TestClient) -> None:
    r = client.post(
        "/v1/admin/context/validate",
        json={
            "context": {
                "type": "goods_card",
                "kwargs": {"goods_id": 2001, "goods_name": "iPhone 17"},
            }
        },
    )
    data = r.json()
    assert data["ok"] is True
    assert data["missing"] == []


def test_validate_image_missing_media_url(client: TestClient) -> None:
    r = client.post(
        "/v1/admin/context/validate",
        json={"context": {"type": "image", "kwargs": {}}},
    )
    data = r.json()
    assert data["requires_human"] is True
    assert "kwargs.media_url" in data["missing"]


# ── scene cache clear ───────────────────────────────────────────────


def test_scene_cache_clear_all(client: TestClient) -> None:
    # warm the cache
    scene_classifier._SCENE_CACHE["s1"] = "aftersale"
    scene_classifier._SCENE_CACHE_TS["s1"] = 1.0
    r = client.post("/v1/admin/scene-cache/clear", json={})
    assert r.status_code == 200
    assert scene_classifier._SCENE_CACHE == {}


def test_scene_cache_clear_specific_session(client: TestClient) -> None:
    scene_classifier._SCENE_CACHE["s1"] = "aftersale"
    scene_classifier._SCENE_CACHE["s2"] = "presale"
    r = client.post("/v1/admin/scene-cache/clear", json={"session_id": "s1"})
    assert r.status_code == 200
    assert "s1" not in scene_classifier._SCENE_CACHE
    assert "s2" in scene_classifier._SCENE_CACHE


# ── context-types catalog ───────────────────────────────────────────


def test_context_types_catalog(client: TestClient) -> None:
    r = client.get("/v1/admin/context-types")
    assert r.status_code == 200
    data = r.json()
    assert len(data["context_types"]) >= 15
    assert any(c["value"] == "text" for c in data["context_types"])
    assert any(c["value"] == "image" and c["requires_human"] for c in data["context_types"])
    assert any(c["value"] == "withdraw" and c["is_silent"] for c in data["context_types"])
    assert any(c["value"] == "jingdong" for c in data["channel_types"])


# ── auth (bearer token) ──────────────────────────────────────────────


@pytest.fixture
def authed_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-test-token")
    app = FastAPI()
    app.include_router(admin_router)
    return TestClient(app, headers={"Authorization": "Bearer secret-test-token"})


def test_auth_required_when_token_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-test-token")
    app = FastAPI()
    app.include_router(admin_router)
    client = TestClient(app)

    r = client.get("/v1/admin/context-types")
    assert r.status_code == 401

    r = client.get(
        "/v1/admin/context-types",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r.status_code == 401

    r = client.get(
        "/v1/admin/context-types",
        headers={"Authorization": "secret-test-token"},  # no Bearer prefix
    )
    assert r.status_code == 401


def test_auth_disabled_when_token_unset(client: TestClient) -> None:
    """Without ADMIN_API_TOKEN env var, requests go through (dev convenience)."""
    r = client.get("/v1/admin/context-types")
    assert r.status_code == 200


def test_auth_correct_token_passes(authed_client: TestClient) -> None:
    r = authed_client.get("/v1/admin/context-types")
    assert r.status_code == 200
    r = authed_client.post(
        "/v1/admin/turn-context/dry-run", json={"raw_query": "在吗？"}
    )
    assert r.status_code == 200