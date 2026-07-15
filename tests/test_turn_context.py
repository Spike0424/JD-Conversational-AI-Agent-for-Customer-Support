"""Tests for the TurnContext parser (regex-based extraction from raw_query)."""

from app.orchestrator.turn_context import (
    parse_turn_context,
    turn_context_to_dict,
)


# ── Pure text ────────────────────────────────────────────────────────


def test_pure_text() -> None:
    tc = parse_turn_context("iPhone 17 Pro 参数")
    assert tc.customer_text == "iPhone 17 Pro 参数"
    assert not tc.turn_type.has_product_card
    assert not tc.turn_type.has_order_card
    assert not tc.turn_type.has_media


# ── Text + product card ──────────────────────────────────────────────


def test_text_with_product_card() -> None:
    raw = (
        "客户消息：在吗？\n"
        "商品ID：12345\n"
        "商品：【iPhone 17 Pro】\n"
        "规格：256G 黑色\n"
        "价格：9999"
    )
    tc = parse_turn_context(raw)
    assert tc.customer_text == "在吗？"
    assert tc.turn_type.has_text
    assert tc.turn_type.has_product_card
    assert tc.product_card.goods_id == "12345"
    assert tc.product_card.goods_name == "iPhone 17 Pro"
    assert tc.product_card.spec == "256G 黑色"
    assert tc.product_card.price_raw == "9999"
    assert tc.product_card.price_yuan == "99.99"


# ── Product card only (no text) ──────────────────────────────────────


def test_product_card_only_no_text() -> None:
    raw = "商品ID：2001\n商品：【iPhone 17】\n价格：5999"
    tc = parse_turn_context(raw)
    assert tc.customer_text == ""
    assert tc.turn_type.has_product_card
    assert not tc.turn_type.has_text
    assert tc.product_card.goods_id == "2001"
    assert tc.product_card.goods_name == "iPhone 17"


# ── Text + order card ────────────────────────────────────────────────


def test_text_with_order_card() -> None:
    raw = (
        "客户消息：我的快递到哪了\n"
        "订单卡片：\n"
        "订单号：202601010001\n"
        "当前订单状态：已发货\n"
        "快递单号：SF1234567890"
    )
    tc = parse_turn_context(raw)
    assert tc.customer_text == "我的快递到哪了"
    assert tc.turn_type.has_order_card
    assert tc.order_card.order_sn == "202601010001"
    assert tc.order_card.order_status_text == "已发货"
    assert tc.order_card.tracking_no == "SF1234567890"


# ── Image ────────────────────────────────────────────────────────────


def test_image_attachment() -> None:
    raw = "客户发送了图片：https://example.com/x.jpg"
    tc = parse_turn_context(raw)
    assert tc.media.has_image
    assert tc.turn_type.has_media
    assert "https://example.com/x.jpg" in tc.media.image_urls


# ── Video ────────────────────────────────────────────────────────────


def test_video_attachment() -> None:
    raw = "[视频消息] https://example.com/clip.mp4"
    tc = parse_turn_context(raw)
    assert tc.media.has_video
    assert "https://example.com/clip.mp4" in tc.media.video_urls


# ── System message ───────────────────────────────────────────────────


def test_system_status_message() -> None:
    raw = "当前业务场景：aftersale"
    tc = parse_turn_context(raw)
    assert tc.raw_scene_hint == "aftersale"
    assert tc.customer_text == ""


# ── Previous customer message ────────────────────────────────────────


def test_previous_customer_text() -> None:
    raw = "上一条客户问题：什么时候发货\n客户消息：顺丰到哪了"
    tc = parse_turn_context(raw)
    assert tc.previous_customer_text == "什么时候发货"
    assert tc.customer_text == "顺丰到哪了"


# ── to_dict round-trip ───────────────────────────────────────────────


def test_turn_context_to_dict_shape() -> None:
    raw = "商品ID：42\n商品：【Test】\n价格：1234"
    tc = parse_turn_context(raw)
    d = turn_context_to_dict(tc)
    assert d["product_card"]["goods_id"] == "42"
    assert d["product_card"]["goods_name"] == "Test"
    assert d["product_card"]["price_yuan"] == "12.34"
    assert d["turn_type"]["has_product_card"] is True