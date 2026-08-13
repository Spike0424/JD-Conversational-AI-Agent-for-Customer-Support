"""TurnContext: 结构化底座，将原始客户 turn 解析为干净的结构化数据。

不做 embedding，不做意图路由，不做知识库变更。仅解析 + 日志记录。

Mirrors `Agent/CustomerAgent/custom/turn_context.py` from the pinduoduo project,
simplified to drop the PyQt / bridge dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ── Data classes ───────────────────────────────────────────────────────


@dataclass
class ProductCard:
    present: bool = False
    goods_id: str = ""
    goods_name: str = ""
    spec: str = ""
    price_raw: str = ""
    price_yuan: str = ""


@dataclass
class OrderCard:
    present: bool = False
    order_sn: str = ""
    order_status_text: str = ""
    main_status_code: str = ""
    logistics_status_code: str = ""
    payment_status_code: str = ""
    order_status_code: str = ""
    tracking_no: str = ""


@dataclass
class MediaInfo:
    has_image: bool = False
    has_video: bool = False
    image_urls: List[str] = field(default_factory=list)
    video_urls: List[str] = field(default_factory=list)


@dataclass
class TurnType:
    has_text: bool = False
    has_product_card: bool = False
    has_order_card: bool = False
    has_media: bool = False


@dataclass
class TurnContext:
    raw_query: str = ""
    customer_text: str = ""
    previous_customer_text: str = ""
    product_card: ProductCard = field(default_factory=ProductCard)
    order_card: OrderCard = field(default_factory=OrderCard)
    media: MediaInfo = field(default_factory=MediaInfo)
    turn_type: TurnType = field(default_factory=TurnType)
    raw_scene_hint: str = ""
    parse_warnings: List[str] = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────


def _price_fen_to_yuan(raw: str) -> str:
    """分 -> 元：1161 -> 11.61。仅转换平台商品卡常见的 4 位及以上分值。"""
    if raw.isdigit() and len(raw) >= 4:
        return f"{int(raw) / 100:.2f}"
    return raw


def _infer_media_type_from_url(url: str) -> str:
    """根据 URL 推断媒体类型（image / video / ''）。"""
    lower = url.lower()
    if re.search(r"\.(png|jpe?g|gif|webp|bmp)(\?|$)", lower):
        return "image"
    if re.search(r"\.(mp4|mov|avi|mkv|webm)(\?|$)", lower):
        return "video"
    return ""


def _normalize_media_url(url: str) -> str:
    """去除 URL 末尾的标点。"""
    return url.rstrip("。，,.;；!！？?）)】]》>'\" \t")


# ── Parsers ────────────────────────────────────────────────────────────


def parse_product_card(text: str) -> ProductCard:
    card = ProductCard()

    m = re.search(r"商品ID[：:]\s*(\d+)", text)
    if m:
        card.goods_id = m.group(1)

    m = re.search(
        r"商品[：:]\s*[【]?(.*?)[】]?(?:[，,]|规格|价格|商品ID|$)",
        text,
        re.MULTILINE,
    )
    if m:
        card.goods_name = m.group(1).strip()

    m = re.search(
        r"规格[：:]\s*(.*?)(?:[，,]|商品ID|价格|$)",
        text,
        re.MULTILINE,
    )
    if m:
        card.spec = m.group(1).strip()

    m = re.search(r"价格[：:]\s*([\d.]+)", text)
    if m:
        raw = m.group(1)
        card.price_raw = raw
        card.price_yuan = _price_fen_to_yuan(raw)

    if card.goods_id or card.goods_name:
        card.present = True

    return card


def parse_order_card(text: str) -> OrderCard:
    card = OrderCard()

    m = re.search(r"订单(?:号)?[：:]\s*(\d[\d-]+\d)", text)
    if m:
        card.order_sn = m.group(1)

    m = re.search(
        r"当前订单状态[：:]\s*(\S+?)(?:[，,]|$|\s+当前|\s+订单)",
        text,
        re.MULTILINE,
    )
    if m:
        card.order_status_text = m.group(1)

    m = re.search(r"订单主状态码[：:]\s*(\d)", text)
    if m:
        card.main_status_code = m.group(1)

    m = re.search(r"物流状态码[：:]\s*(\d)", text)
    if m:
        card.logistics_status_code = m.group(1)

    m = re.search(r"支付状态码[：:]\s*(\d)", text)
    if m:
        card.payment_status_code = m.group(1)

    m = re.search(r"订单状态码[：:]\s*(\d)", text)
    if m:
        card.order_status_code = m.group(1)

    m = re.search(r"快递单号[：:]\s*(\S+?)(?:[，,]|$|\s+物流)", text, re.MULTILINE)
    if m:
        card.tracking_no = m.group(1)

    if card.order_sn:
        card.present = True

    return card


def parse_media(text: str) -> MediaInfo:
    media = MediaInfo()

    if re.search(r"客户发送了图片|图片[：:]|(?:^|[：:\s])\[图片\](?:$|\s)|【图片】", text, re.I):
        media.has_image = True
    if re.search(r"客户发送了视频|(?:^|[：:\s])\[视频消息?\](?:$|\s)|【视频】", text, re.I):
        media.has_video = True

    for m in re.finditer(r"客户发送了图片[：:]\s*(https?://\S+)", text, re.I):
        url = _normalize_media_url(m.group(1))
        media.image_urls.append(url)
        media.has_image = True

    for m in re.finditer(r"客户发送了视频[：:]\s*(https?://\S+)", text, re.I):
        url = _normalize_media_url(m.group(1))
        media.video_urls.append(url)
        media.has_video = True

    for url in re.findall(r"https?://\S+", text, re.I):
        url = _normalize_media_url(url)
        media_type = _infer_media_type_from_url(url)
        if media_type == "image" and url not in media.image_urls:
            media.image_urls.append(url)
            media.has_image = True
        elif media_type == "video" and url not in media.video_urls:
            media.video_urls.append(url)
            media.has_video = True

    return media


def _extract_customer_text(raw_query: str) -> tuple[str, str]:
    """Extract clean customer_text and previous_customer_text from raw_query."""
    text = raw_query
    previous_customer_text = ""
    metadata_boundary = (
        r"商品卡片[：:]",
        r"订单卡片[：:]",
        r"商品ID[：:]",
        r"当前业务场景[：:]",
        r"上一条客户问题[：:]",
        r"上一条客户消息[：:]",
    )
    boundary_pattern = "|".join(f"(?:{item})" for item in metadata_boundary)

    m = re.search(r"上一条客户问题[：:]\s*(.+?)(?:\n|$)", text)
    if m:
        previous_customer_text = m.group(1).strip()

    m = re.search(
        rf"客户消息[：:]\s*(.+?)(?=\n\s*(?:{boundary_pattern})|$)",
        text,
        re.S,
    )
    if m:
        return m.group(1).strip(), previous_customer_text

    for line in text.splitlines():
        m = re.match(r"^\s*内容[：:]\s*(.*)$", line)
        if not m:
            continue
        content = m.group(1).strip()
        nested_previous = re.match(r"上一条客户(?:问题|消息)[：:]\s*(.+)$", content)
        if nested_previous:
            if not previous_customer_text:
                previous_customer_text = nested_previous.group(1).strip()
            continue
        if re.match(r"^(商品卡片|订单卡片)[：:]", content):
            return "", previous_customer_text
        if content:
            return content, previous_customer_text
        return "", previous_customer_text

    if previous_customer_text:
        return previous_customer_text, previous_customer_text

    if re.match(r"^商品[：:]", text):
        return "", previous_customer_text

    has_product_card = bool(re.search(r"商品ID[：:]", text))
    has_order_card = bool(re.search(r"订单(?:号)?[：:]\s*\d|当前订单状态|订单卡片", text))

    if has_product_card or has_order_card:
        card_start = re.search(r"(?:商品[：:]|订单(?:卡片)?[：:]|订单[：:])", text)
        if card_start:
            before = text[:card_start.start()].strip().rstrip("，,。.、 ")
            if before and len(before) > 1:
                return before, previous_customer_text
            return "", previous_customer_text

    return text.strip(), previous_customer_text


def _parse_raw_scene_hint(raw_query: str) -> str:
    m = re.search(r"当前业务场景[：:]\s*([^，,\s]+)", raw_query)
    return m.group(1) if m else ""


def _strip_metadata(customer_text: str) -> str:
    """从 customer_text 中移除元数据残留（goods_id、order_sn、卡片标记等）。"""
    text = customer_text
    text = re.sub(r"商品ID[：:]\s*\d+", "", text)
    text = re.sub(r"订单(?:号)?[：:]\s*\d[\d-]+\d", "", text)
    text = re.sub(r"(商品卡片|订单卡片)[：:].*", "", text)
    text = re.sub(r"当前业务场景[：:]\s*\S+", "", text)
    text = text.strip(" ，,。.!！~～")
    return text


# ── Public API ─────────────────────────────────────────────────────────


def parse_turn_context(raw_query: str) -> TurnContext:
    """将原始客户 turn 解析为 TurnContext 结构体。"""
    tc = TurnContext(raw_query=raw_query or "")
    warnings: List[str] = []

    if not raw_query:
        return tc

    tc.product_card = parse_product_card(raw_query)
    tc.order_card = parse_order_card(raw_query)
    tc.media = parse_media(raw_query)
    tc.raw_scene_hint = _parse_raw_scene_hint(raw_query)

    customer_text, previous_customer_text = _extract_customer_text(raw_query)
    tc.previous_customer_text = previous_customer_text

    cleaned_text = _strip_metadata(customer_text)
    tc.customer_text = cleaned_text

    if customer_text and not cleaned_text:
        warnings.append("customer_text emptied after metadata stripping")

    has_text = bool(cleaned_text.strip())
    tc.turn_type = TurnType(
        has_text=has_text,
        has_product_card=tc.product_card.present,
        has_order_card=tc.order_card.present,
        has_media=tc.media.has_image or tc.media.has_video,
    )

    if tc.product_card.present and not tc.product_card.goods_id:
        warnings.append("product_card present but goods_id missing")
    if tc.order_card.present and not tc.order_card.order_sn:
        warnings.append("order_card present but order_sn missing")
    if tc.product_card.price_raw and tc.product_card.price_raw.isdigit():
        if int(tc.product_card.price_raw) > 100:
            if not tc.product_card.price_yuan:
                warnings.append("price_fen_to_yuan conversion failed")

    tc.parse_warnings = warnings
    return tc


def turn_context_to_dict(tc: TurnContext) -> dict:
    """TurnContext → JSON-serializable dict."""
    return {
        "raw_query": tc.raw_query,
        "customer_text": tc.customer_text,
        "previous_customer_text": tc.previous_customer_text,
        "product_card": {
            "present": tc.product_card.present,
            "goods_id": tc.product_card.goods_id,
            "goods_name": tc.product_card.goods_name,
            "spec": tc.product_card.spec,
            "price_raw": tc.product_card.price_raw,
            "price_yuan": tc.product_card.price_yuan,
        },
        "order_card": {
            "present": tc.order_card.present,
            "order_sn": tc.order_card.order_sn,
            "order_status_text": tc.order_card.order_status_text,
            "main_status_code": tc.order_card.main_status_code,
            "logistics_status_code": tc.order_card.logistics_status_code,
            "payment_status_code": tc.order_card.payment_status_code,
            "order_status_code": tc.order_card.order_status_code,
            "tracking_no": tc.order_card.tracking_no,
        },
        "media": {
            "has_image": tc.media.has_image,
            "has_video": tc.media.has_video,
            "image_urls": tc.media.image_urls,
            "video_urls": tc.media.video_urls,
        },
        "turn_type": {
            "has_text": tc.turn_type.has_text,
            "has_product_card": tc.turn_type.has_product_card,
            "has_order_card": tc.turn_type.has_order_card,
            "has_media": tc.turn_type.has_media,
        },
        "raw_scene_hint": tc.raw_scene_hint,
        "parse_warnings": tc.parse_warnings,
    }