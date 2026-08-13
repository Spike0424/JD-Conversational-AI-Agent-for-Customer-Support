"""Scene-based routing: classify user question as presale / insale / aftersale."""

import re
from typing import Optional

CUSTOMER_SCENE_LABELS = {
    "presale": "售前",
    "insale": "售中",
    "aftersale": "售后",
}

CUSTOMER_SCENE_ALIASES: dict[str, tuple[str, ...]] = {
    "presale": (
        "售前", "售前咨询", "购买前", "下单前", "拍前", "买前", "购买咨询",
        "pre_sale", "presale", "pre-sale",
    ),
    "insale": (
        "售中", "售中-待发货", "售中-物流中", "待发货", "已发货待收货",
        "物流中", "催发货", "加急发货", "改地址", "修改地址", "拦截",
        "insale", "in_sale", "in-sale",
    ),
    "aftersale": (
        "售后", "售后倾向", "已签收", "签收后", "收到后", "质量问题",
        "退换货","退货", "退货退款", "退款补偿", "协商", "aftersale", "after_sale",
        "after-sale",
    ),
}

_NON_WORD_RE = re.compile(r"[^\w]")

# Direct lookup: normalized text → canonical scene key
_DIRECT_SCENE_MAP: dict[str, str] = {
    "presale": "presale",
    "pre_sale": "presale",
    "pre-sale": "presale",
    "insale": "insale",
    "in_sale": "insale",
    "in-sale": "insale",
    "aftersale": "aftersale",
    "after_sale": "aftersale",
    "after-sale": "aftersale",
}


def _normalize(text: str) -> str:
    return _NON_WORD_RE.sub("", text.lower().strip())


def normalize_customer_scene(scene: Optional[str]) -> str:
    """归一化场景字符串 → presale / insale / aftersale，无法识别返回空。"""
    clean = _normalize(scene or "")
    if not clean:
        return ""

    # Direct mapping (includes Chinese labels via CUSTOMER_SCENE_LABELS)
    direct: dict[str, str] = {}
    for label, key in CUSTOMER_SCENE_LABELS.items():
        direct[_normalize(label)] = key
    for k, v in _DIRECT_SCENE_MAP.items():
        direct[_normalize(k)] = v

    if clean in direct:
        return direct[clean]

    # Alias fallback
    for scene_key, aliases in CUSTOMER_SCENE_ALIASES.items():
        for alias in aliases:
            if _normalize(alias) in clean:
                return scene_key

    return ""


def detect_scene(question: str) -> str:
    """Return 'presale', 'insale', or 'aftersale' based on keyword matching."""
    text = question.lower()
    for scene, aliases in CUSTOMER_SCENE_ALIASES.items():
        if any(alias.lower() in text for alias in aliases):
            return scene
    return "presale"  # 默认归类为售前
