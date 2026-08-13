"""Platform context models: ChannelType, ContextType, ChannelKwargs, Context.

Mirrors `bridge/context.py` from the pinduoduo-customer-agent project, but
adapted for a web (FastAPI) flow instead of a PyQt desktop app. Use this to
describe what the platform sends on each customer turn — message kind, channel,
and channel-specific kwargs (product card, order card, media, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChannelType(str, Enum):
    """渠道类型枚举 — 平台标识，方便多渠道扩展。"""

    JINGDONG = "jingdong"
    PINDUODUO = "pinduoduo"
    TAOBAO = "taobao"
    DOUYIN = "douyin"
    KUAISHOU = "kuaishou"


class ContextType(str, Enum):
    """上下文类型枚举 — 平台消息的归一化分类。

    平台发来的原始消息经过归一化后会被映射到以下类型之一，Agent 根据类型
    决定是调用 LLM、转人工追问还是直接静默。
    """

    TEXT = "text"  # 普通文本问题
    IMAGE = "image"  # 图片消息
    VIDEO = "video"  # 视频消息
    GOODS_CARD = "goods_card"  # 纯商品卡无文字
    GOODS_INQUIRY = "goods_inquiry"  # 商品规格咨询
    GOODS_SPEC = "goods_spec"  # 商品规格详情
    ORDER_INFO = "order_info"  # 订单卡（含订单号 + 状态）
    SYSTEM_STATUS = "system_status"  # 系统状态消息
    MALL_SYSTEM_MSG = "mall_system_msg"  # 商城系统消息
    SYSTEM_HINT = "system_hint"  # 系统提示
    SYSTEM_BIZ = "system_biz"  # 系统业务事件
    MALL_CS = "mall_cs"  # 商城客服自动消息
    WITHDRAW = "withdraw"  # 消息撤回
    AUTH = "auth"  # 认证事件
    TRANSFER = "transfer"  # 转接事件

    @property
    def is_silent(self) -> bool:
        """True if this type should not trigger an LLM call (auto / preset)."""
        return self in {
            ContextType.WITHDRAW,
            ContextType.AUTH,
            ContextType.MALL_SYSTEM_MSG,
            ContextType.SYSTEM_BIZ,
            ContextType.SYSTEM_STATUS,
        }

    @property
    def requires_human(self) -> bool:
        """True if this type should be handed off to a human agent immediately."""
        return self in {ContextType.IMAGE, ContextType.VIDEO}


class ChannelKwargs(BaseModel):
    """渠道无关 kwargs：商品卡 / 订单 / 平台元信息。

    前端嵌入商品页时，把读到的商品卡 / 店铺 / 客户身份等信息填到这里。
    """

    shop_id: str | None = Field(default=None, description="店铺 ID（外部业务 ID）")
    shop_name: str | None = Field(default=None, description="店铺名")
    user_id: str | None = Field(default=None, description="客服账号 ID")
    from_uid: str | None = Field(default=None, description="客户 UID（用于 scene 分类）")
    recipient_uid: str | None = Field(default=None, description="收件人 UID（同 from_uid）")

    goods_id: int | None = Field(default=None, description="当前商品 ID")
    goods_name: str | None = Field(default=None, description="当前商品名")
    spec: str | None = Field(default=None, description="商品规格")
    price_raw: str | None = Field(default=None, description="原始价格字符串")
    price_yuan: str | None = Field(default=None, description="换算后的元价格")

    order_sn: str | None = Field(default=None, description="订单号")
    order_status_text: str | None = Field(default=None, description="订单状态文本")

    media_url: str | None = Field(default=None, description="图片 / 视频 URL")
    media_type: str | None = Field(default=None, description="media 类型: image / video")

    raw_data: dict[str, Any] | None = Field(default=None, description="原始平台 payload（调试用）")


class Context(BaseModel):
    """一个客户 turn 的归一化上下文。"""

    type: ContextType = Field(default=ContextType.TEXT, description="消息类型")
    channel_type: ChannelType | None = Field(default=None, description="渠道")
    content: str | None = Field(default=None, description="原始文本内容（包含商品卡元数据）")
    kwargs: ChannelKwargs = Field(default_factory=ChannelKwargs, description="渠道无关参数")