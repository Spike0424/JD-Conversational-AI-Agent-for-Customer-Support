from pydantic import BaseModel, Field

from api.models.context import Context


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    question: str = Field(
        default="",
        description=(
            "User question. May be empty for context-driven turns (goods_card only, "
            "image / video attachment, system messages) where the platform provides "
            "the actual signal via `context`."
        ),
    )
    context: Context | None = Field(
        default=None,
        description=(
            "Optional platform context (shop_id, goods_id, media, context_type). "
            "When the agent is embedded in a product page, the frontend fills this in. "
            "If absent, the request is treated as a plain text turn."
        ),
    )


class Citation(BaseModel):
    source: str
    snippet: str
    score: float
    file_name: str | None = None
    page_number: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace_id: str | None = None
    intent: str = "general_qa"
    context_type: str = "text"
    citations: list[Citation] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    need_handoff: bool = False
    product_cards: list[dict] | None = Field(
        default=None,
        description="LLM 发送的商品卡片列表。前端检测到有值就渲染卡片。",
    )
    metadata: dict | None = Field(
        default=None,
        description="附加元数据（如 scene / scene_label），不直接展示给客户。",
    )


class HandoffRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    reason: str = Field(..., min_length=1, description="Reason to transfer to human support.")
    priority: str = Field(default="normal", description="Ticket priority: low/normal/high.")


class HandoffResponse(BaseModel):
    session_id: str
    ticket_id: str
    queue: str
    status: str = "queued"


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str | None = None


class ShopInfo(BaseModel):
    """Public shop info for the chat frontend."""
    id: int
    shop_name: str
    shop_logo: str | None = None
    description: str | None = None


class ProductInfo(BaseModel):
    """Public product info for the chat frontend product-search results."""
    goods_id: int
    goods_name: str
    price: str | None = None
    thumb_url: str | None = None