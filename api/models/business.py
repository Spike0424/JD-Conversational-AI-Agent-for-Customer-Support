"""Business domain models: channels, shops, accounts, and knowledge tables.

All models share the engine from api.db.  init_db() auto-creates these tables.
"""

from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint

from api.models.db import create_session


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Channel (渠道) ──────────────────────────────────────────────────

class Channel(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "channels"
    id: int | None = Field(default=None, primary_key=True)
    channel_name: str = Field(max_length=50, unique=True, nullable=False)
    description: str | None = Field(default=None, max_length=255)

    shops: list["Shop"] = Relationship(back_populates="channel")


# ── Shop (店铺) ─────────────────────────────────────────────────────

class Shop(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "shops"
    __table_args__ = (
        UniqueConstraint("channel_id", "shop_id", name="uix_shop_channel_shop_id"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    channel_id: int = Field(foreign_key="channels.id", nullable=False)
    shop_id: str = Field(max_length=100, nullable=False)
    shop_name: str = Field(max_length=100, nullable=False)
    shop_logo: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=255)

    channel: Channel = Relationship(back_populates="shops")
    accounts: list["Account"] = Relationship(back_populates="shop")
    product_knowledge: list["ProductKnowledge"] = Relationship(back_populates="shop")
    customer_service_knowledge: list["CustomerServiceKnowledge"] = Relationship(back_populates="shop")
    knowledge_meta_entries: list["KnowledgeMetaEntry"] = Relationship(back_populates="shop")
    presale_knowledge: list["PresaleKnowledge"] = Relationship(back_populates="shop")
    insale_knowledge: list["InsaleKnowledge"] = Relationship(back_populates="shop")
    aftersale_knowledge: list["AftersaleKnowledge"] = Relationship(back_populates="shop")
    transfer_target_configs: list["TransferTargetConfig"] = Relationship(back_populates="shop")


# ── Account (账号) ──────────────────────────────────────────────────

class Account(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "accounts"
    __table_args__ = (
        UniqueConstraint("shop_id", "user_id", name="uix_account_shop_user"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop_id: int = Field(foreign_key="shops.id", nullable=False)
    user_id: str = Field(max_length=100, nullable=False)
    username: str = Field(max_length=100, nullable=False)
    password: str = Field(max_length=255, nullable=False)
    cookies: str | None = Field(default=None)
    status: int | None = Field(default=None)

    shop: Shop = Relationship(back_populates="accounts")


# ── Keyword (关键词) ────────────────────────────────────────────────

class Keyword(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "keywords"
    id: int | None = Field(default=None, primary_key=True)
    keyword: str = Field(max_length=100, nullable=False)


# ── ProductKnowledge (产品知识) ─────────────────────────────────────

class ProductKnowledge(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "product_knowledge"
    __table_args__ = (
        UniqueConstraint("shop_id", "goods_id", name="uix_product_knowledge_shop_goods"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop_id: int = Field(foreign_key="shops.id", nullable=False)
    goods_id: int = Field(nullable=False)
    goods_name: str = Field(max_length=255, nullable=False)
    price: str | None = Field(default=None, max_length=50)
    price_min: int | None = Field(default=None)
    price_max: int | None = Field(default=None)
    sold_quantity: int | None = Field(default=None)
    thumb_url: str | None = Field(default=None, max_length=500)
    specifications: dict | None = Field(default=None, sa_column=Column(JSONB))
    extracted_content: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    last_extracted_at: datetime = Field(default_factory=_now)

    shop: Shop = Relationship(back_populates="product_knowledge")


# ── CustomerServiceKnowledge (客服知识) ─────────────────────────────

class CustomerServiceKnowledge(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "customer_service_knowledge"
    id: int | None = Field(default=None, primary_key=True)
    shop_id: int = Field(foreign_key="shops.id", nullable=False)
    title: str = Field(max_length=255, nullable=False)
    content: str = Field(nullable=False)
    tags: str | None = Field(default=None, max_length=255)
    enabled: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    shop: Shop = Relationship(back_populates="customer_service_knowledge")


# ── KnowledgeMetaEntry (知识元数据) ─────────────────────────────────

class KnowledgeMetaEntry(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "knowledge_meta_entries"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "source_id", "scenario", "sub_intent", "aliases",
            name="uix_meta_source_alias",
        ),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop_id: int = Field(foreign_key="shops.id", nullable=False)
    source_type: str = Field(max_length=50, nullable=False)
    source_id: int = Field(nullable=False)
    goods_id: int | None = Field(default=None)
    product_family: str | None = Field(default=None, max_length=100)
    scenario: str = Field(max_length=100, nullable=False)
    sub_intent: str | None = Field(default=None, max_length=100)
    aliases: str = Field(nullable=False)
    answer: str = Field(nullable=False)
    section_title: str | None = Field(default=None, max_length=255)
    tags: str | None = Field(default=None, max_length=255)
    enabled: bool = Field(default=True, nullable=False)
    priority: int = Field(default=0, nullable=False)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    shop: Shop = Relationship(back_populates="knowledge_meta_entries")


# ── Scene knowledge mixin ───────────────────────────────────────────

class SceneKnowledgeMixin(SQLModel):
    """Shared columns for presale/insale/aftersale knowledge tables.  Not a table."""

    shop_id: int = Field(foreign_key="shops.id", nullable=False)
    goods_id: int | None = Field(default=None)
    product_family: str | None = Field(default=None, max_length=50)
    sub_intent: str | None = Field(default=None, max_length=100)
    aliases: str = Field(nullable=False)
    answer: str = Field(nullable=False)
    section_title: str | None = Field(default=None, max_length=255)
    tags: str | None = Field(default=None, max_length=255)
    priority: int = Field(default=0, nullable=False)
    enabled: bool = Field(default=True, nullable=False)
    source_type: str | None = Field(default=None, max_length=50)
    source_id: int | None = Field(default=None)
    source_meta_id: int | None = Field(default=None)
    migrated_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ── PresaleKnowledge (售前知识) ─────────────────────────────────────

class PresaleKnowledge(SceneKnowledgeMixin, SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "presale_knowledge"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "source_id", "source_meta_id", "sub_intent", "aliases", "answer",
            name="uix_presale_dedup",
        ),
        Index("ix_presale_goods", "shop_id", "goods_id", "enabled", "priority"),
        Index("ix_presale_family", "shop_id", "product_family", "enabled"),
        Index("ix_presale_intent", "shop_id", "sub_intent", "enabled"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop: Shop = Relationship(back_populates="presale_knowledge")


# ── InsaleKnowledge (售中知识) ──────────────────────────────────────

class InsaleKnowledge(SceneKnowledgeMixin, SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "insale_knowledge"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "source_id", "source_meta_id", "sub_intent", "aliases", "answer",
            name="uix_insale_dedup",
        ),
        Index("ix_insale_goods", "shop_id", "goods_id", "enabled", "priority"),
        Index("ix_insale_family", "shop_id", "product_family", "enabled"),
        Index("ix_insale_intent", "shop_id", "sub_intent", "enabled"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop: Shop = Relationship(back_populates="insale_knowledge")


# ── AftersaleKnowledge (售后知识) ──────────────────────────────────

class AftersaleKnowledge(SceneKnowledgeMixin, SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "aftersale_knowledge"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "source_id", "source_meta_id", "sub_intent", "aliases", "answer",
            name="uix_aftersale_dedup",
        ),
        Index("ix_aftersale_goods", "shop_id", "goods_id", "enabled", "priority"),
        Index("ix_aftersale_family", "shop_id", "product_family", "enabled"),
        Index("ix_aftersale_intent", "shop_id", "sub_intent", "enabled"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop: Shop = Relationship(back_populates="aftersale_knowledge")


# ── SceneKnowledgeEmbedding (场景 embedding) ────────────────────────

class SceneKnowledgeEmbedding(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "scene_knowledge_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "scene", "knowledge_table", "knowledge_id", "content_hash",
            name="uix_scene_kb_embed_dedup",
        ),
        Index("ix_ske_shop_goods_scene", "shop_id", "goods_id", "scene"),
        Index("ix_ske_table_id", "knowledge_table", "knowledge_id"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    scene: str = Field(max_length=20, nullable=False)
    knowledge_table: str = Field(max_length=50, nullable=False)
    knowledge_id: int = Field(nullable=False)
    shop_id: int = Field(nullable=False)
    goods_id: int | None = Field(default=None)
    embedding_text: str = Field(nullable=False)
    embedding: Any = Field(sa_column=Column(Vector(384), nullable=False))
    embedding_model: str = Field(max_length=100, nullable=False)
    embedding_dim: int | None = Field(default=None)
    content_hash: str = Field(max_length=64, nullable=False)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ── TransferTargetConfig (转人工配置) ───────────────────────────────

class TransferTargetConfig(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "transfer_target_configs"
    __table_args__ = (
        UniqueConstraint("shop_id", "source_user_id", name="uix_transfer_target_shop_source"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    shop_id: int = Field(foreign_key="shops.id", nullable=False)
    source_user_id: str = Field(max_length=100, nullable=False)
    target_user_id: str = Field(max_length=100, nullable=False)
    target_username: str | None = Field(default=None, max_length=100)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    shop: Shop = Relationship(back_populates="transfer_target_configs")


# ── AftersaleChunk (售后知识切片 + 向量) ──────────────────────────

class AftersaleChunk(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "aftersale_chunks"
    __table_args__ = (
        Index("ix_aftersale_chunk_knowledge", "knowledge_id"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    knowledge_id: int = Field(foreign_key="aftersale_knowledge.id", nullable=False)
    chunk_content: str = Field(nullable=False)
    embedding: Any = Field(default=None, sa_column=Column(Vector(384), nullable=True))
    aliases: dict | None = Field(default=None, sa_column=Column(JSONB))


# ── AgentMessage (会话消息) ──────────────────────────────────────────


class AgentMessage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "agent_messages"
    __table_args__ = (
        Index("ix_agent_messages_session_timestamp", "session_id", "timestamp"),
        Index("ix_agent_messages_user_id", "user_id"),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, max_length=255, nullable=False)
    user_id: str | None = Field(default=None, max_length=64)  # 登录用户（来自 JWT）
    role: str = Field(max_length=32, nullable=False)  # system | user | assistant | tool
    content: str | None = Field(default=None)
    tool_call_id: str | None = Field(default=None, max_length=128)
    goods_name: str | None = Field(default=None, max_length=255)  # 仅首条 user 消息存
    timestamp: datetime = Field(default_factory=_now)


# ── User (登录账号) ───────────────────────────────────────────────────


class User(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email", "email", unique=True),
        {"extend_existing": True},
    )

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(max_length=255, nullable=False)
    password_hash: str = Field(max_length=128, nullable=False)
    created_at: datetime = Field(default_factory=_now)


# ── OrderModel (订单 + 物流 — 3 表合一) ─────────────────────────────


class OrderModel(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = "orders"
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    order_id: str = Field(index=True, max_length=64, nullable=False)
    user_id: str = Field(index=True, max_length=64, nullable=False)
    sku_id: int = Field(default=0)
    final_unit_price: float = Field(default=0.0)
    quantity: int = Field(default=1)
    order_date: datetime = Field(default_factory=_now)
    order_time: datetime = Field(default_factory=_now)
    type: str = Field(default="normal", max_length=32)
    sku_type: str = Field(default="general", max_length=32)
    arr_time: datetime | None = Field(default=None)

    # ── 扩展字段 ──
    shop_id: int | None = Field(default=None)
    goods_id: int | None = Field(default=None)
    order_status: str = Field(default="paid", max_length=32)
    payment_status: str = Field(default="paid", max_length=32)
    shipping_address: str = Field(default="")
    delivery_address: str = Field(default="")
    recipient_name: str = Field(default="", max_length=50)
    recipient_phone: str = Field(default="", max_length=20)
    current_location: str | None = Field(default=None, max_length=200)
    carrier: str | None = Field(default=None, max_length=50)
    tracking_number: str | None = Field(default=None, max_length=100)
    confirmed_receipt: bool = Field(default=False)


# ── Seed helper ─────────────────────────────────────────────────────

def seed_channels() -> None:
    """Insert 京东 channel if not present.  Safe to call multiple times."""
    from sqlmodel import text

    session = create_session()
    try:
        session.exec(
            text(
                "INSERT INTO channels (channel_name, description) "
                "VALUES ('京东', '京东平台') "
                "ON CONFLICT (channel_name) DO NOTHING"
            )
        )
        session.commit()
    finally:
        session.close()
