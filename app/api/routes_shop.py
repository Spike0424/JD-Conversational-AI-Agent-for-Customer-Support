"""Public shop/product endpoints for the chat frontend (no auth, rate-limited)."""

import logging

from fastapi import APIRouter, Query
from sqlalchemy import or_

from app.business_models import ProductKnowledge, Shop
from app.db import create_session
from app.schemas import ProductInfo, ShopInfo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["shop"])


@router.get("/v1/shops", response_model=list[ShopInfo])
def list_shops() -> list[ShopInfo]:
    """List all shops (public, for the chat form shop-selector)."""
    session = create_session()
    try:
        rows = session.query(Shop).all()
        return [
            ShopInfo(id=r.id, shop_name=r.shop_name, shop_logo=r.shop_logo, description=r.description)
            for r in rows
        ]
    finally:
        session.close()


@router.get("/v1/products", response_model=list[ProductInfo])
def search_products(
    shop_id: int = Query(..., description="店铺 ID（必填）"),
    q: str = Query("", description="搜索关键词（商品名模糊匹配）"),
    limit: int = Query(20, ge=1, le=50, description="返回条数上限"),
) -> list[ProductInfo]:
    """Search products by shop + keyword (public, for the chat form product-picker)."""
    session = create_session()
    try:
        stmt = session.query(ProductKnowledge).filter(
            ProductKnowledge.shop_id == shop_id,
        )
        if q.strip():
            stmt = stmt.filter(
                or_(
                    ProductKnowledge.goods_name.ilike(f"%{q}%"),
                    ProductKnowledge.goods_id == q if q.isdigit() else False,
                )
            )
        rows = stmt.limit(limit).all()
        return [
            ProductInfo(
                goods_id=r.goods_id,
                goods_name=r.goods_name,
                price=r.price,
                thumb_url=r.thumb_url,
            )
            for r in rows
        ]
    finally:
        session.close()
