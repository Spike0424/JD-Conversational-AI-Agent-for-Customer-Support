"""Scene classification based on platform order data (not keyword matching).

Follows the PDD pattern: async platform API call → structured order context → scene decision.
"""

import asyncio
import logging
import time
from typing import Any

from api.core.config import get_settings
from api.core.intent_router import CUSTOMER_SCENE_ALIASES

# JD connection pool (formerly in app.tools)
_jd_pool: Any = None


def _jd_pool_getconn() -> Any:
    global _jd_pool
    if _jd_pool is None:
        import psycopg2.pool
        settings = get_settings()
        _jd_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=settings.database_url)
    return _jd_pool.getconn()


def _jd_pool_putconn(conn: Any) -> None:
    global _jd_pool
    if _jd_pool is not None:
        _jd_pool.putconn(conn)


def _query_jd(sql: str, params: tuple | None = None) -> list[tuple]:
    if params is None:
        params = ()
    conn = _jd_pool_getconn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        _jd_pool_putconn(conn)

logger = logging.getLogger(__name__)

# ── Cache (session_id → scene, with TTL) ──────────────────────────

_SCENE_CACHE: dict[str, str] = {}
_SCENE_CACHE_TS: dict[str, float] = {}
_CACHE_TTL_SECONDS = 1800  # 30 minutes


def _get_cached_scene(session_id: str) -> str | None:
    """Return cached scene if not expired, else None."""
    scene = _SCENE_CACHE.get(session_id)
    if scene is None:
        return None
    ts = _SCENE_CACHE_TS.get(session_id, 0)
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _SCENE_CACHE.pop(session_id, None)
        _SCENE_CACHE_TS.pop(session_id, None)
        return None
    return scene


def _set_cached_scene(session_id: str, scene: str) -> None:
    _SCENE_CACHE[session_id] = scene
    _SCENE_CACHE_TS[session_id] = time.time()


# ── Order context ──────────────────────────────────────────────────

class OrderContext:
    """Structured result from platform order API / JD database lookup."""

    __slots__ = (
        "has_order", "order_count", "order_ids", "order_types",
        "all_signed", "scene_hint", "shipping_status",
    )

    def __init__(
        self,
        has_order: bool = False,
        order_count: int = 0,
        order_ids: list[str] | None = None,
        order_types: list[str] | None = None,
        all_signed: bool = False,
        scene_hint: str = "",
        shipping_status: str = "",
    ):
        self.has_order = has_order
        self.order_count = order_count
        self.order_ids = order_ids or []
        self.order_types = order_types or []
        self.all_signed = all_signed
        self.scene_hint = scene_hint
        self.shipping_status = shipping_status


_NO_ORDER_CONTEXT = OrderContext()

# ── Build helper ───────────────────────────────────────────────────

def _build_order_context(rows: list[tuple]) -> OrderContext:
    """Transform JD order rows into structured order context."""
    if not rows:
        return _NO_ORDER_CONTEXT

    order_ids: list[str] = []
    order_types: set[str] = set()

    for r in rows:
        order_ids.append(str(r[0]))
        order_types.add(str(r[5]) if r[5] else "unknown")

    # Single query for all delivery statuses (instead of N queries)
    all_signed = True
    if order_ids:
        placeholders = ",".join(["%s"] * len(order_ids))
        delivery_rows = _query_jd(
            f"SELECT order_id, arr_time FROM orders WHERE order_id IN ({placeholders})",
            tuple(order_ids),
        )
        signed_map = {str(d[0]): d[1] is not None for d in delivery_rows}
        all_signed = all(signed_map.get(oid, False) for oid in order_ids)
    signed_count = sum(signed_map.get(oid, False) for oid in order_ids)
    mixed_signed = 0 < signed_count < len(order_ids)

    if len(order_types) > 1 or mixed_signed:
        scene_hint = "mixed_orders"
    elif all_signed:
        scene_hint = "aftersale"
    else:
        scene_hint = "insale"

    return OrderContext(
        has_order=True,
        order_count=len(order_ids),
        order_ids=order_ids,
        order_types=sorted(order_types),
        all_signed=all_signed,
        scene_hint=scene_hint,
        shipping_status="all_signed" if all_signed else "in_transit",
    )


# ── Classifier ─────────────────────────────────────────────────────

class SceneClassifier:
    """Classify user session scene as presale / insale / aftersale / mixed.

    Caches per session_id with a 30-minute TTL.
    Exposes `last_scene_hint` after each classify() call for downstream consumers
    that want to distinguish "mixed_orders" (multiple order types) from a plain insale.
    """

    def __init__(self) -> None:
        self._last_scene_hint: str = ""

    @property
    def last_scene_hint(self) -> str:
        return self._last_scene_hint

    async def classify(
        self,
        dependencies: dict[str, Any],
        question: str,
        session_id: str,
    ) -> str:
        cached = _get_cached_scene(session_id)
        if cached is not None:
            logger.debug("Scene cache hit: session_id=%s scene=%s", session_id, cached)
            self._last_scene_hint = ""
            return cached

        ctx = await self._refresh_order_context(dependencies)
        self._last_scene_hint = ctx.scene_hint

        if ctx.scene_hint == "mixed_orders":
            scene = "insale"
        elif not ctx.has_order:
            scene = "presale"
        elif ctx.all_signed:
            scene = "aftersale"
        else:
            scene = "insale"

        _set_cached_scene(session_id, scene)
        logger.info(
            "Scene classified: session_id=%s scene=%s has_order=%s scene_hint=%s all_signed=%s",
            session_id, scene, ctx.has_order, ctx.scene_hint, ctx.all_signed,
        )
        return scene

    async def _refresh_order_context(self, dependencies: dict[str, Any]) -> OrderContext:
        customer_uid = dependencies.get("customer_uid") or dependencies.get("user_id")
        if not customer_uid:
            logger.debug("No customer_uid in dependencies, skipping order refresh")
            return _NO_ORDER_CONTEXT

        def load_context() -> OrderContext:
            rows = _query_jd(
                "SELECT order_id, order_date, final_unit_price, quantity, "
                "sku_type, type AS order_type "
                "FROM orders "
                "WHERE user_id = %s ORDER BY order_date DESC, order_time DESC LIMIT 5",
                (str(customer_uid),),
            )
            return _build_order_context(rows)

        try:
            ctx = await asyncio.to_thread(load_context)
        except Exception as exc:
            logger.warning(
                "Order context fetch failed customer_uid=%s error=%s", customer_uid, exc
            )
            ctx = _NO_ORDER_CONTEXT

        logger.info(
            "Order context: has_order=%s count=%d scene_hint=%s all_signed=%s ids=%s",
            ctx.has_order,
            ctx.order_count,
            ctx.scene_hint,
            ctx.all_signed,
            ctx.order_ids,
        )
        return ctx

    @staticmethod
    def clear_cache(session_id: str | None = None) -> None:
        if session_id:
            _SCENE_CACHE.pop(session_id, None)
            _SCENE_CACHE_TS.pop(session_id, None)
        else:
            _SCENE_CACHE.clear()
            _SCENE_CACHE_TS.clear()
