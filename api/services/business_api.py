import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from api.core.config import Settings


class BusinessAPIError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class BusinessResult:
    status: str
    code: str
    data: dict[str, Any] | None = None
    message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status, "code": self.code}
        if self.data is not None:
            payload["data"] = self.data
        if self.message:
            payload["message"] = self.message
        return payload


class BusinessService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._timeout = max(1, settings.business_api_timeout_seconds)
        self._retries = max(0, settings.business_api_retries)
        self._headers = {"X-API-Key": settings.business_api_key} if settings.business_api_key else {}

    async def _request(self, base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not base_url.strip():
            raise BusinessAPIError("BACKEND_NOT_CONFIGURED", "Business backend base URL is not configured.")

        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
                    response = await client.request(method=method, url=url, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise BusinessAPIError("INVALID_BACKEND_PAYLOAD", "Backend returned non-object JSON.")
                return data
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self._retries:
                    raise BusinessAPIError("BACKEND_TIMEOUT", "Business backend request timed out.") from exc
                await asyncio.sleep(0.2 * (attempt + 1))
            except httpx.HTTPStatusError as exc:
                raise BusinessAPIError("BACKEND_HTTP_ERROR", f"HTTP {exc.response.status_code} from backend.") from exc
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self._retries:
                    raise BusinessAPIError("BACKEND_UNAVAILABLE", "Business backend request failed.") from exc
                await asyncio.sleep(0.2 * (attempt + 1))
        raise BusinessAPIError("BACKEND_UNAVAILABLE", str(last_error) if last_error else "Unknown backend error.")

    async def search_products(self, category: str, budget_min: int, budget_max: int) -> BusinessResult:
        if self._settings.crm_base_url.strip():
            data = await self._request(
                base_url=self._settings.crm_base_url,
                method="POST",
                path="/products/search",
                payload={"category": category, "budget_min": budget_min, "budget_max": budget_max},
            )
            return BusinessResult(status="ok", code="PRODUCT_SEARCH_OK", data=data)

        mock_items = [
            {"sku": "PHONE-001", "name": "Nebula X1 12+256", "category": "手机", "price": 3299},
            {"sku": "LAPTOP-002", "name": "FalconBook Pro 14", "category": "笔记本", "price": 6999},
            {"sku": "PAD-003", "name": "Aurora Pad 11", "category": "平板", "price": 2499},
            {"sku": "PHONE-004", "name": "Nebula X1 Ultra 16+512", "category": "手机", "price": 4799},
        ]
        items = [i for i in mock_items if category in i["category"] and budget_min <= i["price"] <= budget_max]
        return BusinessResult(
            status="ok",
            code="PRODUCT_SEARCH_OK_MOCK",
            data={"items": items},
            message="CRM_BASE_URL not configured; using mock data.",
        )

    async def get_order_status(self, order_id: str) -> BusinessResult:
        if self._settings.oms_base_url.strip():
            data = await self._request(
                base_url=self._settings.oms_base_url,
                method="GET",
                path=f"/orders/{order_id}",
            )
            return BusinessResult(status="ok", code="ORDER_STATUS_OK", data=data)

        mocked = {
            "ORDER-1001": {"status": "已发货", "logistics": "顺丰 SF123456789CN", "eta": "2026-05-06"},
            "ORDER-1002": {"status": "已签收", "logistics": "京东物流 JD987654321", "eta": "2026-05-03"},
        }
        order = mocked.get(order_id.upper())
        if not order:
            return BusinessResult(status="error", code="ORDER_NOT_FOUND", message=f"Order {order_id} not found.")
        return BusinessResult(
            status="ok",
            code="ORDER_STATUS_OK_MOCK",
            data=order,
            message="OMS_BASE_URL not configured; using mock data.",
        )

    async def check_warranty(self, sn_or_imei: str) -> BusinessResult:
        token = sn_or_imei.strip().upper()
        if not token:
            return BusinessResult(status="error", code="INVALID_WARRANTY_TOKEN", message="Empty serial/IMEI.")

        if self._settings.aftersale_base_url.strip():
            data = await self._request(
                base_url=self._settings.aftersale_base_url,
                method="GET",
                path=f"/warranty/{token}",
            )
            return BusinessResult(status="ok", code="WARRANTY_OK", data=data)

        return BusinessResult(
            status="ok",
            code="WARRANTY_OK_MOCK",
            data={
                "token": token,
                "status": "in_warranty",
                "valid_until": "2027-12-31",
                "coverage": "主板、电池、屏幕（非人为）",
            },
            message="AFTERSALE_BASE_URL not configured; using mock data.",
        )

    async def create_after_sale_ticket(self, order_id: str, issue_type: str, details: str = "") -> BusinessResult:
        oid = order_id.strip().upper()
        issue = issue_type.strip()
        if not oid or not issue:
            return BusinessResult(
                status="error",
                code="INVALID_AFTERSALE_REQUEST",
                message="order_id and issue_type are required.",
            )

        if self._settings.aftersale_base_url.strip():
            data = await self._request(
                base_url=self._settings.aftersale_base_url,
                method="POST",
                path="/tickets",
                payload={"order_id": oid, "issue_type": issue, "details": details.strip()},
            )
            return BusinessResult(status="ok", code="AFTERSALE_TICKET_CREATED", data=data)

        return BusinessResult(
            status="ok",
            code="AFTERSALE_TICKET_CREATED_MOCK",
            data={
                "ticket_id": f"AS-{uuid.uuid4().hex[:12].upper()}",
                "order_id": oid,
                "issue_type": issue,
                "details": details.strip(),
                "status": "queued",
            },
            message="AFTERSALE_BASE_URL not configured; using mock data.",
        )