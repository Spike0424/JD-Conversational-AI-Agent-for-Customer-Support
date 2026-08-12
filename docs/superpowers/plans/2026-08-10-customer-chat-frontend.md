# Customer Chat Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone React SPA (form-first chat page) that lets customers consult the AI agent, served by the existing FastAPI backend with new shop/product endpoints.

**Architecture:** React SPA (Vite + TypeScript + Ant Design) built to `frontend/dist/`, served as static files by FastAPI (same origin, zero CORS). Form-first flow: customer selects shop + product + optional order number, then chats via SSE streaming (`/v1/chat/stream`). Session ID (UUID) and message history persisted in browser localStorage. New backend endpoints (`GET /v1/shops`, `GET /v1/products`) added to existing FastAPI with IP rate limiting.

**Tech Stack:** Backend: FastAPI + SQLModel + slowapi (rate limiting). Frontend: Vite + React 18 + TypeScript + Ant Design 5 + fetch/ReadableStream (SSE). No BFF, no WebSocket.

---

## File Structure

### Backend (modify existing)
- `app/api/routes_shop.py` (CREATE) - `GET /v1/shops`, `GET /v1/products` endpoints
- `app/schemas.py` (MODIFY) - add `ShopInfo` / `ProductInfo` response models
- `main.py` (MODIFY) - register shop router, mount `frontend/dist` static files, configure rate limiter
- `pyproject.toml` (MODIFY) - add `slowapi` dependency
- `tests/test_routes_shop.py` (CREATE) - tests for shop/product endpoints

### Frontend (new `frontend/` directory)
- `frontend/package.json` (CREATE) - npm deps (react, antd, vite, typescript)
- `frontend/vite.config.ts` (CREATE) - Vite config + dev proxy to :8000
- `frontend/index.html` (CREATE) - HTML entry
- `frontend/tsconfig.json` (CREATE) - TS config
- `frontend/src/main.tsx` (CREATE) - React entry
- `frontend/src/App.tsx` (CREATE) - top-level component (form vs chat routing)
- `frontend/src/types.ts` (CREATE) - TS types matching backend schemas
- `frontend/src/api.ts` (CREATE) - API client (fetchShops, fetchProducts, streamChat)
- `frontend/src/hooks/useSession.ts` (CREATE) - UUID session ID + localStorage
- `frontend/src/hooks/useChatHistory.ts` (CREATE) - message list + localStorage persistence
- `frontend/src/components/ConsultationForm.tsx` (CREATE) - form-first selection UI
- `frontend/src/components/ChatRoom.tsx` (CREATE) - chat interface (message list + input + SSE)
- `frontend/src/components/MessageBubble.tsx` (CREATE) - single message render
- `frontend/src/components/ProductCardView.tsx` (CREATE) - render product_cards from ChatResponse

---

## Task 1: Backend - Add ShopInfo / ProductInfo schemas

**Files:**
- Modify: `app/schemas.py` (append at end)

- [ ] **Step 1: Add response models to `app/schemas.py`**

Append to the end of `app/schemas.py`:

```python
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
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from app.schemas import ShopInfo, ProductInfo; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add app/schemas.py
git commit -m "feat: add ShopInfo / ProductInfo schemas for chat frontend"
```

---

## Task 2: Backend - Create routes_shop.py with shops + products endpoints

**Files:**
- Create: `app/api/routes_shop.py`
- Test: `tests/test_routes_shop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_routes_shop.py`:

```python
"""Tests for public shop/product endpoints (chat frontend)."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def shop_client(monkeypatch: pytest.MonkeyPatch):
    """Client with mocked DB session returning canned shops/products."""
    from app.api import deps
    from main import app

    # Mock create_session to return a session with canned query results
    mock_session = MagicMock()
    mock_shop = MagicMock(id=1, shop_name="测试店铺", shop_logo=None, description="测试描述")
    mock_product = MagicMock(goods_id=57430876, goods_name="iPhone 17 Pro Max", price="9999", thumb_url="http://x/y.jpg")

    mock_session.query.return_value.filter.return_value.all.return_value = [mock_shop]
    mock_session.query.return_value.filter.return_value.filter.return_value.limit.return_value.all.return_value = [mock_product]

    monkeypatch.setattr("app.api.routes_shop.create_session", lambda: mock_session)
    return TestClient(app)


def test_list_shops(shop_client: TestClient) -> None:
    r = shop_client.get("/v1/shops")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["shop_name"] == "测试店铺"


def test_search_products(shop_client: TestClient) -> None:
    r = shop_client.get("/v1/products", params={"shop_id": 1, "q": "iPhone"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["goods_id"] == 57430876
    assert "iPhone" in data[0]["goods_name"]


def test_search_products_missing_shop_id(shop_client: TestClient) -> None:
    r = shop_client.get("/v1/products", params={"q": "iPhone"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run .venv/bin/pytest tests/test_routes_shop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.routes_shop'`

- [ ] **Step 3: Create `app/api/routes_shop.py`**

```python
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
```

- [ ] **Step 4: Register router in `main.py`**

In `main.py`, add the import and `include_router` call. The `main.py` top section should become:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import get_orchestrator
from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.api.routes_shop import router as shop_router
from util.logger_setup import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热 orchestrator + agent，避免首次请求承担冷启动开销。"""
    orchestrator = get_orchestrator()
    orchestrator._agent.prewarm()
    yield


app = FastAPI(title="LangChain ReAct QA Agent", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(shop_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run .venv/bin/pytest tests/test_routes_shop.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add app/api/routes_shop.py tests/test_routes_shop.py main.py
git commit -m "feat: add /v1/shops and /v1/products endpoints for chat frontend"
```

---

## Task 3: Backend - Add IP rate limiting with slowapi

**Files:**
- Modify: `pyproject.toml` (add slowapi dep)
- Modify: `main.py` (configure limiter)
- Modify: `app/api/routes_shop.py` (apply limit to `/v1/products`)
- Modify: `app/api/routes_chat.py` (apply limit to `/v1/chat/stream`)

- [ ] **Step 1: Add slowapi dependency**

In `pyproject.toml`, find the `dependencies = [...]` list and add `"slowapi>=0.1.9",` to it. Then run:

```bash
uv sync
```

Expected: slowapi installed without errors.

- [ ] **Step 2: Configure limiter in `main.py`**

Modify `main.py` to add the limiter. Add these imports after the existing ones:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
```

Add the limiter instance after `setup_logging()`:

```python
limiter = Limiter(key_func=get_remote_address)
```

Inside the `lifespan` function is NOT where we add middleware. After `app = FastAPI(...)` and the `include_router` calls, add:

```python
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

The final `main.py` should look like:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.deps import get_orchestrator
from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.api.routes_shop import router as shop_router
from util.logger_setup import setup_logging

setup_logging()

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热 orchestrator + agent，避免首次请求承担冷启动开销。"""
    orchestrator = get_orchestrator()
    orchestrator._agent.prewarm()
    yield


app = FastAPI(title="LangChain ReAct QA Agent", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(shop_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 3: Apply rate limit to `/v1/products`**

In `app/api/routes_shop.py`, add imports and decorator. The `search_products` function should become:

```python
from fastapi import APIRouter, Query, Request
from slowapi import Limiter
# ... other imports stay the same

@router.get("/v1/products", response_model=list[ProductInfo])
@limiter.limit("30/minute")
def search_products(
    request: Request,
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
```

Note: `request: Request` must be the first parameter for slowapi to work. The `limiter` must be imported from `main` — to avoid circular import, use a local import inside the function:

Actually, the cleaner pattern is to define `limiter` in a separate module. Create `app/api/rate_limit.py`:

```python
"""Shared rate limiter instance (avoids circular imports with main.py)."""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
```

Then in `main.py`, import from there:

```python
from app.api.rate_limit import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# ... after app = FastAPI(...)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

And in `routes_shop.py` / `routes_chat.py`:

```python
from app.api.rate_limit import limiter

@router.get("/v1/products")
@limiter.limit("30/minute")
def search_products(request: Request, ...):
    ...
```

- [ ] **Step 4: Apply rate limit to `/v1/chat/stream`**

In `app/api/routes_chat.py`, add the import at the top:

```python
from app.api.rate_limit import limiter
```

And update the `chat_stream` endpoint to add the decorator and `request: Request` param. Find the `@router.post("/v1/chat/stream")` block and update it:

```python
@router.post("/chat/stream")
@router.post("/v1/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    return await _streaming_response_for_question(body.session_id, body.question, body.context, orchestrator)
```

Do the same for `chat_stream_get` if it exists.

- [ ] **Step 5: Verify existing tests still pass**

Run: `uv run .venv/bin/pytest tests/test_routes_shop.py tests/test_api_smoke.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add app/api/rate_limit.py app/api/routes_shop.py app/api/routes_chat.py main.py pyproject.toml uv.lock
git commit -m "feat: add IP rate limiting (30/min products, 10/min chat) via slowapi"
```

---

## Task 4: Frontend - Scaffold Vite + React + TypeScript project

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/index.html`, `frontend/tsconfig.json`, `frontend/src/main.tsx`, `frontend/src/App.tsx`

- [ ] **Step 1: Scaffold the Vite project**

Run from repo root:

```bash
mkdir -p frontend
cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install antd
```

If `npm create vite` asks to confirm the directory is not empty, choose "Ignore files and continue".

- [ ] **Step 2: Configure Vite dev proxy to FastAPI**

Overwrite `frontend/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/v1': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
```

- [ ] **Step 3: Minimal App.tsx to verify it runs**

Overwrite `frontend/src/App.tsx`:

```tsx
function App() {
  return (
    <div style={{ padding: 24, fontFamily: 'system-ui' }}>
      <h1>客服聊天页（脚手架）</h1>
      <p>后端 API: <a href="/v1/shops">/v1/shops</a></p>
    </div>
  )
}

export default App
```

- [ ] **Step 4: Verify dev server runs**

Run (in `frontend/`):
```bash
npm run dev
```
Expected: Vite dev server starts on `http://localhost:5173`. Open it in browser, see the heading. The `/v1/shops` link should return JSON (proxied to FastAPI on :8000 — start FastAPI in another terminal with `uv run uvicorn main:app --reload` first).

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold Vite + React + TS + Ant Design frontend"
```

---

## Task 5: Frontend - TypeScript types matching backend schemas

**Files:**
- Create: `frontend/src/types.ts`

- [ ] **Step 1: Write the types file**

Create `frontend/src/types.ts`:

```typescript
// Backend schemas mirrored for the frontend (keep in sync with app/schemas.py + app/context_models.py)

export interface ShopInfo {
  id: number
  shop_name: string
  shop_logo: string | null
  description: string | null
}

export interface ProductInfo {
  goods_id: number
  goods_name: string
  price: string | null
  thumb_url: string | null
}

export type ContextType =
  | 'text' | 'image' | 'video' | 'goods_card' | 'withdraw'
  | 'auth' | 'system_biz' | 'system_status' | 'mall_system_msg'

export interface ChatContext {
  type: ContextType
  content?: string
  kwargs: {
    shop_id?: string | number
    shop_name?: string
    goods_id?: number
    goods_name?: string
    order_sn?: string
    user_id?: string
    from_uid?: string
    recipient_uid?: string
    media_url?: string
    media_type?: string
    channel_type?: string
  }
}

export interface ProductCard {
  session_id: string
  shop_id: number
  goods_id: number
  goods_name: string
  price: string | null
  price_min: number | null
  price_max: number | null
  thumb_url: string | null
  specifications: Record<string, string>
  message: string
}

export interface ChatResponse {
  session_id: string
  answer: string
  trace_id: string | null
  intent: string
  context_type: string
  citations: unknown[]
  actions: string[]
  need_handoff: boolean
  product_cards: ProductCard[] | null
  metadata: { scene: string; scene_label: string } | null
}

export type ChatMessage =
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string; product_cards?: ProductCard[] }
```

- [ ] **Step 2: Verify TS compiles**

Run (in `frontend/`):
```bash
npx tsc --noEmit
```
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types.ts
git commit -m "feat: add TypeScript types matching backend schemas"
```

---

## Task 6: Frontend - API client (fetchShops, fetchProducts, streamChat)

**Files:**
- Create: `frontend/src/api.ts`

- [ ] **Step 1: Write the API client**

Create `frontend/src/api.ts`:

```typescript
import type { ChatContext, ChatMessage, ProductInfo, ShopInfo } from './types'

const API_BASE = ''  // same-origin in production; Vite proxy in dev

export async function fetchShops(): Promise<ShopInfo[]> {
  const r = await fetch(`${API_BASE}/v1/shops`)
  if (!r.ok) throw new Error(`fetchShops failed: ${r.status}`)
  return r.json()
}

export async function fetchProducts(shopId: number, q: string): Promise<ProductInfo[]> {
  const params = new URLSearchParams({ shop_id: String(shopId), q })
  const r = await fetch(`${API_BASE}/v1/products?${params}`)
  if (!r.ok) throw new Error(`fetchProducts failed: ${r.status}`)
  return r.json()
}

/**
 * Stream a chat turn via SSE (POST /v1/chat/stream).
 * Calls onChunk for each delta token, returns the full answer + product_cards.
 */
export async function streamChat(opts: {
  sessionId: string
  question: string
  context: ChatContext
  history: ChatMessage[]
  onChunk: (delta: string) => void
  signal?: AbortSignal
}): Promise<{ answer: string; productCards: ChatMessage extends { product_cards?: infer P } ? P : never[] }> {
  const { sessionId, question, context, onChunk, signal } = opts

  const r = await fetch(`${API_BASE}/v1/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question, context }),
    signal,
  })

  if (!r.ok || !r.body) {
    throw new Error(`streamChat failed: ${r.status}`)
  }

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let answer = ''
  let productCards: any[] = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE: lines start with "data: ", separated by "\n\n"
    const events = buffer.split('\n\n')
    buffer = events.pop() || ''

    for (const evt of events) {
      const line = evt.trim()
      if (!line.startsWith('data: ')) continue
      const payload = line.slice(6)
      if (payload === '[DONE]') continue
      try {
        const msg = JSON.parse(payload)
        if (msg.delta) {
          answer += msg.delta
          onChunk(msg.delta)
        }
        if (msg.error_code) {
          throw new Error(msg.message || 'stream error')
        }
      } catch (e) {
        // partial JSON in buffer, skip — will be completed in next chunk
      }
    }
  }

  // Note: product_cards come via the non-streaming ChatResponse when meta is sent.
  // For v1, we capture them from the meta event if the backend includes them.
  return { answer, productCards }
}
```

Note: The backend's `event_stream()` in `routes_chat.py` sends `meta` (with trace_id/intent/citations/actions) first, then `delta` chunks, then `[DONE]`. Product cards are returned in the `ChatResponse.product_cards` field of the non-streaming `/v1/chat` endpoint, but the streaming `/v1/chat/stream` endpoint does NOT currently include them in the SSE stream. For v1, product cards will be empty in streaming mode — this is a known limitation. The chat UI can still display the answer text.

- [ ] **Step 2: Verify TS compiles**

Run (in `frontend/`):
```bash
npx tsc --noEmit
```
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat: add API client (fetchShops, fetchProducts, streamChat via SSE)"
```

---

## Task 7: Frontend - useSession hook (UUID + localStorage)

**Files:**
- Create: `frontend/src/hooks/useSession.ts`

- [ ] **Step 1: Write the hook**

Create `frontend/src/hooks/useSession.ts`:

```typescript
import { useCallback, useState } from 'react'

const SESSION_KEY = 'chat_session_id'

function generateId(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  // Fallback for older browsers
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function useSession() {
  const [sessionId, setSessionId] = useState<string>(() => {
    const existing = localStorage.getItem(SESSION_KEY)
    if (existing) return existing
    const id = generateId()
    localStorage.setItem(SESSION_KEY, id)
    return id
  })

  const newSession = useCallback(() => {
    const id = generateId()
    localStorage.setItem(SESSION_KEY, id)
    setSessionId(id)
  }, [])

  return { sessionId, newSession }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useSession.ts
git commit -m "feat: add useSession hook (UUID + localStorage)"
```

---

## Task 8: Frontend - useChatHistory hook (messages + localStorage)

**Files:**
- Create: `frontend/src/hooks/useChatHistory.ts`

- [ ] **Step 1: Write the hook**

Create `frontend/src/hooks/useChatHistory.ts`:

```typescript
import { useCallback, useEffect, useState } from 'react'
import type { ChatMessage } from '../types'

const MESSAGES_KEY = 'chat_messages'

export function useChatHistory() {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(MESSAGES_KEY)
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  useEffect(() => {
    localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages))
  }, [messages])

  const appendUser = useCallback((content: string) => {
    setMessages((prev) => [...prev, { role: 'user', content }])
  }, [])

  const appendAssistantChunk = useCallback((delta: string) => {
    setMessages((prev) => {
      if (prev.length === 0 || prev[prev.length - 1].role !== 'assistant') {
        return [...prev, { role: 'assistant', content: delta }]
      }
      const last = prev[prev.length - 1]
      return [...prev.slice(0, -1), { ...last, content: last.content + delta }]
    })
  }, [])

  const clear = useCallback(() => {
    setMessages([])
    localStorage.removeItem(MESSAGES_KEY)
  }, [])

  return { messages, appendUser, appendAssistantChunk, clear }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useChatHistory.ts
git commit -m "feat: add useChatHistory hook (messages + localStorage persistence)"
```

---

## Task 9: Frontend - ConsultationForm component

**Files:**
- Create: `frontend/src/components/ConsultationForm.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/ConsultationForm.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { Button, Card, Form, Input, message as antdMessage, Select, Typography } from 'antd'
import { fetchProducts, fetchShops } from '../api'
import type { ChatContext, ProductInfo, ShopInfo } from '../types'

const { Title, Text } = Typography

export interface FormState {
  shopId: number
  goodsId: number
  goodsName: string
  orderSn: string
}

interface Props {
  onStart: (state: FormState) => void
}

export function ConsultationForm({ onStart }: Props) {
  const [shops, setShops] = useState<ShopInfo[]>([])
  const [products, setProducts] = useState<ProductInfo[]>([])
  const [shopId, setShopId] = useState<number | null>(null)
  const [productQuery, setProductQuery] = useState('')
  const [selectedGoodsId, setSelectedGoodsId] = useState<number | null>(null)
  const [selectedGoodsName, setSelectedGoodsName] = useState('')
  const [orderSn, setOrderSn] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchShops()
      .then(setShops)
      .catch(() => antdMessage.error('店铺列表加载失败'))
  }, [])

  useEffect(() => {
    if (shopId === null) return
    if (!productQuery.trim()) {
      fetchProducts(shopId, '').then(setProducts).catch(() => antdMessage.error('商品搜索失败'))
      return
    }
    const timer = setTimeout(() => {
      fetchProducts(shopId, productQuery).then(setProducts).catch(() => antdMessage.error('商品搜索失败'))
    }, 300)
    return () => clearTimeout(timer)
  }, [shopId, productQuery])

  const handleSubmit = () => {
    if (shopId === null) {
      antdMessage.warning('请选择店铺')
      return
    }
    if (selectedGoodsId === null) {
      antdMessage.warning('请选择商品')
      return
    }
    onStart({
      shopId,
      goodsId: selectedGoodsId,
      goodsName: selectedGoodsName,
      orderSn: orderSn.trim(),
    })
  }

  return (
    <Card style={{ maxWidth: 640, margin: '40px auto' }}>
      <Title level={3}>客服咨询</Title>
      <Text type="secondary">请先选择您要咨询的商品，方便客服为您提供准确信息。</Text>

      <Form layout="vertical" style={{ marginTop: 24 }} onFinish={handleSubmit}>
        <Form.Item label="店铺" required>
          <Select
            placeholder="选择店铺"
            value={shopId}
            onChange={(v) => setShopId(v)}
            options={shops.map((s) => ({ value: s.id, label: s.shop_name }))}
          />
        </Form.Item>

        <Form.Item label="商品" required>
          <Input.Search
            placeholder="搜索商品名称"
            value={productQuery}
            onChange={(e) => setProductQuery(e.target.value)}
            enterButton
            style={{ marginBottom: 8 }}
          />
          <Select
            placeholder="选择咨询的商品"
            value={selectedGoodsId}
            onChange={(v, option) => {
              setSelectedGoodsId(v)
              setSelectedGoodsName(option?.label as string || '')
            }}
            options={products.map((p) => ({
              value: p.goods_id,
              label: `${p.goods_name}${p.price ? `（¥${p.price}）` : ''}`,
            }))}
            showSearch
            optionFilterProp="label"
          />
        </Form.Item>

        <Form.Item label="订单号（选填，售后问题请填写）">
          <Input
            placeholder="请输入京东订单号"
            value={orderSn}
            onChange={(e) => setOrderSn(e.target.value)}
          />
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block size="large">
            开始咨询
          </Button>
        </Form.Item>
      </Form>
    </Card>
  )
}

export function formStateToContext(state: FormState): ChatContext {
  return {
    type: 'text',
    kwargs: {
      shop_id: state.shopId,
      goods_id: state.goodsId,
      goods_name: state.goodsName,
      order_sn: state.orderSn || undefined,
    },
  }
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ConsultationForm.tsx
git commit -m "feat: add ConsultationForm (shop/product/order selection)"
```

---

## Task 10: Frontend - MessageBubble + ProductCardView components

**Files:**
- Create: `frontend/src/components/MessageBubble.tsx`
- Create: `frontend/src/components/ProductCardView.tsx`

- [ ] **Step 1: Write ProductCardView**

Create `frontend/src/components/ProductCardView.tsx`:

```tsx
import { Card, Image, Typography } from 'antd'
import type { ProductCard } from '../types'

const { Text } = Typography

export function ProductCardView({ card }: { card: ProductCard }) {
  return (
    <Card
      size="small"
      style={{ maxWidth: 320, marginTop: 8 }}
      cover={card.thumb_url ? <Image src={card.thumb_url} alt={card.goods_name} style={{ maxHeight: 200, objectFit: 'cover' }} /> : null}
    >
      <Card.Meta
        title={card.goods_name}
        description={
          <>
            {card.price && <Text strong>¥{card.price}</Text>}
            {card.message && <div style={{ marginTop: 4, fontSize: 13 }}>{card.message}</div>}
          </>
        }
      />
      {card.specifications && Object.keys(card.specifications).length > 0 && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
          {Object.entries(card.specifications).slice(0, 4).map(([k, v]) => (
            <div key={k}>{k}: {v}</div>
          ))}
        </div>
      )}
    </Card>
  )
}
```

- [ ] **Step 2: Write MessageBubble**

Create `frontend/src/components/MessageBubble.tsx`:

```tsx
import { Typography } from 'antd'
import type { ChatMessage } from '../types'
import { ProductCardView } from './ProductCardView'

export function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 12,
    }}>
      <div style={{
        maxWidth: '70%',
        padding: '8px 14px',
        borderRadius: 8,
        background: isUser ? '#1677ff' : '#f0f0f0',
        color: isUser ? '#fff' : '#000',
      }}>
        <Typography.Text style={{ color: 'inherit', whiteSpace: 'pre-wrap' }}>
          {msg.content}
        </Typography.Text>
        {!isUser && 'product_cards' in msg && msg.product_cards?.map((c, i) => (
          <ProductCardView key={i} card={c} />
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Verify TS compiles**

Run: `npx tsc --noEmit`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/MessageBubble.tsx frontend/src/components/ProductCardView.tsx
git commit -m "feat: add MessageBubble + ProductCardView components"
```

---

## Task 11: Frontend - ChatRoom component (SSE streaming + input)

**Files:**
- Create: `frontend/src/components/ChatRoom.tsx`

- [ ] **Step 1: Write the component**

Create `frontend/src/components/ChatRoom.tsx`:

```tsx
import { useRef, useState } from 'react'
import { Button, Input, Space, Spin, Typography, message as antdMessage } from 'antd'
import { streamChat } from '../api'
import { useChatHistory } from '../hooks/useChatHistory'
import type { ChatContext } from '../types'
import { MessageBubble } from './MessageBubble'

const { Text } = Typography

interface Props {
  sessionId: string
  context: ChatContext
  goodsName: string
  onNewConversation: () => void
}

export function ChatRoom({ sessionId, context, goodsName, onNewConversation }: Props) {
  const { messages, appendUser, appendAssistantChunk, clear } = useChatHistory()
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const send = async () => {
    const question = input.trim()
    if (!question || streaming) return
    setInput('')
    appendUser(question)
    setStreaming(true)

    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      await streamChat({
        sessionId,
        question,
        context,
        history: messages,
        onChunk: (delta) => {
          appendAssistantChunk(delta)
          scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
        },
        signal: ctrl.signal,
      })
    } catch (e: any) {
      if (e.name === 'AbortError') {
        antdMessage.info('已停止生成')
      } else {
        appendAssistantChunk(`\n\n[出错：${e.message}]`)
        antdMessage.error('请求失败：' + e.message)
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  const stop = () => {
    abortRef.current?.abort()
  }

  const handleNewConversation = () => {
    if (streaming) abortRef.current?.abort()
    clear()
    onNewConversation()
  }

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #f0f0f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Text strong>当前咨询：{goodsName}</Text>
        <Button onClick={handleNewConversation} size="small">新对话</Button>
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {messages.length === 0 && (
          <Text type="secondary">请输入您的问题，例如：{goodsName}有什么优势？</Text>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} msg={m} />
        ))}
        {streaming && <Spin size="small" style={{ marginLeft: 16 }} />}
      </div>

      <div style={{ padding: 16, borderTop: '1px solid #f0f0f0' }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input.TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="输入您的问题..."
            autoSize={{ minRows: 1, maxRows: 4 }}
            onPressEnter={(e) => {
              if (!e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            disabled={streaming}
          />
          {streaming ? (
            <Button danger onClick={stop} style={{ width: 100 }}>停止</Button>
          ) : (
            <Button type="primary" onClick={send} style={{ width: 100 }}>发送</Button>
          )}
        </Space.Compact>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TS compiles**

Run: `npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatRoom.tsx
git commit -m "feat: add ChatRoom (SSE streaming + input + abort + new conversation)"
```

---

## Task 12: Frontend - App.tsx wiring (form ↔ chat routing)

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Overwrite App.tsx**

```tsx
import { useState } from 'react'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { ChatRoom } from './components/ChatRoom'
import { ConsultationForm, formStateToContext, type FormState } from './components/ConsultationForm'
import { useSession } from './hooks/useSession'

function App() {
  const { sessionId, newSession } = useSession()
  const [formState, setFormState] = useState<FormState | null>(null)

  if (!formState) {
    return (
      <ConfigProvider locale={zhCN}>
        <ConsultationForm onStart={(s) => setFormState(s)} />
      </ConfigProvider>
    )
  }

  return (
    <ConfigProvider locale={zhCN}>
      <ChatRoom
        sessionId={sessionId}
        context={formStateToContext(formState)}
        goodsName={formState.goodsName}
        onNewConversation={() => {
          newSession()
          setFormState(null)
        }}
      />
    </ConfigProvider>
  )
}

export default App
```

- [ ] **Step 2: Verify dev server renders the form**

Run FastAPI in one terminal:
```bash
uv run uvicorn main:app --reload
```

Run Vite in another terminal:
```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`. Expected: see the "客服咨询" form with shop selector + product search + order input + "开始咨询" button.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: wire App.tsx (form-first → chat room routing)"
```

---

## Task 13: Integration - FastAPI serves frontend static files

**Files:**
- Modify: `main.py` (mount StaticFiles after all API routes)

- [ ] **Step 1: Add StaticFiles import and mount in main.py**

In `main.py`, add import at top:

```python
from pathlib import Path

from fastapi.staticfiles import StaticFiles
```

After all `app.include_router(...)` calls and the `health` endpoint, add:

```python
# Serve frontend SPA (must be after all /v1/* routes to avoid shadowing them)
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
```

The `main.py` should end with:

```python
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve frontend SPA (must be after all /v1/* routes to avoid shadowing them)
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
```

- [ ] **Step 2: Build the frontend**

```bash
cd frontend && npm run build
```

Expected: `frontend/dist/` directory created with `index.html` + `assets/` folder.

- [ ] **Step 3: Verify FastAPI serves the SPA**

```bash
uv run uvicorn main:app
```

Open `http://localhost:8000/` in browser. Expected: see the "客服咨询" form (not a 404). Verify `http://localhost:8000/v1/shops` still returns JSON (not the SPA).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: FastAPI serves frontend/dist as static SPA (same-origin, zero CORS)"
```

---

## Task 14: E2E verification - full chat flow

**Files:**
- No new files — this is manual verification.

- [ ] **Step 1: Start the backend**

```bash
uv run uvicorn main:app --reload
```

Expected: server starts on :8000, logs "Agent pre-warm complete".

- [ ] **Step 2: Verify form-first flow**

Open `http://localhost:8000/` in browser.

- See the "客服咨询" form.
- Shop selector loads (if DB has shops).
- Type "iPhone" in product search -> see results.
- Select a product.
- Leave order number empty.
- Click "开始咨询".

Expected: form disappears, chat room appears with "当前咨询：iPhone 17 Pro Max" header.

- [ ] **Step 3: Verify SSE streaming**

In the chat room:
- Type "请问这手机有什么优势吗"
- Press Enter (or click 发送).
- See user message bubble appear immediately.
- See assistant bubble start streaming token-by-token.
- See "停止" button while streaming; click it -> generation stops.
- See "新对话" button -> click it -> returns to form, clears messages.

- [ ] **Step 4: Verify error handling**

Stop the backend (Ctrl+C in the uvicorn terminal). In the browser, type a message and send.

Expected: error toast "请求失败：..." appears, no infinite spinner. Restart backend.

- [ ] **Step 5: Verify rate limiting**

```bash
for i in $(seq 1 35); do curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8000/v1/products?shop_id=7&q=iPhone"; done
```

Expected: first 30 return 200, then 429 (Too Many Requests).

- [ ] **Step 6: Final commit (if any fixups needed)**

If any issues were found and fixed during verification, commit them. Otherwise no commit needed.

---

## Self-Review

**Spec coverage:**
- ✅ SSE streaming (Task 6, 11) — uses existing `/v1/chat/stream`
- ✅ Standalone React SPA (Task 4)
- ✅ Form-first flow: shop/product/order selection (Task 9)
- ✅ Product search box (Task 2 backend, Task 9 frontend)
- ✅ Order number plain text input (Task 9)
- ✅ New endpoints in FastAPI, no BFF (Task 2)
- ✅ UUID + localStorage session (Task 7)
- ✅ Vite + React + TS + Ant Design (Task 4) + hooks/Context (Task 7, 8) + fetch/ReadableStream (Task 6)
- ✅ FastAPI serves static files, same-origin (Task 13)
- ✅ IP rate limiting on /v1/products + /v1/chat/stream (Task 3)
- ✅ Frontend localStorage for chat history (Task 8)
- ✅ `frontend-design` skill — not invoked in plan; user can invoke during Task 9-11 for UI polish

**Placeholder scan:** No TBD/TODO. All code blocks are complete.

**Type consistency:** `FormState` defined in Task 9, used in Task 12. `ChatContext` defined in Task 5, used in Task 6 + 9 + 11 + 12. `ChatMessage` defined in Task 5, used in Task 6 + 8 + 10 + 11.
