# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

京东第三方商家 AI 客服 agent：FastAPI 后端 + React SPA 前端。Agent 通过 `Context` payload 感知商品页上下文（shop_id / goods_id / order_sn / media 等），用混合检索（alias + keyword + vector）+ LLM 单轮工具调用回答售前/售中/售后问题。

## Commands

```bash
uv sync                                       # install backend deps
uv run uvicorn main:app --reload              # FastAPI :8000
uv run .venv/bin/pytest -q                    # tests
PGPASSWORD=pg2024 psql -h 127.0.0.1 -p 5434 -U postgres -d jd_agent -c "..."  # DB
cd frontend && npm install && npm run dev     # Vite :5173 (proxy /v1 -> :8000)
cd frontend && npm run build                  # build to frontend/dist/ (FastAPI serves it)
```

## Architecture

```
HTTP (FastAPI)
  -> routes_chat.py / routes_admin.py / routes_shop.py
  -> ChatOrchestrator (app/orchestrator/service.py)
    ├─ turn_context.py        # parse raw turn -> ProductCard / OrderCard / MediaInfo
    ├─ scene_classifier.py    # presale / insale / aftersale / mixed via orders table
    ├─ request_state.py       # RequestTracker state machine (per-request logging)
    ├─ agent_runtime.py       # LLM + bind_tools single-round (no ReAct loop)
    │    ├─ input_builder.py  # scene prompt + history compression + session-info
    │    └─ tools/search_knowledge.py   # hybrid retrieval (alias + keyword + vector)
    └─ session_store.py       # agent_messages table
```

`ChatOrchestrator.chat` branches on `Context.type` before LLM：silent（withdraw/system）-> 空答；image/video -> transfer_to_human；goods_card 无文字 -> 欢迎语；greeting -> 欢迎语跳过 LLM；否则走 scene classifier + agent。`main.py` lifespan 里 `prewarm()` 预热 embedding 模型和 DB pool。

**4 层空答兜底**（`agent_runtime.py::ask` / `ask_stream`）：L1 重试 + nudge -> L2 KB 检索 -> L3 默认回复 -> L4 空答不入历史。**3 层 API 错误兜底**（`_call_llm_ainvoke`）：4xx 直接 ClientError / 5xx 重试 2 次换 fallback 模型 / 网络错误指数退避重试 3 次。共享工具在 `app/llm/retry.py`。

**Pre-RAG 占位符填充**（`input_builder.py::_fill_db_placeholders`）：发 LLM 前把 prompt 里的 `[DB_SHOP_ADVANTAGES]` / `[DB_PRODUCT_KNOWLEDGE]` 占位符替换成 KB 数据，避免 LLM 为填占位符多余调 `search_knowledge`。`SearchKnowledge.fetch_shop_advantages` / `fetch_product_knowledge` 60s TTL 缓存。

## Endpoints

| 类型 | 路由 | 说明 |
|------|------|------|
| Chat | `POST /v1/chat` / `/v1/chat/stream` | 流式用 SSE（`fetch` + `ReadableStream`） |
| Public | `GET /v1/shops` / `/v1/products` | 前端表单用，slowapi IP 限流（30/min, 10/min） |
| Admin | `/v1/admin/*` | `routes_admin.py`，bearer token 鉴权 |

## Frontend

`frontend/` 是 Vite + React 18 + TS + Ant Design 5 SPA。Form-first 流程：`ConsultationForm` 收集店铺/商品/订单号 -> `ChatRoom` SSE 流式聊天。session_id（UUID）+ 消息历史存浏览器 `localStorage`。`npm run build` 产物在 `frontend/dist/`，FastAPI 用 `StaticFiles` 挂载到 `/`（同源，零 CORS）。

## Workflow Rules

- **Auto-sync docs**：改代码后评估并更新本文件，保持架构/命令/约定和代码一致。
- **Prompt injection 防御**：平台传入的值（shop_id / goods_name 等）必须过 `InputBuilder.safe_value()`（`<input>` 标签 + 双花括号转义），不能绕过。
- pgvector 扩展注册用 `pgvector.psycopg2.register.register_vector`（NOT `pgvector.sqlalchemy.psycopg2`）。

## File map

| Want to... | Edit |
|------------|------|
| Add a contextType | `app/context_models.py` -> service.py branch |
| Change scene routing | `app/orchestrator/scene_classifier.py` |
| Change scene prompt | `prompt/{scene}.md` |
| Add an agent tool | `app/tools/*.py` with `@agent_tool` |
| Add admin endpoint | `app/api/routes_admin.py` |
| Add public endpoint | `app/api/routes_shop.py`（rate-limited via `app/api/rate_limit.py`） |
| Tune fallback / API retry | `app/llm/agent_runtime.py` |
| Tune message building | `app/llm/input_builder.py` |
| Tune frontend UI | `frontend/src/components/*.tsx` |
| Add seed data | `script/seed_*.py` |
